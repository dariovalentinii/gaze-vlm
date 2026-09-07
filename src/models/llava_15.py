from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import json

import torch
from PIL import Image
import numpy as np

from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from src.models.unwrapping import unwrap_to_llava
from src.data.heatmaps import (
    heatmaps_to_rgb_pils,
    heatmap_to_patch_weights,
    apply_patch_weighting,
    GazeInjector,
    DualEncodingGazeInjector,
)
from src.data.prompts import build_prompt_texts
from src.inference.runner import GenerationConfig


class LlavaHFAdapter:
    """
    Minimal adapter for inference using HuggingFace LlavaForConditionalGeneration.
    To be used with
    llava-hf/llava-1.5-7b-hf
    llava-hf/llava-1.5-13b-hf
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        lora_dir: Optional[str] = None,
        gaze_injector_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.model_dtype = torch_dtype
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=None,
        ).to(self.device)

        # Vision config
        self.vision_patch_size = getattr(self.model.config.vision_config, "patch_size", 14)
        self.image_size = getattr(self.model.config.vision_config, "image_size", None)  # often 336

        # Optional learned modules
        self.gaze_injector: Optional[GazeInjector] = None
        self.dual_encoding_injector: Optional[DualEncodingGazeInjector] = None
        self.heatmap_encoder = None
        self._heatmap_hook_handle = None
        self._heatmap_layer_out: Optional[torch.Tensor] = None

        # Resolve LoRA/checkpoint layout (current LGG/DE format):
        # - run folder containing projector_lora/ (optional), gaze_injector.pt, components.json
        # - projector_lora folder directly
        self.checkpoint_dir: Optional[Path] = None
        projector_lora_dir: Optional[Path] = None
        if lora_dir:
            projector_lora_dir, self.checkpoint_dir = self._resolve_lora_layout(Path(lora_dir))
            if projector_lora_dir is not None:
                self.model = PeftModel.from_pretrained(self.model, str(projector_lora_dir)).to(self.device)

        # Explicit gaze_injector_dir has priority; otherwise use checkpoint_dir
        inj_dir = Path(gaze_injector_dir) if gaze_injector_dir else self.checkpoint_dir
        if inj_dir is not None:
            self._load_gaze_injector(inj_dir)

        # Dual Encoding heatmap encoder LoRA
        if self.checkpoint_dir is not None:
            self._load_heatmap_encoder(self.checkpoint_dir)

        self.model.eval()


    def _resolve_lora_layout(self, path: Path) -> tuple[Optional[Path], Optional[Path]]:
        """
        Returns:
          (projector_lora_dir, checkpoint_root_dir)

        Accepted inputs:
          - LGG/DE run dir (contains projector_lora/ or lora_adapter/)
          - LGG/DE projector dir itself (.../projector_lora)
          - legacy LLM adapter directory (.../lora_adapter)
          - LGG/DE run dir without projector LoRA (injector-only): no projector_lora/
        """
        if not path.exists():
            raise FileNotFoundError(f"LoRA path not found: {path}")

        if path.is_dir() and path.name == "projector_lora" and (path / "adapter_config.json").exists():
            print(f"[LlavaHFAdapter] Loaded projector LoRA from: {path}")
            return path, path.parent

        if path.is_dir() and path.name == "lora_adapter" and (path / "adapter_config.json").exists():
            print(f"[LlavaHFAdapter] Loaded lora_adapter from: {path}")
            return path, path.parent

        candidate = path / "projector_lora"
        if candidate.exists() and (candidate / "adapter_config.json").exists():
            print(f"[LlavaHFAdapter] Loaded projector LoRA from: {candidate}")
            return candidate, path

        # lora_adapter candidate (LLM [+ proj])
        candidate = path / "lora_adapter"
        if candidate.exists() and (candidate / "adapter_config.json").exists():
            print(f"[LlavaHFAdapter] Loaded lora_adapter from: {candidate}")
            return candidate, path

        # Injector-only run dir (no projector LoRA or adapter)
        print(f"[LlavaHFAdapter] No LoRA adapters found in: {path}. Checking for injector-only checkpoint...")
        if path.is_dir() and ((path / "gaze_injector.pt").exists() or (path / "components.json").exists()):
            return None, path

        raise ValueError(
            f"Unsupported checkpoint layout at: {path}. "
            "Expected a run folder containing projector_lora/ or lora_adapter/ and/or gaze_injector.pt, "
            "or a direct projector_lora/ or lora_adapter/ folder."
        )


    def _read_injector_min_gate(self, inj_dir: Path) -> float:
        min_gate = 0.05

        # LGG/DE sidecar
        comp = inj_dir / "components.json"
        if comp.exists():
            try:
                meta = json.loads(comp.read_text(encoding="utf-8"))
                inj = meta.get("injector", {})
                if isinstance(inj, dict) and "min_gate" in inj:
                    return float(inj["min_gate"])
            except Exception:
                pass

        return min_gate
        
        
    def _load_gaze_injector(self, inj_dir: Path) -> None:
        pt = inj_dir / "gaze_injector.pt"
        if not pt.exists():
            return

        min_gate = self._read_injector_min_gate(inj_dir)
        sd = torch.load(pt, map_location="cpu")

        # Dual Encoding injector state has projection/norm params and works on [B,N,D].
        is_dual_encoding = any(k.startswith("proj.") or k.startswith("norm.") for k in sd.keys())
        if is_dual_encoding:
            d_model = int(getattr(self.model.config.vision_config, "hidden_size", 1024))
            dual_encoding_injector = DualEncodingGazeInjector(d_model=d_model, min_gate=min_gate)
            dual_encoding_injector.load_state_dict(sd, strict=True)
            self.dual_encoding_injector = dual_encoding_injector.to(self.device).eval()
            self.gaze_injector = None
            print(f"[LlavaHFAdapter] Loaded Dual Encoding gaze injector from: {pt}")
            return

        injector = GazeInjector(min_gate=min_gate)
        injector.load_state_dict(sd, strict=True)
        self.gaze_injector = injector.to(self.device).eval()
        self.dual_encoding_injector = None
        print(f"[LlavaHFAdapter] Loaded gaze injector from: {pt}")


    def _clone_vision_tower(self, vision_tower) -> Any:
        cls = vision_tower.__class__
        cloned = cls(vision_tower.config)
        cloned.load_state_dict(vision_tower.state_dict(), strict=True)
        return cloned


    def _vision_layer_to_encoder_block_idx(self, layer_idx: Optional[int], n_layers: int) -> Optional[int]:
        if layer_idx is None or n_layers <= 0:
            return None
        hs_len = n_layers + 1
        hs_idx = layer_idx if layer_idx >= 0 else (hs_len + layer_idx)
        if hs_idx <= 0 or hs_idx > n_layers:
            return None
        return hs_idx - 1


    def _find_encoder_block_for_hook(self, module_root: Any, block_idx: int) -> Optional[Any]:
        suffix = f"vision_model.encoder.layers.{block_idx}"
        for name, mod in module_root.named_modules():
            if name.endswith(suffix):
                return mod
        return None


    def _vision_layer_requires_hidden_states(self, layer_idx: Optional[int], n_layers: int) -> bool:
        if layer_idx is None:
            return False
        if layer_idx == -1:
            return False
        if n_layers > 0 and layer_idx == n_layers:
            return False
        return True


    def _load_heatmap_encoder(self, ckpt_dir: Path) -> None:
        hm_lora_dir = ckpt_dir / "heatmap_encoder_lora"
        if not hm_lora_dir.exists():
            return

        core = unwrap_to_llava(self.model)
        img_vision = core.model.vision_tower
        heatmap_encoder_base = self._clone_vision_tower(img_vision).to(self.device)
        self.heatmap_encoder = PeftModel.from_pretrained(heatmap_encoder_base, str(hm_lora_dir)).to(self.device)
        self.heatmap_encoder.eval()

        # Mirror train_de logic: when possible, capture the selected vision layer
        # via hook to avoid requesting full hidden_states.
        vision_layer = getattr(self.model.config, "vision_feature_layer", None)
        num_vision_layers = int(getattr(self.model.config.vision_config, "num_hidden_layers", 0) or 0)
        hm_block_idx = self._vision_layer_to_encoder_block_idx(vision_layer, num_vision_layers)
        if hm_block_idx is not None:
            hm_block = self._find_encoder_block_for_hook(self.heatmap_encoder, hm_block_idx)
            if hm_block is not None:
                def _heatmap_block_hook(_module, _inp, out):
                    self._heatmap_layer_out = out[0] if isinstance(out, (tuple, list)) else out

                self._heatmap_hook_handle = hm_block.register_forward_hook(_heatmap_block_hook)

        print(f"[LlavaHFAdapter] Loaded heatmap encoder LoRA from: {hm_lora_dir}")


    # @torch.no_grad()
    # def _heatmaps_to_rgb_pils(self, heatmaps: list[torch.Tensor]) -> list[Image.Image]:
    #     heatmap_pils: list[Image.Image] = []
    #     for hm in heatmaps:
    #         assert hm.ndim == 4 and hm.shape[0] == 1 and hm.shape[1] == 1
    #         hm_2d = hm[0, 0].detach().float().cpu().numpy()
    #         hm_2d = hm_2d - hm_2d.min()
    #         if hm_2d.max() > 0:
    #             hm_2d = hm_2d / hm_2d.max()
    #         hm_uint8 = (hm_2d * 255.0).astype(np.uint8)
    #         heatmap_pils.append(Image.fromarray(hm_uint8, mode="L").convert("RGB"))
    #     return heatmap_pils


    def _forward_heatmap_encoder(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> Any:
        if self.heatmap_encoder is None:
            raise RuntimeError("Heatmap encoder is not loaded.")

        kwargs = {
            "pixel_values": pixel_values,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }

        if isinstance(self.heatmap_encoder, PeftModel):
            try:
                return self.heatmap_encoder(**kwargs)
            except (TypeError, KeyError) as e:
                if "inputs_embeds" not in str(e):
                    raise
                if not hasattr(self.heatmap_encoder, "base_model"):
                    raise
                return self.heatmap_encoder.base_model(**kwargs)
        return self.heatmap_encoder(**kwargs)


    @torch.no_grad()
    def _embed_heatmaps_for_dual_encoding(self, heatmaps: list[torch.Tensor]) -> torch.Tensor:
        """
        Dual Encoding preprocessing:
          - convert heatmaps to RGB PIL
          - process like images for heatmap encoder
          - encode with heatmap vision tower (possibly LoRA-finetuned)
          - return patch embeddings [B,N,D]
        """
        if self.heatmap_encoder is None:
            raise RuntimeError("Dual Encoding inference requested but heatmap encoder is not loaded.")

        heatmap_pils = heatmaps_to_rgb_pils(heatmaps)
        hm_pix = self.processor.image_processor(
            images=heatmap_pils,
            return_tensors="pt"
        )["pixel_values"].to(self.device)

        vision_layer = getattr(self.model.config, "vision_feature_layer", None)
        num_layers = int(getattr(self.model.config.vision_config, "num_hidden_layers", 0) or 0)
        need_hidden_states = self._vision_layer_requires_hidden_states(vision_layer, num_layers) and self._heatmap_hook_handle is None

        self._heatmap_layer_out = None
        hm_out = self._forward_heatmap_encoder(
            hm_pix,
            output_hidden_states=need_hidden_states
        )
        if self._heatmap_layer_out is not None:
            hm_feats_all = self._heatmap_layer_out
        elif need_hidden_states:
            hm_feats_all = hm_out.hidden_states[vision_layer]
        else:
            hm_feats_all = hm_out.last_hidden_state

        self._heatmap_layer_out = None
        hm_feats = hm_feats_all[:, 1:, :]  # drop CLS
        return hm_feats


    @torch.no_grad()
    def _preprocess_heatmaps_to_weights(self, heatmaps: list[torch.Tensor]) -> torch.Tensor:
        """
        heatmaps: list of [1,1,H,W] in original image space.
        Convert to PIL grayscale, replicate to RGB, and run the same processor in batch.
        Returns tensor [B,1,H',W'] in preprocessed space.
        """
        heatmap_pils = heatmaps_to_rgb_pils(heatmaps)

        hm_out = self.processor.image_processor(
            images=heatmap_pils,
            return_tensors="pt",
            do_normalize=False,
            do_rescale=True,
        )["pixel_values"]  # [B,3,H',W']
        
        grid_h = hm_out.shape[-2] // self.vision_patch_size
        grid_w = hm_out.shape[-1] // self.vision_patch_size
        hm = hm_out.mean(dim=1, keepdim=True)  # [B,1,H,W]
        return heatmap_to_patch_weights(hm, grid_h, grid_w).to(self.device)  # [B,N]

    @torch.no_grad()
    def generate_with_weighted_patches(
        self,
        images: list[Image.Image],
        heatmaps: list[torch.Tensor],
        cfg: GenerationConfig,
        *,
        cors: list[str],
        use_vision_hook: bool = True,
    ) -> list[str]:
        """
        inference end-to-end (batched):
          - preprocess heatmaps with same pipeline as images
          - compute patch weights from heatmaps
          - build and tokenize prompts on-demand
          - inject gaze-weighted patches via vision tower hook
          - generate using model.generate (handles projection and fusion)
          
        Args:
            images: List of PIL images [B]
            heatmaps: List of heatmap tensors, each [1,1,H,W] [B]
            cfg: Configuration
            cors: List of COR labels [B]
            use_vision_hook: Whether to apply gaze-weighted patch hook

        Returns:
            List of generated texts [B]
        """
        B = len(images)
        assert len(heatmaps) == B
        assert len(cors) == B

        use_dual_encoding = use_vision_hook and (self.dual_encoding_injector is not None) and (self.heatmap_encoder is not None)

        # 1) preprocess heatmaps with the image processor (batched)
        if use_vision_hook:
            if use_dual_encoding:
                hm_feats = self._embed_heatmaps_for_dual_encoding(heatmaps)  # [B,N,D]
            else:
                weights = self._preprocess_heatmaps_to_weights(heatmaps)  # [B,N]

        # 2) build prompts, tokenize text and process images (batched)
        prompt_texts = build_prompt_texts(self.processor, cors, prompt_version=cfg.prompt_version)
        inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
        )
        if cfg.do_sample:
            gen_kwargs["temperature"] = max(cfg.temperature, 1e-6)

        # Hook vision tower to inject gaze-weighted patches before fusion
        if use_vision_hook:
            vision_layer = getattr(self.model.config, "vision_feature_layer", None)
            select_strategy = getattr(self.model.config, "vision_feature_select_strategy", None)

            def _vision_hook(module, inp, out):
                # out is typically a ModelOutput with .hidden_states and .last_hidden_state
                # Determine which tensor the model will use.
                if hasattr(out, "hidden_states") and vision_layer is not None and out.hidden_states is not None:
                    feats = out.hidden_states[vision_layer] # [B, 1+N, Dv] = [B, 577, 1024] for LLaVA-1.5 with 14x14 patches
                    hs = list(out.hidden_states)
                else:
                    feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    hs = None

                Bt, T, D = feats.shape

                if use_dual_encoding:
                    if Bt != hm_feats.shape[0]:
                        raise RuntimeError(
                            f"[LlavaHFAdapter] Heatmap alignment mismatch: "
                            f"vision batch={Bt} images but heatmap_feats={hm_feats.shape[0]}."
                        )
                else:
                    if Bt != weights.shape[0]:
                        raise RuntimeError(
                            f"[LlavaHFAdapter] Heatmap alignment mismatch: "
                            f"vision batch={Bt} images but weights={weights.shape[0]}. "
                            f"Heatmaps must be processed with the same image_processor settings/order."
                        )

                if cfg.debug:
                    print(f"[DEBUG] vision_hook: feats.shape={feats.shape}, vision_layer={vision_layer}, select_strategy={select_strategy}")

                # "default" strategy removes CLS later, but at this layer we still have it (hm_feats already dropped CLS)
                cls = feats[:, :1, :]
                patches = feats[:, 1:, :]

                if use_dual_encoding:
                    if patches.shape[1] != hm_feats.shape[1]:
                        raise RuntimeError(
                            f"[LlavaHFAdapter] Patch count mismatch: {patches.shape[1]} vs heatmap feats {hm_feats.shape[1]}"
                        )
                else:
                    if patches.shape[1] != weights.shape[1]:
                        raise RuntimeError(
                            f"[LlavaHFAdapter] Patch count mismatch: {patches.shape[1]} vs weights {weights.shape[1]}"
                        )

                try:
                    if use_dual_encoding:
                        hm_local = hm_feats.to(dtype=patches.dtype)
                        g = self.dual_encoding_injector(hm_local.float()).to(dtype=patches.dtype)  # [B,N,1]
                        weighted_patches = patches * g
                    # If a learned injector is available, use it:
                    #   gates = injector(weights) in [min_gate,1]
                    #   patches' = patches * gates
                    elif self.gaze_injector is not None:
                        # Keep injector math stable in fp32, then cast to patch dtype
                        w_fp32 = weights.float()
                        g_fp32 = self.gaze_injector(w_fp32)          # [B,N] fp32
                        g = g_fp32.to(dtype=patches.dtype)           # [B,N] patch dtype
                        weighted_patches = patches * g
                    else:
                        # inference direct weighting
                        weighted_patches = apply_patch_weighting(patches, weights)

                    new_feats = torch.cat([cls, weighted_patches], dim=1)

                    if hs is not None:
                        hs[vision_layer] = new_feats
                        out.hidden_states = tuple(hs)
                    if hasattr(out, "last_hidden_state"):
                        out.last_hidden_state = new_feats
                    return out
                except Exception:
                    shape_msg = f"heatmap feats shape: {hm_feats.shape}." if use_dual_encoding else f"weights shape: {weights.shape}."
                    raise RuntimeError(
                        f"[LlavaHFAdapter] Failed to weight patches."
                        f"Patch tokens shape: {patches.shape}, {shape_msg}"
                    )     

            # handle = self.model.model.vision_tower.register_forward_hook(_vision_hook)
            handle = unwrap_to_llava(self.model).model.vision_tower.register_forward_hook(_vision_hook)
        try:
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if use_vision_hook:
                handle.remove()

        # decode
        out_texts = []
        for i in range(B):
            full_text = self.processor.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            if "ASSISTANT:" in full_text:
                answer = full_text.split("ASSISTANT:", 1)[1].strip()
            else:
                answer = full_text.strip()
            out_texts.append(answer)
            if answer == "":
                print(f"[WARNING] Decoded empty answer for input {i}. Full decoded text: '{full_text}'")
        return out_texts
