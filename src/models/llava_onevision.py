"""LLaVA-OneVision inference adapter."""

from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel
from src.models.unwrapping import unwrap_to_llava
from src.models.checkpoint_utils import (
    clone_vision_tower,
    find_encoder_block_for_hook,
    read_injector_min_gate,
    resolve_checkpoint_layout,
    vision_layer_requires_hidden_states,
    vision_layer_to_encoder_block_idx,
)
from src.data.heatmaps import (
    heatmaps_to_rgb_pils,
    heatmap_to_patch_weights,
    apply_patch_weighting,
    GazeInjector,
    DualEncodingGazeInjector,
)
from src.data.prompts import build_prompt_texts
from src.inference.runner import GenerationConfig


class LlavaOnevisionHFAdapter:
    """
    Minimal adapter for inference using HuggingFace LlavaOnevisionForConditionalGeneration.
    To be used with:
    llava-hf/llava-onevision-qwen2-7b-ov-chat-hf
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

        # LLava Onevision requires left padding and a pad token == eos token for batch generation.
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            tok = self.processor.tokenizer
            tok.padding_side = "left"
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = tok.eos_token    

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(self.device)

        self.model.config.pad_token_id = tok.pad_token_id
        self.model.generation_config.pad_token_id = tok.pad_token_id


        # Vision geometry: token grid for ONE processed patch-image
        vcfg = self.model.config.vision_config
        self.vision_image_size = int(getattr(vcfg, "image_size", None))
        self.vision_patch_size = int(getattr(vcfg, "patch_size", None))
        if self.vision_image_size is None or self.vision_patch_size is None:
            raise ValueError(
                f"Missing vision geometry config: image_size={self.vision_image_size}, patch_size={self.vision_patch_size}"
            )
            
        # these can be removed!!!
        self.grid_hw = self.vision_image_size // self.vision_patch_size
        self.num_patch_tokens = self.grid_hw * self.grid_hw
        
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
            projector_lora_dir, self.checkpoint_dir = resolve_checkpoint_layout(
                Path(lora_dir),
                adapter_label="LlavaOneVisionHFAdapter",
            )
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

    def _load_gaze_injector(self, inj_dir: Path) -> None:
        pt = inj_dir / "gaze_injector.pt"
        if not pt.exists():
            return

        min_gate = read_injector_min_gate(inj_dir)
        sd = torch.load(pt, map_location="cpu")

        # Dual Encoding injector state has projection/norm params and works on [B,N,D].
        is_dual_encoding = any(k.startswith("proj.") or k.startswith("norm.") for k in sd.keys())
        if is_dual_encoding:
            d_model = int(getattr(self.model.config.vision_config, "hidden_size", 1024))
            dual_encoding_injector = DualEncodingGazeInjector(d_model=d_model, min_gate=min_gate)
            dual_encoding_injector.load_state_dict(sd, strict=True)
            self.dual_encoding_injector = dual_encoding_injector.to(self.device).eval()
            self.gaze_injector = None
            print(f"[LlavaOneVisionHFAdapter] Loaded Dual Encoding gaze injector from: {pt}")
            return

        injector = GazeInjector(min_gate=min_gate)
        injector.load_state_dict(sd, strict=True)
        self.gaze_injector = injector.to(self.device).eval()
        self.dual_encoding_injector = None
        print(f"[LlavaOneVisionHFAdapter] Loaded gaze injector from: {pt}")

    def _load_heatmap_encoder(self, ckpt_dir: Path) -> None:
        hm_lora_dir = ckpt_dir / "heatmap_encoder_lora"
        if not hm_lora_dir.exists():
            return

        core = unwrap_to_llava(self.model)
        img_vision = core.model.vision_tower
        heatmap_encoder_base = clone_vision_tower(img_vision).to(self.device)
        self.heatmap_encoder = PeftModel.from_pretrained(heatmap_encoder_base, str(hm_lora_dir)).to(self.device)
        self.heatmap_encoder.eval()

        # Mirror train_de logic: when possible, capture the selected vision layer
        # via hook to avoid requesting full hidden_states.
        vision_layer = getattr(self.model.config, "vision_feature_layer", None)
        num_vision_layers = int(getattr(self.model.config.vision_config, "num_hidden_layers", 0) or 0)
        hm_block_idx = vision_layer_to_encoder_block_idx(vision_layer, num_vision_layers)
        if hm_block_idx is not None:
            hm_block = find_encoder_block_for_hook(self.heatmap_encoder, hm_block_idx)
            if hm_block is not None:
                def _heatmap_block_hook(_module, _inp, out):
                    self._heatmap_layer_out = out[0] if isinstance(out, (tuple, list)) else out

                self._heatmap_hook_handle = hm_block.register_forward_hook(_heatmap_block_hook)

        print(f"[LlavaOneVisionHFAdapter] Loaded heatmap encoder LoRA from: {hm_lora_dir}")


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
        )["pixel_values"]
        
        if hm_pix.ndim == 5:
            b, t, c, h, w = hm_pix.shape
            hm_pix = hm_pix.view(b * t, c, h, w).to(self.device)

        vision_layer = getattr(self.model.config, "vision_feature_layer", None)
        num_layers = int(getattr(self.model.config.vision_config, "num_hidden_layers", 0) or 0)
        need_hidden_states = vision_layer_requires_hidden_states(vision_layer, num_layers) and self._heatmap_hook_handle is None

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
        # NO need to drop CLS in OV, because it does not prepend a CLS token to the patch tokens. The output is already [B,N,D] without CLS.
        return hm_feats_all  # [B,N,D]

    @torch.no_grad()
    def _preprocess_heatmaps_to_weights(self, heatmaps: list[torch.Tensor]) -> torch.Tensor:
        """
        Preprocess heatmaps through the same image processor as images, keeping AnyRes tiling.
        Returns patch-grid weights aligned to LlavaNext vision tower order.
        output shape: [num_tiles_total, N]
        """
        heatmap_pils = heatmaps_to_rgb_pils(heatmaps)

        hm_out = self.processor.image_processor(
            images=heatmap_pils,
            return_tensors="pt",
            do_normalize=False,
            do_rescale=True,
        )["pixel_values"]
        
        H, W = hm_out.shape[-2:]
        grid_h = H // self.vision_patch_size
        grid_w = W // self.vision_patch_size

        # hm_out can be either:
        #  - [B, C, H, W]   (no AnyRes)
        #  - [B, T, C, H, W] (AnyRes tiles)
        if hm_out.ndim == 4:
            hm = hm_out.mean(dim=1, keepdim=True)  # [B,1,H,W]
            return heatmap_to_patch_weights(hm, grid_h, grid_w).to(self.device)  # [B,N]

        if hm_out.ndim == 5:
            hm = hm_out.mean(dim=2, keepdim=True)  # [B,T,1,H,W]
            B, T, _, H, W = hm.shape
            hm = hm.reshape(B * T, 1, H, W)
            return heatmap_to_patch_weights(hm, grid_h, grid_w).to(self.device)  # [B*T,N]

        raise RuntimeError(f"Unexpected heatmap pixel_values shape from processor: {hm_out.shape}")



    @torch.no_grad()
    def generate_with_weighted_patches(
        self,
        images: list[Image.Image],
        heatmaps: Optional[list[torch.Tensor]],
        cfg: GenerationConfig,
        *,
        cors: list[str],
        use_vision_hook: bool = True,
    ) -> list[str]:
        B = len(images)
        assert len(cors) == B

        if use_vision_hook:
            if heatmaps is None:
                raise ValueError("Heatmaps are required when gaze injection is enabled.")
            assert len(heatmaps) == B
        
        use_dual_encoding = use_vision_hook and (self.dual_encoding_injector is not None) and (self.heatmap_encoder is not None)

        # 1) preprocess heatmaps with the same image processor (batched, AnyRes aware)
        if use_vision_hook:
            if use_dual_encoding:
                hm_feats = self._embed_heatmaps_for_dual_encoding(heatmaps)  # [B,N,D]
            else:
                weights = self._preprocess_heatmaps_to_weights(heatmaps)  # [B*T,N] or [B,N] depending on AnyRes.
            
        # 2) build prompts, tokenize text and process images (batched)
        prompt_texts = build_prompt_texts(self.processor, cors, prompt_version=getattr(cfg, "prompt_version", "v2"))
        inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
        )
        if cfg.do_sample:
            gen_kwargs["temperature"] = max(cfg.temperature, 1e-6)

        if use_vision_hook:
            vision_layer = getattr(self.model.config, "vision_feature_layer", None)
            select_strategy = getattr(self.model.config, "vision_feature_select_strategy", None)

            def _vision_hook(module, inp, out):
                # Select features
                if hasattr(out, "hidden_states") and vision_layer is not None and out.hidden_states is not None:
                    feats = out.hidden_states[vision_layer]
                    hs = list(out.hidden_states)
                else:
                    feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    hs = None

                # feats: [num_tiles_total, seq_len, D]
                Bt, T, D = feats.shape

                if use_dual_encoding:
                    if Bt != hm_feats.shape[0]:
                        raise RuntimeError(
                            f"[LlavaOneVisionHFAdapter] Heatmap alignment mismatch: "
                            f"vision batch={Bt} images but heatmap_feats={hm_feats.shape[0]}."
                        )
                else:
                    if Bt != weights.shape[0]:
                        raise RuntimeError(
                            f"[LlavaOneVisionHFAdapter] Heatmap alignment mismatch: "
                            f"vision batch={Bt} images but weights={weights.shape[0]}. "
                            f"Heatmaps must be processed with the same image_processor settings/order."
                        )

                if cfg.debug:
                    print(f"[DEBUG] vision_hook: feats.shape={feats.shape}, vision_layer={vision_layer}, select_strategy={select_strategy}")

                # LLavaOnevision DOES NOT prepend a CLS token to the patch tokens, so we don't need to separate it before weighting.
                patches = feats

                if use_dual_encoding:
                    if patches.shape[1] != hm_feats.shape[1]:
                        raise RuntimeError(
                            f"[LlavaOneVisionHFAdapter] Patch count mismatch: {patches.shape[1]} vs heatmap feats {hm_feats.shape[1]}"
                        )
                else:
                    if patches.shape[1] != weights.shape[1]:
                        raise RuntimeError(
                            f"[LlavaOneVisionHFAdapter] Patch count mismatch: {patches.shape[1]} vs weights {weights.shape[1]}"
                        )

                try:
                    if use_dual_encoding:
                        g = self.dual_encoding_injector(hm_feats.float()).to(dtype=patches.dtype)  # [B,N,1]
                        new_feats = patches * g
                    # If a learned injector is available, use it:
                    elif self.gaze_injector is not None:
                        g = self.gaze_injector(weights.float()).to(dtype=patches.dtype)  # [B,N,1]
                        new_feats = patches * g
                    else:
                        # inference direct weighting
                        new_feats = apply_patch_weighting(patches, weights)

                    if hs is not None:
                        hs[vision_layer] = new_feats
                        out.hidden_states = tuple(hs)
                    if hasattr(out, "last_hidden_state"):
                        out.last_hidden_state = new_feats
                    return out
                except Exception:
                    shape_msg = f"heatmap feats shape: {hm_feats.shape}." if use_dual_encoding else f"weights shape: {weights.shape}."
                    raise RuntimeError(
                        f"[LlavaOneVisionHFAdapter] Failed to weight patches."
                        f"Patch tokens shape: {patches.shape}, {shape_msg}"
                    )

            handle = unwrap_to_llava(self.model).model.vision_tower.register_forward_hook(_vision_hook)
        try:
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if use_vision_hook:
                handle.remove()

        out_texts = []
        for i in range(B):
            full_text = self.processor.tokenizer.decode(output_ids[i], skip_special_tokens=True,         clean_up_tokenization_spaces=False)
            if "assistant\n" in full_text:
                answer = full_text.split("assistant\n", 1)[1].strip()
            else:
                answer = full_text.strip()
            out_texts.append(answer)
            if answer == "":
                print(f"[WARNING] Decoded empty answer for input {i}. Full decoded text: '{full_text}'")
        return out_texts
