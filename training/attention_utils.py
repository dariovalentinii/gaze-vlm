import torch

class LastKAttnCapture:
    """
    Capture attention weights only from the last K decoder layers.

    For LlavaNextForConditionalGeneration where layers live at:
      model.model.language_model.layers
    use layers_path=("model","language_model","layers").
    """
    def __init__(self, model, k: int, layers_path=("model", "language_model", "layers")):
        self.model = model
        self.k = int(k)
        self.layers_path = layers_path
        self.handles = []
        self.attns = []

    def _get_layers(self):
        obj = self.model
        for name in self.layers_path:
            obj = getattr(obj, name)
        return obj  # ModuleList

    def __enter__(self):
        self.attns = []
        layers = self._get_layers()
        n = len(layers)
        start = max(0, n - self.k)

        for i in range(start, n):
            sa = layers[i].self_attn

            # Force only this layer's attention module to materialize attn weights
            h_pre = sa.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

            # Capture the attn weights (keep grads; do NOT detach)
            h_fwd = sa.register_forward_hook(self._fwd_hook)

            self.handles.extend([h_pre, h_fwd])

        return self

    def _pre_hook(self, module, args, kwargs):
        kwargs["output_attentions"] = True
        return args, kwargs

    def _fwd_hook(self, module, args, output):
        # LLaMA-style: (attn_output, attn_weights, past_key_value)
        attn_weights = output[1]
        self.attns.append(attn_weights)
        return output

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

