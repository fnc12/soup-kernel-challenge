"""Keep the activations in fp16 between the 4-bit matmuls.

`prepare_model_for_kbit_training` upcasts the norms and the embedding to fp32, so the residual
stream of the base run is fp32. bitsandbytes' `Linear4bit.forward` casts its input to the compute
dtype (fp16), runs the matmul, and casts the result BACK to the input's dtype: with fp32 inputs
every q/k/v/o/gate/up/down projection writes an fp32 copy of its fp16 result, the MLP's silu and
product run in fp32 on values that are fp16-exact, and the next matmul casts them down again.
Those casts are a good part of the "44% element-wise ops" in the organizers' profile, and the fp32
copies are what the backward pass keeps.

With this patch a 4-bit linear returns its fp16 result as it is. The values are identical (the
cast back was exact), the residual adds stay fp32 (fp32 + fp16 -> fp32), and everything the
backward pass saves in the attention and MLP blocks is half the size.
"""

import torch


def activation_dtype():
    """What Soup computes in on this card: bf16 where the GPU has bf16 units, else fp16 (a T4)."""
    try:
        from soup_cli.utils.gpu import cuda_supports_bf16

        return torch.bfloat16 if cuda_supports_bf16() else torch.float16
    except Exception:  # noqa: BLE001
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16


def patch_bnb_linear4bit(own_dequant=True):
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import modules
    except Exception:  # noqa: BLE001 - no bitsandbytes (the regression tests): nothing to patch
        return False
    from . import nf4

    cls = modules.Linear4bit
    original = cls.forward

    def forward(self, x):
        compute_dtype = getattr(self, "compute_dtype", None)
        if not isinstance(x, torch.Tensor) or not x.is_cuda or compute_dtype is None \
                or getattr(self, "bias", None) is not None:
            return original(self, x)
        if hasattr(modules, "fix_4bit_weight_quant_state_from_module"):
            modules.fix_4bit_weight_quant_state_from_module(self)
        quant_state = self.weight.quant_state
        if not getattr(self, "compute_type_is_set", True):
            self.set_compute_type(x)
            self.compute_type_is_set = True
        if x.dtype != compute_dtype:
            x = x.to(compute_dtype)
        if own_dequant and nf4.supported(quant_state) and x.shape[-1] == quant_state.shape[1]:
            return nf4.quant_linear(x, self.weight, quant_state)
        return bnb.matmul_4bit(x, self.weight, bias=None, quant_state=quant_state)

    cls.forward = forward
    return True
