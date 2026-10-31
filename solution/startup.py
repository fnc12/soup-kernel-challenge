"""Start-up work that the measured `soup train` command pays for and that we can make cheaper
without touching what is trained: a timeline of the run (`SOLUTION_TIMELINE=1`), one scan of
the installed packages instead of one per `importlib.metadata.version()` call, no fingerprint
hashing of the tokenizer by `datasets` (caching off: `map` runs exactly the same, it just does
not write cache files), and PEFT's fp32 upcast of the frozen embedding done on the CPU so the
fp16 and the fp32 copy are never on the GPU together.
"""

import atexit
import os
import re
import sys
import time

_T0 = time.perf_counter()
_ON = os.environ.get("SOLUTION_TIMELINE") == "1"


def mark(name):
    if _ON:
        mem = ""
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                mib = lambda b: int(b / 2**20)  # noqa: E731
                mem = (f" alloc {mib(torch.cuda.memory_allocated())} res {mib(torch.cuda.memory_reserved())}"
                       f" max-alloc {mib(torch.cuda.max_memory_allocated())} max-res {mib(torch.cuda.max_memory_reserved())}")
        except Exception:  # noqa: BLE001
            pass
        print(f"[solution-tl] {name} +{time.perf_counter() - _T0:.2f}s{mem} @{time.time():.3f}", file=sys.stderr, flush=True)


if _ON:
    atexit.register(mark, "exit")  # LIFO: runs after everything registered later, before `leave`


def cache_package_versions():
    """`importlib.metadata.version(name)` walks every dist-info directory on sys.path on every
    call; transformers, trl and peft call it for dozens of packages at import. One walk, one dict."""
    import importlib.metadata as md

    norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()  # noqa: E731
    table = {}
    for dist in md.distributions():
        try:
            name = dist.metadata["Name"]
            if name and norm(name) not in table:
                table[norm(name)] = dist.version
        except Exception:  # noqa: BLE001
            continue
    original = md.version

    def version(name):
        v = table.get(norm(name))
        if v is None:
            return original(name)  # unknown here: the real lookup, with its real error
        return v

    md.version = version
    try:
        import importlib_metadata  # the backport some packages use

        importlib_metadata.version = version
    except Exception:  # noqa: BLE001
        pass
    return len(table)


def datasets_caching_off():
    try:
        import datasets

        datasets.disable_caching()
        return True
    except Exception:  # noqa: BLE001
        return False


def patch_kbit_upcast():
    """PEFT's `prepare_model_for_kbit_training` casts every non-quantized parameter to fp32,
    the 467 MB tied embedding included: for a moment the GPU holds it twice (fp16 + fp32), the
    highest point of the whole run once the training itself is light. The cast happens on the
    CPU instead; solution/embed.py brings the table back to fp16 on the GPU afterwards."""
    try:
        import peft
        import torch
    except Exception:  # noqa: BLE001
        return False
    original = peft.prepare_model_for_kbit_training

    def prepare(model, *args, **kwargs):
        moved = []
        get_in = getattr(model, "get_input_embeddings", None)
        embed = get_in() if callable(get_in) else None
        w = getattr(embed, "weight", None)
        if isinstance(w, torch.Tensor) and w.is_cuda and w.dtype in (torch.float16, torch.bfloat16):
            device = w.device
            w.data = w.data.to("cpu")
            moved.append((w, device))
        try:
            return original(model, *args, **kwargs)
        finally:
            from .lowprec import activation_dtype

            for w, device in moved:
                w.data = w.data.to(activation_dtype()).to(device)  # what solution/embed.py keeps anyway

    peft.prepare_model_for_kbit_training = prepare
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith("soup_cli") \
                and getattr(module, "prepare_model_for_kbit_training", None) is original:
            module.prepare_model_for_kbit_training = prepare
    return True


def patch_loader_cast():
    """transformers moves each checkpoint tensor with one `tensor.to(device=..., dtype=...)`;
    for a CPU -> GPU move with a dtype change PyTorch lands the source bytes on the GPU first
    and converts there, so the 467 MB bf16 embedding briefly occupies twice its size on the
    GPU: the highest point of the load. Tensors over 64 MB are converted on the CPU first."""
    try:
        import torch
        from transformers import core_model_loading as cml
    except Exception:  # noqa: BLE001
        return False
    original = getattr(cml, "_materialize_copy", None)
    if original is None:
        return False

    def materialize(tensor, device=None, dtype=None):
        try:
            t = tensor[...]  # the safetensors slice becomes a (CPU, memory-mapped) tensor
            big = isinstance(t, torch.Tensor) and t.numel() * t.element_size() > 64 << 20
            if big:
                mark(f"materialize {tuple(t.shape)} {t.dtype} on {t.device} -> {device} {dtype}")
            cuda = device is not None and torch.device(device).type == "cuda" if not isinstance(device, int) \
                else True  # an integer device index means a CUDA device
            if big and dtype is not None and t.dtype != dtype and not t.is_cuda and cuda and t.dim() >= 1:
                # in slices: the bf16 bytes of one slice on the GPU at a time, converted in place
                out = torch.empty(t.shape, dtype=dtype, device=device)
                rows = max(1, t.shape[0] // 8)
                for a in range(0, t.shape[0], rows):
                    out[a:a + rows].copy_(t[a:a + rows].to(device=device))
                return out
            tensor = t
        except Exception:  # noqa: BLE001
            pass
        return original(tensor, device=device, dtype=dtype)

    cml._materialize_copy = materialize
    return True


def patch_allocator_warmup():
    """transformers warms the caching allocator up before a load by allocating one block the
    size of the weights it expects (about 1 GB here) and freeing it: the GPU's highest point
    of the whole run once the training itself is light. With `expandable_segments` the warm-up
    buys nothing; the block is simply not allocated."""
    try:
        from transformers import modeling_utils
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(modeling_utils, "caching_allocator_warmup"):
        return False
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    return True


_EXIT = {"code": None}


def patch_fast_exit():
    """The measured command ends when the process ends. After `soup train` has written its
    outputs and the harness its metrics, the interpreter still spends a while tearing down:
    freeing the pinned host memory and the CUDA context, collecting every object. The final
    `sys.exit(0)` is noted, and at the end of the atexit handlers every open file is flushed and
    the process leaves at once (`os._exit`); a run that ends any other way is left alone."""
    original = sys.exit

    def exit(code=None):
        _EXIT["code"] = code
        original(code)

    sys.exit = exit

    def leave():
        if _EXIT["code"] not in (0, None) or os.environ.get("SOLUTION_NO_FAST_EXIT") == "1":
            return
        try:
            import gc
            import io

            for obj in gc.get_objects():
                if isinstance(obj, io.IOBase) and not obj.closed and obj.writable():
                    try:
                        obj.flush()
                    except Exception:  # noqa: BLE001
                        pass
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            return
        os._exit(0)

    # atexit runs handlers last-registered first; this one must be the very last to run, so it is
    # registered here, before any other module's handler, by re-registering the earlier ones after it
    atexit.register(leave)
    if _ON:
        atexit.unregister(mark)
        atexit.register(mark, "exit")
    return True
