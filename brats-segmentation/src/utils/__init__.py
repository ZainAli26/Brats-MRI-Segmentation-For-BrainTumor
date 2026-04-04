import torch


def inference_wrapper(model):
    """Wrap model so it returns only the main output during inference.

    Deep supervision models (nnU-Net v2 PlainConvUNet, DynUNet) return
    multiple outputs during training. sliding_window_inference can't
    aggregate lists, so we extract only the first (full-resolution) head.
    """
    def wrapped(x):
        out = model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        # DynUNet stacked: [B, heads, C, H, W, D] -> [B, C, H, W, D]
        if isinstance(out, torch.Tensor) and out.ndim == 6:
            return out[:, 0]
        return out
    return wrapped
