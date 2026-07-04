import torch


# ── label / region helpers ────────────────────────────────────────────────────
# The pipeline is class-count agnostic: class names, evaluation regions and the
# label palette are all derived from the config so the same code serves both the
# BraTS-2024 5-class post-treatment scheme (NETC/SNFH/ET/RC) and the BraTS-2023
# 4-class scheme (NCR/ED/ET).

# Default foreground names keyed by number of foreground classes.
_DEFAULT_CLASS_NAMES = {
    4: {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"},   # BraTS 2024 post-treatment
    3: {1: "NCR", 2: "ED", 3: "ET"},               # BraTS 2023 (pre-treatment)
}

# RGBA palette keyed by label index (used by the visualization utilities).
_DEFAULT_LABEL_COLORS = {
    1: [1.0, 0.27, 0.27, 0.65],   # red
    2: [1.0, 0.84, 0.0, 0.60],    # gold
    3: [0.0, 0.55, 1.0, 0.65],    # blue
    4: [0.65, 0.35, 0.95, 0.65],  # purple
}


def get_class_names(config: dict) -> dict:
    """Return ``{class_index: name}`` for the foreground classes from a config.

    Reads ``data.class_names`` when present; otherwise falls back to sensible
    defaults based on ``data.num_classes`` (5-class BraTS-2024 vs 4-class 2023).
    YAML parses ``{1: ET}`` keys as ints, but we coerce defensively.
    """
    data = config.get("data", config) if isinstance(config, dict) else {}
    raw = data.get("class_names")
    if raw:
        return {int(k): str(v) for k, v in raw.items()}
    n_fg = int(data.get("num_classes", 4)) - 1
    return dict(_DEFAULT_CLASS_NAMES.get(n_fg, {i: f"class{i}" for i in range(1, n_fg + 1)}))


def get_label_colors(config: dict = None) -> dict:
    """Return ``{class_index: RGBA}`` for foreground labels (config-aware)."""
    names = get_class_names(config) if config else {}
    return {idx: _DEFAULT_LABEL_COLORS.get(idx, [0.5, 0.5, 0.5, 0.6])
            for idx in (names or _DEFAULT_LABEL_COLORS)}


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
