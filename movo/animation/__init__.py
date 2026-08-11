"""movo.animation — キーフレーム・イージング・モジュレーター・値の解決。"""

from __future__ import annotations

from .easing import EASINGS, apply_easing, cubic_bezier, get_easing, list_easings
from .keyframes import keyframe_range, sample_keyframes
from .modulators import (
    COMBINE_MODES,
    MODULATOR_TYPES,
    combine_values,
    evaluate_modulator,
    list_modulators,
)
from .resolver import (
    ANIMATED_KEYS,
    apply_animations,
    get_path,
    is_animated_spec,
    local_time_for,
    resolve_animated,
    resolve_number,
    set_path,
)

__all__ = [
    "ANIMATED_KEYS",
    "COMBINE_MODES",
    "EASINGS",
    "MODULATOR_TYPES",
    "apply_animations",
    "apply_easing",
    "combine_values",
    "cubic_bezier",
    "evaluate_modulator",
    "get_easing",
    "get_path",
    "is_animated_spec",
    "keyframe_range",
    "list_easings",
    "list_modulators",
    "local_time_for",
    "resolve_animated",
    "resolve_number",
    "sample_keyframes",
    "set_path",
]
