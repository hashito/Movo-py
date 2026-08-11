"""movo-core — 画素・コーデック・色・数学・乱数・キャッシュ・設定。

**ここが «依存の底» です。** core は他の movo のパッケージを import しません。
逆向きの依存が 1 本でも入ると、レンダラを差し替えたときにコーデックまで
巻き込まれます。

速度の作法は :mod:`movo.core.bitmap` の冒頭に書いてあります。要点だけ:

- 全画面の一様な処理は **NumPy の一括演算**
- 画素ごとに分岐する処理（PNG のフィルタ復元・JPEG の逆 DCT）は **Numba**
- ``uint8`` のまま掛けない（255 で巻き戻ります）
"""

from __future__ import annotations

from .assets import AssetStore, create_placeholder, infer_spec
from .bitmap import Bitmap, blend_over, to_float, to_u8
from .cache import Cache
from .color import (
    NAMED,
    color_to_css,
    color_to_rgba8,
    hsl_to_rgb,
    mix_color,
    parse_color,
    rgb_to_hsl,
)
from .config import (
    config_file_path,
    get_config_value,
    is_secret_key,
    list_config,
    load_config,
    mask_secret,
    save_config,
    set_config_value,
    unset_config_value,
)
from .errors import ErrorCodes, MovoError, MovoValidationError, to_movo_error
from .flash_guard import FLASH_DEFAULTS, FlashGuard, describe_flash_report
from .hash import hash_file, hash_json, sha256, short_hash, stable_stringify
from .image import decode_bmp, decode_image, is_bmp, load_image, save_image
from .jpeg import decode_jpeg, is_jpeg
from .logger import Logger, logger, style
from .lut import (
    DEFAULT_MAX_LUT_BYTES,
    MAX_LUT_3D_SIZE,
    Lut3D,
    apply_lut,
    identity_lut,
    parse_cube_lut,
    sample_lut,
)
from .lyrics import (
    detect_lyrics_format,
    parse_lrc,
    parse_lyrics,
    parse_subtitles,
    slice_lyrics,
)
from .math import (
    DEG,
    TAU,
    Mat2D,
    approximately,
    catmull_rom,
    clamp,
    inverse_lerp,
    js_round,
    lerp,
    sample_polyline,
    smoothstep,
    solve2x2,
    to_degrees,
    to_radians,
)
from .platform import (
    cpu_count,
    describe_environment,
    detect_gpu,
    find_ffmpeg,
    find_ffprobe,
    list_font_files,
    movo_home,
    platform,
    resolve_project_path,
    run,
    system_font_dirs,
    temp_dir,
    to_project_relative,
)
from .png import decode_png, encode_png, is_png
from .profile_library import list_profiles, load_profile_target
from .rng import (
    Random,
    RandomSource,
    create_random,
    fbm1d,
    fbm2d,
    fbm2d_grid,
    hash_string,
    value_noise_1d,
    value_noise_2d,
    value_noise_3d,
)
from .svg_path import (
    DEFAULT_MAX_ELEMENTS,
    DEFAULT_MAX_SEGMENTS,
    DEFAULT_SVG_MAX_BYTES,
    arc_to_cubics,
    extract_svg_shapes,
    flatten_segments,
    identity_matrix,
    is_trim_active,
    multiply,
    parse_path_data,
    parse_transform,
    path_to_subpaths,
    subpaths_bounds,
    trim_subpaths,
)
from .version import (
    COMPONENT_VERSIONS,
    MOVO_JSON_VERSION,
    MOVO_VERSION,
    is_compatible_json_version,
)
from .video_compare import METRICS, compare_profile, compare_to_reference, describe_comparison
from .video_profile import VideoProfiler
from .wav import AudioBuffer, create_silence, decode_wav, encode_wav, resample

__all__ = [
    "AssetStore", "AudioBuffer", "Bitmap", "COMPONENT_VERSIONS", "Cache",
    "DEFAULT_MAX_ELEMENTS", "DEFAULT_MAX_LUT_BYTES", "DEFAULT_MAX_SEGMENTS",
    "DEFAULT_SVG_MAX_BYTES", "DEG", "ErrorCodes", "FLASH_DEFAULTS", "FlashGuard",
    "Logger", "Lut3D", "MAX_LUT_3D_SIZE", "METRICS", "MOVO_JSON_VERSION", "MOVO_VERSION",
    "Mat2D", "MovoError", "MovoValidationError", "NAMED", "Random", "RandomSource",
    "TAU", "VideoProfiler",
    "apply_lut", "approximately", "arc_to_cubics", "blend_over", "catmull_rom", "clamp",
    "color_to_css", "color_to_rgba8", "compare_profile", "compare_to_reference",
    "config_file_path", "cpu_count", "create_placeholder", "create_random",
    "create_silence", "decode_bmp", "decode_image", "decode_jpeg", "decode_png",
    "decode_wav", "describe_comparison", "describe_environment", "describe_flash_report",
    "detect_gpu", "detect_lyrics_format", "encode_png", "encode_wav",
    "extract_svg_shapes", "fbm1d", "fbm2d", "fbm2d_grid", "find_ffmpeg", "find_ffprobe",
    "flatten_segments", "get_config_value", "hash_file", "hash_json", "hash_string",
    "hsl_to_rgb", "identity_lut", "identity_matrix", "infer_spec", "inverse_lerp",
    "is_bmp", "is_compatible_json_version", "is_jpeg", "is_png", "is_secret_key",
    "is_trim_active", "js_round", "lerp", "list_config", "list_font_files",
    "list_profiles", "load_config", "load_image", "load_profile_target", "logger",
    "mask_secret", "mix_color", "movo_home", "multiply", "parse_color", "parse_cube_lut",
    "parse_lrc", "parse_lyrics", "parse_path_data", "parse_subtitles", "parse_transform",
    "path_to_subpaths", "platform", "resample", "resolve_project_path", "rgb_to_hsl",
    "run", "sample_lut", "sample_polyline", "save_config", "save_image", "set_config_value",
    "sha256", "short_hash", "slice_lyrics", "smoothstep", "solve2x2", "stable_stringify",
    "style", "subpaths_bounds", "system_font_dirs", "temp_dir", "to_degrees", "to_float",
    "to_movo_error", "to_project_relative", "to_radians", "to_u8", "trim_subpaths",
    "unset_config_value", "value_noise_1d", "value_noise_2d", "value_noise_3d",
]
