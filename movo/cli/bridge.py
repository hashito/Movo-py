"""**他の担当が移植中のモジュールに繋ぐ場所。**

`core` / `schema` / `renderer` / `physics` / `audio` / `timeline` / `exporters` は
別のエージェントが並行して移植しています。CLI はそれらを使いますが、
**まだ揃っていないことがあります。**

そこで «インポートを試して、無ければ理由を持った関数を返す» 口をここに 1 か所
だけ作りました。狙いは 3 つです。

1. **CLI 自体は必ず起動する。** `movo --version` や `movo skill list` のように、
   繋ぎ先が要らないコマンドは未接続でも動きます。
2. **未接続を «黙って壊れる» にしない。** 使おうとした瞬間に
   「〈どのモジュール〉が未接続です（後で繋ぐ）」と名指しで止まります。
   None が奥まで流れて `AttributeError` になるのがいちばん困る壊れ方です。
3. **繋ぎ替えが 1 行で済む。** 相手が仕上がったら、ここの候補名に足すだけです。

`movo doctor --json` の `bridge` 欄に «今どこまで繋がっているか» が出ます。
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from .errors import ErrorCodes, MovoError

# «この機能はどのモジュールに要るか» の対応表。doctor の表示にも使います。
#
# 値は `(説明, 繋がったと言える目印)`。**目印まで見るのが要点です。** 移植中の
# モジュールは «空の `__init__.py` だけある» 状態を通るので、import が通ったこと
# だけで «繋がった» と数えると、doctor が «23/23 繋がっています» と嘘をつきます
# （実際に一度そう出ました）。
BRIDGE_TARGETS: dict[str, tuple[str, str]] = {
    "movo.core.bitmap": ("画素の入れ物", "Bitmap"),
    "movo.core.png": ("PNG の読み書き", "encode_png"),
    "movo.core.wav": ("WAV の書き出し", "encode_wav"),
    "movo.core.platform": ("ffmpeg / GPU / フォントの探索", "find_ffmpeg"),
    "movo.core.config": ("設定（API キー）", "get_config_value"),
    "movo.core.cache": ("キャッシュ", "Cache"),
    "movo.core.assets": ("素材の解決", "AssetStore"),
    "movo.core.flash_guard": ("光過敏性発作（PSE）の検査", "FlashGuard"),
    "movo.core.video_profile": ("映像を数値にする", "VideoProfiler"),
    "movo.core.video_compare": ("数値をくらべる", "compare_profile"),
    "movo.core.profile_library": ("作風の目標値", "list_profiles"),
    "movo.schema": ("検証と正規化", "normalize_project"),
    "movo.schema.params": ("params と «作り方»", "prepare_project"),
    "movo.timeline": ("シーンとレイヤーの時間解決", "build_timeline"),
    "movo.renderer": ("レンダラー", "Renderer"),
    "movo.renderer.effects": ("エフェクト一覧", "list_effects"),
    "movo.animation.modulators": ("モジュレーター一覧", "list_modulators"),
    "movo.animation.easing": ("イージング一覧", "list_easings"),
    "movo.deformer": ("変形処理", "list_deformers"),
    "movo.physics": ("物理演算", "describe_physics"),
    "movo.expression": ("式エンジン", "ExpressionEngine"),
    "movo.audio": ("音声（WAV / BPM 検出 / ミックス）", "analyze_audio"),
    "movo.exporters": ("書き出し（mp4 / webm / gif / png 連番）", "create_exporter"),
}


class NotConnectedError(MovoError):
    """まだ移植が終わっていないモジュールを使おうとしたときのエラー。

    «後で繋ぐ» ことがはっきり分かる文言にしてあります。バグと区別が付かない
    エラーを出すと、繋ぎ忘れなのか壊れたのかを毎回調べ直すことになります。
    """

    def __init__(self, module: str, what: str = "") -> None:
        label = BRIDGE_TARGETS.get(module, (what or module,))[0]
        super().__init__(
            ErrorCodes.MOVO_RENDERER_UNAVAILABLE,
            f"{module}（{label}）はまだ移植されていません — **その部分は後で繋ぐ**",
            hint=(
                "この機能は他の担当が移植中のモジュールを使います。\n"
                "  繋がっているものだけを確かめるには: movo doctor"
            ),
        )
        self.module = module


_CACHE: dict[str, Any] = {}


def optional_module(name: str):
    """モジュールを読めたら返す。読めなければ None。

    **例外は握り潰しません** — 読めなかった理由は `module_status()` に残します。
    移植途中の相手が «import は通るのに中で落ちる» 状態になることがあり、
    そのときに「無い」と「壊れている」を区別できないと原因を探せません。
    """
    if name in _CACHE:
        value = _CACHE[name]
        return value if not isinstance(value, BaseException) else None
    try:
        module = importlib.import_module(name)
    except BaseException as error:  # noqa: BLE001 - 理由ごと覚えておきたい
        _CACHE[name] = error
        return None
    _CACHE[name] = module
    return module


def require_module(name: str):
    """無ければ «後で繋ぐ» と名指しで止まる版。"""
    module = optional_module(name)
    if module is None:
        raise NotConnectedError(name)
    return module


def pick(module_name: str, *candidates: str) -> Callable[..., Any]:
    """モジュールから «この名前のどれか» を取る。

    JS 版と Python 版で名前が変わる（`normalizeProject` → `normalize_project`）
    ので、候補を並べて先に見つかったものを使います。相手の命名がまだ決まって
    いない段階でも CLI を書き進められるようにするためです。
    """
    module = optional_module(module_name)
    if module is not None:
        for candidate in candidates:
            found = getattr(module, candidate, None)
            if found is not None:
                return found

    def _not_connected(*_args: Any, **_kwargs: Any):
        raise NotConnectedError(module_name, candidates[0] if candidates else "")

    _not_connected.__name__ = candidates[0] if candidates else "not_connected"
    _not_connected.movo_not_connected = True  # type: ignore[attr-defined]
    return _not_connected


def is_connected(module_name: str) -> bool:
    """**目印まで見て** 繋がっているかを判定する。

    import が通っただけでは «中身が空» のことがあります（移植中は必ずその状態を
    通ります）。呼ぶ側が使う名前が実際にあるかまで確かめます。
    """
    module = optional_module(module_name)
    if module is None:
        return False
    marker = BRIDGE_TARGETS.get(module_name, (None, None))[1]
    return marker is None or getattr(module, marker, None) is not None


def module_status() -> list[dict[str, Any]]:
    """繋がり具合の一覧（`movo doctor` が使います）。"""
    rows = []
    for name, (label, marker) in BRIDGE_TARGETS.items():
        module = optional_module(name)
        connected = is_connected(name)
        if connected:
            reason = None
        elif module is None:
            reason = _reason_of(_CACHE.get(name))
        else:
            reason = f"移植中（{marker} がまだありません）"
        rows.append({"module": name, "label": label, "connected": connected, "reason": reason})
    return rows


def _reason_of(problem: Any) -> str | None:
    if isinstance(problem, ModuleNotFoundError):
        return "未移植"
    if isinstance(problem, BaseException):
        return f"読み込みに失敗: {problem}"
    return "未移植"


def to_bitmap(width: int, height: int, raw: bytes):
    """生の RGBA バイト列を Bitmap にする。

    ffmpeg から `-pix_fmt rgba` で受けたフレームを包むためのものです。
    `Bitmap` は `(高さ, 幅, 4)` の uint8 を持つので、そのまま形を合わせます。
    """
    import numpy as np

    bitmap_class = require_module("movo.core.bitmap").Bitmap
    data = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).copy()
    return bitmap_class(width, height, data)


def listing(module_name: str, *candidates: str) -> list:
    """一覧を取り出す。未接続なら空の一覧（`movo list` を止めないため）。

    ここだけは «止まらない» ほうを選んでいます。`movo list effects` は
    «何が使えるか» を調べるコマンドなので、繋がっていないものは «0 件» と
    出るほうが、エラーで止まるより情報量が多いためです。

    **関数でも定数でも受けます。** 一覧は `list_effects()` のように関数のことも、
    `BLEND_MODES` のように定数のこともあります。関数として呼ぶ前提で書くと、
    定数のほうが黙って «0 件» になります（`movo list blends` が実際にそうでした）。
    """
    found = pick(module_name, *candidates)
    if getattr(found, "movo_not_connected", False):
        return []
    try:
        result = found() if callable(found) else found
    except Exception:  # noqa: BLE001 - 一覧のために止まりたくはない
        return []
    if isinstance(result, dict):
        return list(result)
    return list(result or [])


# ── よく使う繋ぎ先（呼ぶ側はここだけを見ればよい）─────────────────
def encode_png(bitmap, **kwargs):
    return pick("movo.core.png", "encode_png", "encodePng")(bitmap, **kwargs)


def decode_png(data):
    return pick("movo.core.png", "decode_png", "decodePng")(data)


def encode_wav(audio):
    return pick("movo.core.wav", "encode_wav", "encodeWav")(audio)


def find_ffmpeg(refresh: bool = False):
    """ffmpeg の場所。core が未接続でも **自力で PATH を見ます。**

    並列レンダリングは «繋げるかどうか» の判定に ffmpeg の有無を使うので、
    ここが未接続で止まると «割れるのに 1 本で描く» ことになります。
    それは黙って遅くなるだけなので、最低限の探索はこちらで持ちます。
    """
    module = optional_module("movo.core.platform")
    if module is not None:
        function = getattr(module, "find_ffmpeg", None) or getattr(module, "findFfmpeg", None)
        if function is not None:
            try:
                return function(refresh) if refresh else function()
            except TypeError:
                return function()
    return _which_tool("ffmpeg")


def find_ffprobe():
    module = optional_module("movo.core.platform")
    if module is not None:
        function = getattr(module, "find_ffprobe", None) or getattr(module, "findFfprobe", None)
        if function is not None:
            return function()
    return _which_tool("ffprobe")


def _which_tool(name: str):
    """PATH と、**`movo setup-ffmpeg` が置いた場所** から探す。

    置き場を見ないと、取ってきた直後なのに «見つかりません» と言われる
    （PATH には入れないため）。PATH を先に見るのは、利用者が自分で入れた
    ものがあればそちらを尊重するという判断である。
    """
    import shutil

    path = shutil.which(name)
    if path:
        return {"path": path, "version": None}

    try:
        from .config_store import movo_home

        for candidate in (movo_home() / "bin" / name, movo_home() / "bin" / (name + ".exe")):
            if candidate.exists():
                return {"path": str(candidate), "version": None}
    except Exception:
        pass
    return None
