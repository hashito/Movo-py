"""素材の解決。

素材は ``project.assets`` に 1 度だけ宣言し、レイヤー・モディファイア・マスク・
音声トラックから **名前で**参照します。ここはその宣言を «復号済みのビットマップ»
や «音のバッファ» に解決し、必要ならダウンロードや生成をして、結果を覚えます。

AI による生成は **コールバックとして差し込みます。** core が AI のパッケージに
依存しないためで、両者を繋ぐのは CLI の仕事です。

## 壊れた素材 1 つで動画が出ないのは割に合わない

``resolve_all()`` は失敗を **集めて返します**（投げません）。プレビューは
出したいからです。``strict=True``（``movo validate`` / CI）にすると例外になります。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Any, Callable

import numpy as np

from .bitmap import Bitmap
from .errors import ErrorCodes, MovoError
from .hash import sha256
from .image import decode_image
from .logger import logger
from .lut import parse_cube_lut
from .lyrics import parse_lyrics
from .platform import find_ffmpeg, resolve_project_path
from .rng import create_random, hash_string
from .svg_path import DEFAULT_SVG_MAX_BYTES, extract_svg_shapes
from .wav import AudioBuffer, decode_wav

_IMAGE_EXTENSIONS = re.compile(r"\.(png|jpe?g|bmp|gif|webp|tiff?)$", re.IGNORECASE)
_AUDIO_EXTENSIONS = re.compile(r"\.(wav|mp3|m4a|aac|ogg|flac)$", re.IGNORECASE)
_VIDEO_EXTENSIONS = re.compile(r"\.(mp4|mov|webm|mkv|avi)$", re.IGNORECASE)


class AssetStore:
    """宣言された素材を «使える形» にして持つ入れ物。

    :param project_root: プロジェクトのルート（相対パスの起点）
    :param assets: ``project.assets`` の中身
    :param cache: ダウンロードを覚えるキャッシュ（無くても動きます）
    :param security: ``allowNetwork`` / ``maxDownloadSizeMB`` など
    :param generator: AI 生成のコールバック
    :param seed: プレースホルダの見た目を決める種
    """

    def __init__(
        self,
        *,
        project_root: str,
        assets: dict[str, Any] | None = None,
        cache: Any = None,
        security: dict[str, Any] | None = None,
        generator: Callable[..., Any] | None = None,
        seed: int = 12345,
    ) -> None:
        self.project_root = project_root
        self.declarations = assets or {}
        self.cache = cache
        self.security = security or {"allowNetwork": True, "maxDownloadSizeMB": 100}
        self.generator = generator
        self.seed = seed
        self.images: dict[str, Bitmap] = {}
        self.audio: dict[str, AudioBuffer] = {}
        self.meta: dict[str, dict] = {}
        self.missing: set[str] = set()
        self._texts: dict[str, str | None] = {}
        self._luts: dict[str, Any] = {}
        self._svgs: dict[str, Any] = {}

    # ── 引く ────────────────────────────────────────────────

    def get(self, name: str) -> Bitmap | None:
        """ビットマップを引く（デフォーマ・マスクから使います）。"""
        return self.images.get(name)

    def has(self, name: str) -> bool:
        return name in self.declarations

    def describe(self, name: str) -> dict | None:
        return self.meta.get(name)

    def get_audio(self, name: str) -> AudioBuffer | None:
        return self.audio.get(name)

    def text(self, name: str) -> str | None:
        """素材を «テキストとして» 読む。

        3D メッシュ（OBJ）のように、画像でも音でもないものを扱うための入口です。
        宣言のパスから直接読み、内容を覚えます。
        """
        if name in self._texts:
            return self._texts[name]
        declaration = self.declarations.get(name)
        relative = declaration if isinstance(declaration, str) else (declaration or {}).get("path")
        if not relative:
            return None
        try:
            with open(resolve_project_path(self.project_root, relative), encoding="utf-8") as handle:
                content: str | None = handle.read()
        except OSError:
            content = None
        self._texts[name] = content
        return content

    def get_lut(self, name: str):
        """ルック（``.cube`` の 3D LUT）を取り出す。lut エフェクトの入口です。

        エフェクトは «同期» なので、``resolve_all()`` を通っていれば解析済みの
        ものを返し、通っていなければその場で読みます。読めなかったときは
        **«警告して None»**（＝エフェクトは何もしない）です。途中で render が
        止まるより、色が付かないほうがましだからです。

        壊れた LUT を «見逃さない» のは ``movo validate``（strict 読み込み）の
        役目で、そちらは :meth:`load` の中で例外になります。
        """
        if name in self._luts:
            return self._luts[name]
        lut = (self.meta.get(name) or {}).get("lut")
        if lut is None:
            try:
                declaration = self.declarations.get(name)
                relative = declaration if isinstance(declaration, str) else (declaration or {}).get("path")
                if not relative:
                    raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f'素材 "{name}" はファイルの lut ではありません')
                absolute = resolve_project_path(self.project_root, relative)
                self._check_lut_size(absolute, name)
                with open(absolute, encoding="utf-8") as handle:
                    lut = parse_cube_lut(handle.read(), max_bytes=self._lut_limit_bytes(), source=absolute)
            except Exception as error:  # noqa: BLE001 — 何が来ても render は続けたい
                logger.warn(f'lut 素材 "{name}" を読めませんでした: {error}')
                lut = None
        self._luts[name] = lut
        return lut

    def get_svg(self, name: str):
        """ベクタのロゴ（``.svg``）から取り出した «形» を返す。shape レイヤーの入口です。

        LUT と同じ考え方で、読めなければ «警告して None» です。ロゴが 1 つ
        壊れているだけで動画全体が出ないのは割に合わないからです。

        返り値は ``{"subpaths", "viewBox", "width", "height", "stats"}``。
        何を取り込んで何を捨てたかは :mod:`movo.core.svg_path` に書いてあります
        （要点: **«形» だけを見て、実行も外部参照もしません**）。
        """
        if name in self._svgs:
            return self._svgs[name]
        parsed = (self.meta.get(name) or {}).get("svg")
        if parsed is None:
            try:
                declaration = self.declarations.get(name)
                relative = declaration if isinstance(declaration, str) else (declaration or {}).get("path")
                if not relative:
                    raise MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, f'素材 "{name}" はファイルの svg ではありません')
                absolute = resolve_project_path(self.project_root, relative)
                self._check_svg_size(absolute, name)
                with open(absolute, encoding="utf-8") as handle:
                    parsed = extract_svg_shapes(handle.read(), max_bytes=self._svg_limit_bytes())
            except Exception as error:  # noqa: BLE001
                logger.warn(f'svg 素材 "{name}" を読めませんでした: {error}')
                parsed = None
        self._svgs[name] = parsed
        return parsed

    # ── 大きさの見張り ──────────────────────────────────────

    def _lut_limit_bytes(self) -> int:
        """LUT に許すバイト数。ダウンロードの上限と同じ考え方で決めます。

        **0 を «上限なし» に読み替えないこと。** 0 と書いた人は «読ませたくない» のです。
        """
        return max(0, self.security.get("maxDownloadSizeMB", 100)) * 1024 * 1024

    def _check_lut_size(self, absolute: str, name: str) -> None:
        """LUT は **«読む前に»** 大きさを見ます。読んでから弾いても、その時点で
        もうメモリを食っているからです。
        """
        limit = self._lut_limit_bytes()
        size = os.path.getsize(absolute)
        if size > limit:
            raise MovoError(
                ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE,
                f'lut "{name}" は {size / 1024 / 1024:.1f} MB で security.maxDownloadSizeMB'
                f"（{limit / 1024 / 1024:.0f}）を超えています",
                path=f"assets.{name}.path",
            )

    def _svg_limit_bytes(self) -> int:
        """SVG に許すバイト数。既定 2 MB です。

        ロゴには十分すぎるほどで、«巨大な XML を読ませて詰まらせる» という手を封じられます。
        """
        configured = self.security.get("maxSvgSizeMB")
        return max(1, configured) * 1024 * 1024 if configured else DEFAULT_SVG_MAX_BYTES

    def _check_svg_size(self, absolute: str, name: str) -> None:
        limit = self._svg_limit_bytes()
        size = os.path.getsize(absolute)
        if size > limit:
            raise MovoError(
                ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE,
                f'svg "{name}" は {size / 1024 / 1024:.1f} MB で上限（{limit / 1024 / 1024:.0f} MB）を超えています',
                path=f"assets.{name}.path",
            )

    # ── 読み込み ────────────────────────────────────────────

    def resolve_all(self, *, strict: bool = False, generate: bool = True) -> list[MovoError]:
        """宣言された素材をすべて解決する。

        **失敗は集めて返します**（``strict=False`` のとき）。1 つ壊れているだけで
        プレビューが出ないのは困るからです。``strict=True`` は
        ``movo validate`` / CI 向けで、最初の失敗で例外にします。
        """
        errors: list[MovoError] = []
        for name in list(self.declarations.keys()):
            try:
                self.load(name, generate=generate)
            except Exception as error:  # noqa: BLE001
                if strict:
                    raise
                errors.append(
                    error if isinstance(error, MovoError) else MovoError(ErrorCodes.MOVO_ASSET_NOT_FOUND, str(error))
                )
                logger.warn(f'素材 "{name}" を読めませんでした: {error}')
        return errors

    def load(self, name: str, *, generate: bool = True) -> dict | None:
        """素材を 1 つ読む。"""
        if name in self.images or name in self.audio:
            return self.meta.get(name)
        declaration = self.declarations.get(name, _MISSING)

        # メッシュとフォントは «画像でも音でもない» ので、ここでは何もしません。
        # 実際の読み込みは使う側（レンダラ・フォントマネージャ）が行います。
        if isinstance(declaration, dict):
            declared_type = declaration.get("type")
        elif declaration is _MISSING:
            declared_type = None
        else:
            declared_type = infer_spec(str(declaration)).get("type")
        if declared_type in ("mesh", "font"):
            source = declaration if isinstance(declaration, str) else declaration.get("path")
            self.meta[name] = {"type": declared_type, "source": source}
            return self.meta[name]

        if declaration is _MISSING:
            raise MovoError(
                ErrorCodes.MOVO_ASSET_NOT_FOUND,
                f'素材 "{name}" は project.assets に宣言されていません',
                path=f"assets.{name}",
            )

        spec = infer_spec(declaration) if isinstance(declaration, str) else dict(declaration)
        kind = spec.get("type") or infer_spec(spec.get("path") or spec.get("url") or "").get("type") or "image"

        if kind.startswith("ai-"):
            generated = self._generate(name, spec, generate=generate)
            if generated:
                return generated

        if spec.get("url") and not spec.get("path"):
            buffer = self._download(spec["url"], name)
            return self._store(name, kind, buffer, source=spec["url"], spec=spec)

        if not spec.get("path"):
            raise MovoError(
                ErrorCodes.MOVO_ASSET_NOT_FOUND,
                f'素材 "{name}" に "path" も "url" もありません',
                path=f"assets.{name}",
            )

        absolute = resolve_project_path(self.project_root, spec["path"])
        if not os.path.exists(absolute):
            if spec.get("fallback"):
                logger.warn(f'素材 "{name}" がありません。代わりに "{spec["fallback"]}" を使います')
                return self.load(spec["fallback"], generate=generate)
            if "placeholder" in spec or kind == "image":
                logger.warn(f'素材 "{name}" が {spec["path"]} に見つかりません。仮の絵を使います')
                placeholder = create_placeholder(spec.get("placeholder") or {}, name, self.seed)
                self.images[name] = placeholder
                meta = {
                    "name": name,
                    "type": "image",
                    "placeholder": True,
                    "width": placeholder.width,
                    "height": placeholder.height,
                }
                self.meta[name] = meta
                self.missing.add(name)
                return meta
            raise MovoError(
                ErrorCodes.MOVO_ASSET_NOT_FOUND,
                f"素材ファイルが見つかりません: {spec['path']}",
                path=f"assets.{name}.path",
            )

        # LUT と SVG は «読む前に» 大きさを見ます（外からもらうテキストなので）。
        if kind == "lut":
            self._check_lut_size(absolute, name)
        if kind == "svg":
            self._check_svg_size(absolute, name)
        with open(absolute, "rb") as handle:
            buffer = handle.read()
        return self._store(name, kind, buffer, source=absolute, spec=spec)

    def _store(self, name: str, kind: str, buffer: bytes, *, source: str, spec: dict) -> dict:
        if kind == "audio" or _AUDIO_EXTENSIONS.search(source or ""):
            audio = self._decode_audio(buffer, source, name)
            self.audio[name] = audio
            meta = {
                "name": name,
                "type": "audio",
                "sampleRate": audio.sample_rate,
                "duration": audio.duration,
                "source": source,
            }
            self.meta[name] = meta
            return meta
        if kind == "video" or _VIDEO_EXTENSIONS.search(source or ""):
            meta = {"name": name, "type": "video", "source": source, "spec": spec}
            self.meta[name] = meta
            return meta
        if kind == "data":
            try:
                value: Any = json.loads(buffer.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                value = buffer.decode("utf-8", errors="replace")
            meta = {"name": name, "type": "data", "value": value, "source": source}
            self.meta[name] = meta
            return meta
        if kind == "lyrics":
            # 時刻付きの歌詞（.lrc / .srt / .vtt / JSON）。形式は中身から当てます。
            # 拡張子は当てになりません（.txt に LRC が入っていることがよくあります）。
            lines = parse_lyrics(buffer.decode("utf-8"), file=source)
            meta = {"name": name, "type": "lyrics", "lines": lines, "count": len(lines), "source": source}
            self.meta[name] = meta
            return meta
        if kind == "font":
            meta = {"name": name, "type": "font", "source": source}
            self.meta[name] = meta
            return meta
        if kind == "svg":
            # ここで «形» に直しておきます。フレームごとに XML を読み直すのは無駄ですし、
            # 壊れた SVG は render を始める前に分かるほうが親切です。
            parsed = extract_svg_shapes(buffer.decode("utf-8"), max_bytes=self._svg_limit_bytes())
            meta = {
                "name": name,
                "type": "svg",
                "svg": parsed,
                "paths": parsed["stats"]["paths"],
                "shapes": parsed["stats"]["shapes"],
                "subpaths": len(parsed["subpaths"]),
                "viewBox": parsed["viewBox"],
                "source": source,
            }
            self.meta[name] = meta
            self._svgs[name] = parsed
            return meta
        if kind == "lut":
            # ここで解析しておくと、壊れた `.cube` は render を始める前に分かります。
            lut = parse_cube_lut(buffer.decode("utf-8"), max_bytes=self._lut_limit_bytes(), source=source)
            meta = {"name": name, "type": "lut", "lut": lut, "size": lut.size, "title": lut.title, "source": source}
            self.meta[name] = meta
            self._luts[name] = lut
            return meta

        bitmap = decode_image(buffer, source_path=source)
        self.images[name] = bitmap
        meta = {"name": name, "type": "image", "width": bitmap.width, "height": bitmap.height, "source": source}
        self.meta[name] = meta
        return meta

    def _decode_audio(self, buffer: bytes, source: str | None, name: str) -> AudioBuffer:
        try:
            return decode_wav(buffer)
        except MovoError as error:
            converted = _convert_audio_with_ffmpeg(source)
            if converted is not None:
                return converted
            raise MovoError(
                ErrorCodes.MOVO_ASSET_DECODE_FAILED,
                f'音の素材 "{name}" を復号できませんでした: {error.reason}',
                hint="ffmpeg を入れるか、WAV で渡してください",
                cause=error,
            ) from error

    def _generate(self, name: str, spec: dict, *, generate: bool) -> dict | None:
        if self.generator is None:
            logger.warn(f'素材 "{name}" は AI 生成が要りますが、生成器が設定されていません。仮の絵を使います')
            placeholder = create_placeholder(spec.get("placeholder") or {}, name, self.seed)
            self.images[name] = placeholder
            meta = {
                "name": name,
                "type": "image",
                "placeholder": True,
                "generated": False,
                "width": placeholder.width,
                "height": placeholder.height,
            }
            self.meta[name] = meta
            return meta
        result = self.generator(name=name, spec=spec, store=self, generate=generate)
        if not result:
            return None
        if result.get("bitmap") is not None:
            bitmap = result["bitmap"]
            self.images[name] = bitmap
            meta = {
                "name": name,
                "type": "image",
                "generated": True,
                "placeholder": bool(result.get("placeholder")),
                "provider": result.get("provider"),
                "width": bitmap.width,
                "height": bitmap.height,
                "source": result.get("path"),
            }
            self.meta[name] = meta
            return meta
        if result.get("parts"):
            for part_name, bitmap in result["parts"].items():
                self.images[f"{name}.{part_name}"] = bitmap
            meta = {
                "name": name,
                "type": "ai-character",
                "generated": True,
                "parts": list(result["parts"].keys()),
                "rig": result.get("rig"),
            }
            self.meta[name] = meta
            return meta
        return None

    def _download(self, url: str, name: str) -> bytes:
        """URL から取ってくる。**許可されていなければ取りに行きません。**"""
        if self.security.get("allowNetwork") is False:
            raise MovoError(
                ErrorCodes.MOVO_NETWORK_DENIED,
                f'"{name}" のダウンロードは security.allowNetwork で止められています',
                path=f"assets.{name}.url",
            )
        cache_key = sha256(url)
        if self.cache is not None:
            cached = self.cache.read_buffer("downloads", cache_key)
            if cached:
                return cached
        limit_mb = self.security.get("maxDownloadSizeMB", 100)
        logger.verbose(f"ダウンロード中 {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — URL は利用者が書いたもの
                declared = int(response.headers.get("content-length") or 0)
                if declared and declared > limit_mb * 1024 * 1024:
                    raise MovoError(
                        ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE,
                        f'"{name}" は {declared / 1024 / 1024:.1f} MB で security.maxDownloadSizeMB'
                        f"（{limit_mb}）を超えています",
                    )
                # **上限 + 1 バイトまでしか読みません。** Content-Length を偽った
                # 相手にメモリを食い潰されないようにするためです。
                buffer = response.read(limit_mb * 1024 * 1024 + 1)
        except MovoError:
            raise
        except Exception as error:  # noqa: BLE001
            raise MovoError(
                ErrorCodes.MOVO_ASSET_NOT_FOUND, f'"{name}" のダウンロードに失敗しました: {error}', cause=error
            ) from error
        if len(buffer) > limit_mb * 1024 * 1024:
            raise MovoError(
                ErrorCodes.MOVO_DOWNLOAD_TOO_LARGE,
                f'"{name}" は {limit_mb} MB（security.maxDownloadSizeMB）を超えています',
            )
        if self.cache is not None:
            self.cache.write_buffer("downloads", cache_key, buffer)
        return buffer

    def stats(self) -> dict[str, int]:
        return {"images": len(self.images), "audio": len(self.audio), "placeholders": len(self.missing)}


class _Missing:
    """«宣言そのものが無い» を表す番人。``None`` と区別するために要ります。"""


_MISSING = _Missing()


def _convert_audio_with_ffmpeg(source: str | None) -> AudioBuffer | None:
    if not source or not os.path.exists(source):
        return None
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    directory = tempfile.mkdtemp(prefix="movo-audio-")
    output = os.path.join(directory, "out.wav")
    try:
        result = subprocess.run(
            [ffmpeg["path"], "-y", "-loglevel", "error", "-i", source, "-ar", "48000", "-ac", "2", output],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(output):
            with open(output, "rb") as handle:
                return decode_wav(handle.read())
        return None
    except (OSError, subprocess.SubprocessError, MovoError):
        return None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def infer_spec(path_or_url: str) -> dict[str, str]:
    """パスや URL の «見た目» から種類を当てる。

    拡張子で先に振り分けるのは、``.obj`` や ``.cube`` を画像として復号しようと
    すると «形式が分からない» と警告が出てしまうためです。
    """
    value = str(path_or_url or "")
    is_url = bool(re.match(r"^https?://", value, re.IGNORECASE))
    kind = "image"
    if _AUDIO_EXTENSIONS.search(value):
        kind = "audio"
    elif _VIDEO_EXTENSIONS.search(value):
        kind = "video"
    elif re.search(r"\.json$", value, re.IGNORECASE):
        kind = "data"
    elif re.search(r"\.(ttf|otf|ttc)$", value, re.IGNORECASE):
        kind = "font"
    # 3D メッシュ。中身はレンダラが text() で読みます。
    elif re.search(r"\.(obj|mtl)$", value, re.IGNORECASE):
        kind = "mesh"
    # カラーグレーディングのルック（3D LUT）。中身はテキストです。
    elif re.search(r"\.cube$", value, re.IGNORECASE):
        kind = "lut"
    # ベクタのロゴ。形（パス）だけを取り出して覚えます（svg_path.py 参照）。
    elif re.search(r"\.svg$", value, re.IGNORECASE):
        kind = "svg"
    elif _IMAGE_EXTENSIONS.search(value):
        kind = "image"
    return {"type": kind, "url": value} if is_url else {"type": kind, "path": value}


def create_placeholder(options: dict, name: str, seed: int = 12345) -> Bitmap:
    """素材や API キーが無くても絵が出るようにする «仮の絵»。

    模様は **素材の名前から決めます。** そうすると「どの素材が欠けているか」が
    レンダリング結果を見ただけで分かり、しかもレイアウトは崩れません。

    JS 版と同じ市松模様・同じ色になるよう、乱数も色変換も JS 版の式のままです。
    塗り自体は NumPy の一括代入なので、512x512 でも 0.4 ミリ秒です。
    """
    width = max(4, int(np.floor((options.get("width") or 512) + 0.5)))
    height = max(4, int(np.floor((options.get("height") or 512) + 0.5)))
    bitmap = Bitmap(width, height)
    random = create_random((seed ^ hash_string(str(name))) & 0xFFFFFFFF)
    hue = random() * 360
    base = _hsl_to_rgb_local(hue / 360, 0.45, 0.55)
    alt = _hsl_to_rgb_local(((hue + 40) % 360) / 360, 0.45, 0.42)
    tile = max(8, int(np.floor(min(width, height) / 8 + 0.5)))

    xs = np.arange(width) // tile
    ys = np.arange(height) // tile
    checker = ((ys[:, None] + xs[None, :]) % 2) == 0
    palette = np.array([alt, base], np.uint8)  # False → alt, True → base
    bitmap.data[..., :3] = palette[checker.astype(np.int8)]
    bitmap.data[..., 3] = 200 if options.get("opaque") is False else 255

    # 斜めの線を引くと «仮の絵» だと一目で分かります。
    x = np.arange(width)
    y = (x / width * height).astype(np.int64)
    for t in (-1, 0, 1):
        yy = y + t
        valid = (yy >= 0) & (yy < height)
        bitmap.data[yy[valid], x[valid], 0] = 255
        bitmap.data[yy[valid], x[valid], 1] = 255
        bitmap.data[yy[valid], x[valid], 2] = 255
    return bitmap


def _hsl_to_rgb_local(h: float, s: float, lightness: float) -> tuple[int, int, int]:
    """JS 版の ``hslToRgbLocal``。

    :mod:`movo.core.color` の :func:`hsl_to_rgb` と式は同じですが、**clamp が
    入っていません**。仮の絵の色を JS 版と 1 ビットも変えないため、あえて
    こちらを写しています。
    """

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        tt = t
        if tt < 0:
            tt += 1
        if tt > 1:
            tt -= 1
        if tt < 1 / 6:
            return p + (q - p) * 6 * tt
        if tt < 1 / 2:
            return q
        if tt < 2 / 3:
            return p + (q - p) * (2 / 3 - tt) * 6
        return p

    q = lightness * (1 + s) if lightness < 0.5 else lightness + s - lightness * s
    p = 2 * lightness - q
    import math as _math

    return (
        _math.floor(hue_to_rgb(p, q, h + 1 / 3) * 255 + 0.5),
        _math.floor(hue_to_rgb(p, q, h) * 255 + 0.5),
        _math.floor(hue_to_rgb(p, q, h - 1 / 3) * 255 + 0.5),
    )
