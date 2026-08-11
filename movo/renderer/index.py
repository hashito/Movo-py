"""movo-renderer — フレームを描くところ。

移植元: ``packages/renderer/src/index.js``

1 フレームの流れ:

1. その時刻までシミュレーションを進める（決定的に）
2. 見えているレイヤーのアニメーション値をすべて解決する
3. レイヤーごとに «中身» のビットマップを作る（画像・文字・図形・粒…）
4. 変形とエフェクトを書いた順に通す
5. できたメッシュをフレームバッファへ描く

既定は software の ``canvas-2d``。``webgl`` / ``gpu`` を指定しても、警告を
出してこれにフォールバックします（仕様の原則 8・9）。

── このディレクトリの地図 ────────────────────────────────────

======================  =======================================================
``index.py``            レイヤー 1 枚を描く流れ（この本体）とシーンの合成
``simulation.py``       物理・ソフトチェーン・パーティクルの «作る/進める»
``frame_history.py``    frameEcho（実フレームの残像）
``layers_builtin.py``   組み込みレイヤーの «中身» を作る処理
``layers_generator.py`` 生成レイヤー（星空・線・ネオン・水面・3D・図形アニメ）
``effects.py``          エフェクトの入口と基本的な加工
``text.py``             文字組み
``raster.py``           ポリゴン／テクスチャ三角形のラスタライズ
``helpers.py``          どこからでも使う小さな純粋関数
======================  =======================================================

── 名前の食い違いについて ───────────────────────────────────

**JSON から来る辞書のキーは camelCase のまま**です（``localStart`` ``zIndex``
``scaleX``…）。プロジェクト JSON を JS 版と共用するので、途中で snake_case に
直すと «JSON の綴り» と «内部の綴り» を二重に覚えることになります。

一方 **Python 側で新しく作る «関数の返り値の辞書»** は、先に移植された
モジュールに合わせます。移植が並列に進んだせいでここが揺れているので、
呼ぶ側（このファイル）で吸収しています。実際に見つかった食い違い:

- ``movo.renderer.text.render_text()`` → ``offset_x`` / ``offset_y``（snake_case）
- ``movo.renderer.shapes.render_shape()`` → ``box_width`` / ``origin_x``（snake_case）
- ``movo.deformer.apply_modifiers()`` → ``offsetX`` / ``boxWidth``（**camelCase**）
- ``movo.deformer.mesh.bake_mesh()`` → ``offsetX`` / ``width``（**camelCase**）

``_render_content`` が返す «中身» は **snake_case にそろえました**
（``box_width`` ``anchor_width`` ``origin_x`` ``scale``）。中身を作る側の 2 つ
（text / shapes）が既に snake_case で、そちらに合わせるほうが直す量が
少なかったためです。
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from movo.animation.easing import get_easing
from movo.animation.keyframes import sample_keyframes
from movo.animation.resolver import apply_animations, resolve_animated, resolve_number
from movo.cli.console import logger
from movo.core.bitmap import Bitmap
from movo.core.math import Mat2D, clamp, js_round, lerp, sample_polyline, to_degrees, to_radians
from movo.deformer import apply_modifiers
from movo.deformer.mask import build_mask_field, sample_field
from movo.deformer.mesh import Mesh, bake_mesh
from movo.expression import ExpressionEngine
from movo.physics import World
from movo.renderer.effect_graph import apply_effect_graph
from movo.renderer.effects import Random, apply_effect, value_noise_2d
from movo.renderer.font import FontManager
from movo.renderer.frame_history import draw_frame_echo, push_frame_history
from movo.renderer.helpers import find_layer, within_range
from movo.renderer.plane3d import camera_basis, draw_plane, is_plane3d, project_plane
from movo.renderer.raster import (
    draw_bitmap,
    expand_region_with,
    fill_coverage,
    parse_color,
    rasterize_contours,
)
from movo.renderer.shapes import render_shape
from movo.renderer.simulation import (
    advance_simulation,
    register_scene_bodies,
    reset_simulation,
)
from movo.renderer.text import (
    apply_karaoke_fill,
    format_counter,
    render_animated_text,
    render_text,
    render_text_on_path,
    resolve_text_style,
)
from movo.renderer.text_extras import draw_text_box, resolve_padding
from movo.timeline import is_layer_active, scenes_at

RENDERER_KINDS = ["canvas-2d", "svg", "webgl", "gpu", "headless-browser", "custom"]

DEFAULT_TRANSFORM = {
    "x": 0,
    "y": 0,
    "rotation": 0,
    "scale": 1,
    "scaleX": 1,
    "scaleY": 1,
    "skewX": 0,
    "skewY": 0,
    "opacity": 1,
    "anchorX": 0.5,
    "anchorY": 0.5,
}

TAU = math.pi * 2

# `"followLayer": "id"` を JSON 全文から拾う正規表現。JS 版と同じ手です。
# 軌跡を «使うレイヤーだけ» 覚えるためで、使わないプロジェクトでは 1 件も積みません。
_FOLLOW_LAYER_RE = re.compile(r'"followLayer"\s*:\s*"([^"]+)"')


class Renderer:
    """フレームを描く本体。

    :param project: 正規化済みのプロジェクト
    :param timeline: ``movo.timeline.build_timeline()`` の結果
    :param assets: ``movo.core.assets.AssetStore``
    :param audio_envelope: ``{"levels": …, "bands": […]}``（音に反応する式が読む）

    ``movo/cli/pipeline.py`` が **キーワード引数**で呼びます。JS 版は
    «オプション辞書 1 つ» でしたが、呼ぶ側に合わせました（辞書 1 つで渡す
    書き方も ``options=`` で受けます。移植中の他のコードが JS の書き方の
    まま呼んでも黙って壊れないようにするためです）。
    """

    def __init__(
        self,
        options: dict | None = None,
        *,
        project: dict | None = None,
        timeline: dict | None = None,
        assets=None,
        plugins=None,
        cache=None,
        font_manager: FontManager | None = None,
        font_dirs=None,
        project_root: str | None = None,
        audio_envelope=None,
        audio=None,
        **extra,
    ) -> None:
        options = dict(options or {})
        options.update({k: v for k, v in extra.items() if v is not None})

        def pick(name: str, camel: str, value):
            if value is not None:
                return value
            return options.get(name, options.get(camel))

        self.project = pick("project", "project", project) or {}
        self.timeline = pick("timeline", "timeline", timeline) or {}
        self.assets = pick("assets", "assets", assets)
        self.plugins = pick("plugins", "plugins", plugins)
        self.cache = pick("cache", "cache", cache)
        self.project_root = pick("project_root", "projectRoot", project_root) or "."
        self.audio_envelope = pick("audio_envelope", "audioEnvelope", audio_envelope)
        self.audio = pick("audio", "audio", audio)

        render_settings = self.project.get("render") or {}
        requested = render_settings.get("renderer") or "canvas-2d"
        self.kind = "canvas-2d"
        if requested != "canvas-2d":
            plugin = self._plugin("renderer", requested)
            if plugin:
                self.kind = requested
            else:
                logger.warn(
                    f'renderer "{requested}" is not available in this build; '
                    "falling back to canvas-2d (software)"
                )

        deterministic = self.project.get("deterministic") or {}
        project_meta = self.project.get("project") or {}
        seed = deterministic.get("seed")
        if seed is None:
            seed = project_meta.get("seed")
        self.seed = 12345 if seed is None else int(seed)
        self.bpm = project_meta.get("bpm")
        self.random = Random(self.seed)
        self.engine = ExpressionEngine(seed=self.seed)
        self.font_manager = pick("font_manager", "fontManager", font_manager) or FontManager(
            project_root=self.project_root,
            fonts=self.project.get("fonts"),
            font_dirs=pick("font_dirs", "fontDirs", font_dirs) or (),
        )

        self.render_scale = int(clamp(js_round(render_settings.get("superSample") or 1), 1, 4))
        deformation = render_settings.get("deformation") or {}
        self.mesh_resolution = max(2, js_round(deformation.get("meshResolution") or 20))
        self.alpha_outline = render_settings.get("alphaOutline") is not False

        self.worlds: dict = {}
        self.bodies: dict = {}
        self.soft_chains: dict = {}
        self.particles: dict = {}
        self.rigs: dict = {}
        self.videos: dict = {}
        self.layer_bitmaps: dict = {}
        self.layer_states: dict = {}

        # 2.5D カメラ（project.camera）。書かれていなければ従来どおり動く。
        self.camera = self.project.get("camera")

        scenes_json = json.dumps(self.project.get("scenes") or [], ensure_ascii=False)
        # レイヤーが通った跡（linePath.followLayer 用）。使うときだけ積む。
        self.motion_trails: dict = {}
        self._tracked_layers = set(_FOLLOW_LAYER_RE.findall(scenes_json))
        # 合成済みフレームの履歴（slitScan 用）。使うレイヤーがあるときだけ積む。
        self.frame_stack: list[Bitmap] = []
        self.frame_stack_limit = int(clamp(js_round(render_settings.get("frameHistory") or 16), 1, 240))
        self._needs_frame_stack = '"slitScan"' in scenes_json
        self.layer_history: dict = {}
        self.frame_history_limit = int(clamp(js_round(render_settings.get("frameHistory") or 16), 1, 240))
        self.history_bytes = 0
        self._history_warned: set = set()
        self._memory_warned = False
        self.current_frame = -1
        self.simulated_frame = -1
        self.warnings: list = []

        # ── Python 版だけの持ち物 ──────────────────────────────
        # `Body` は `__slots__` を持つので «初期状態» や `fixedRotation` を
        # 属性として貼れません。レンダラー側の辞書・集合で覚えます。
        self._initial_states: dict = {}
        self._fixed_rotation: set = set()
        self._shape_targets: dict = {}
        self._warmed_up = False
        self._depth_buffers: dict = {}
        self._plane_depths = None
        self._frame_canvas: Bitmap | None = None
        self._scene_target: Bitmap | None = None
        self._frame_buffer_count = 0
        self._effect_frame_history: list = []

    # ── 小さな入口 ────────────────────────────────────────────

    def _plugin(self, kind: str, name: str):
        """プラグインの登録を引く。無ければ None。"""
        plugins = self.plugins
        if not plugins:
            return None
        factory = plugins.get(kind) if isinstance(plugins, dict) else getattr(plugins, kind, None)
        return factory(name) if callable(factory) else None

    @property
    def width(self) -> int:
        return self.timeline["width"]

    @property
    def height(self) -> int:
        return self.timeline["height"]

    def prepare(self) -> "Renderer":
        """物理ワールド・粒・リグ・動画を用意する。"""
        for scene in self.timeline["scenes"]:
            config = {**(self.project.get("physicsWorld") or {}), **(scene.get("physicsWorld") or {})}
            if config.get("enabled") is False:
                continue
            world = World(
                {
                    "gravity": config.get("gravity"),
                    "timeStep": config.get("timeStep", 1 / 60),
                    "subSteps": config.get("subSteps", 2),
                    "iterations": config.get("iterations", 8),
                    "bounds": config.get("bounds"),
                    "pixelsPerMeter": config.get("pixelsPerMeter", 100),
                }
            )
            self.worlds[scene["id"]] = world
            register_scene_bodies(self, scene, world, config)
        reset_simulation(self)
        return self

    def _audio_at(self, time: float) -> dict:
        """その時刻の音量と 3 帯域。音に反応する式・モジュレーターが読みます。

        ⚠ ``movo.audio.analyze_envelope`` の ``bands`` は **`(3, フレーム数)` の
        NumPy 配列**です（JS 版は «配列 3 本の配列»）。`bands or []` と書くと
        「真理値が曖昧」で落ちるので、**None かどうかだけを見ます。**
        """
        envelope = self.audio_envelope
        if envelope is None:
            return {"level": 0, "bands": [0, 0, 0]}
        levels = envelope["levels"]
        if len(levels) == 0:
            return {"level": 0, "bands": [0, 0, 0]}
        index = int(clamp(js_round(time * self.timeline["fps"]), 0, len(levels) - 1))
        bands = envelope.get("bands")
        if bands is None:
            band_values: list[float] = []
        else:
            band_values = [float(band[index]) if index < len(band) else 0.0 for band in bands]
        return {"level": float(levels[index]), "bands": band_values}

    # ── 式から見える値 ────────────────────────────────────────

    def _context_for(self, layer: dict, scene: dict, scene_time: float, global_time: float, body) -> dict:
        layer_time = scene_time - (layer.get("localStart") or 0)
        audio = self._audio_at(global_time)

        def layer_accessor(layer_id: str) -> dict:
            state = self.layer_states.get(layer_id)
            if state is None:
                logger.verbose(
                    f'expression referenced layer "{layer_id}" before it was rendered this frame'
                )
                return {"transform": dict(DEFAULT_TRANSFORM), "width": 0, "height": 0}
            return state

        # JS は関数オブジェクトに «今のレイヤーの状態» を貼り付けて
        # `layer.x` と `layer("id").x` の両方を書けるようにしています。
        # Python の関数には属性を貼れますが、式エンジンから見えるのは
        # 呼び出しか添字なので、**呼び出し可能な辞書** を渡します。
        current = self.layer_states.get(layer.get("id")) or {
            "id": layer.get("id"),
            "transform": dict(DEFAULT_TRANSFORM),
        }
        layer_scope = _CallableScope(layer_accessor, current)

        bpm = self.bpm if self.bpm is not None else 120
        functions = self.engine.functions

        def wiggle(frequency=1, amplitude=1, seed_offset=0):
            return functions["wiggleAt"](scene_time, frequency, amplitude, seed_offset)

        def beat(division=1, decay=6):
            return functions["beatAt"](scene_time, bpm, division, decay)

        def beat_index(division=1):
            return functions["beatIndexAt"](scene_time, bpm, division)

        def beat_pulse(pulse_bpm=None, *rest):
            # beatPulse(bpm, decay) と beatPulse(bpm, division, decay) の両方を受ける。
            # 隣の beat() が (division, decay) なので 3 引数で書きたくなるのが自然で、
            # 2 引数しか受けないと «division のつもりの 1» が decay として通り、
            # 減衰しないパルスになって気付きにくい。
            if len(rest) >= 2:
                division, decay = rest[0], rest[1]
            else:
                division, decay = 1, (rest[0] if rest else 8)
            return functions["beatAt"](
                scene_time, pulse_bpm if pulse_bpm is not None else bpm, division or 1, decay or 8
            )

        if body is not None:
            speed = math.hypot(body.velocity.x, body.velocity.y)
            physics = {
                "velocity": {"x": body.velocity.x, "y": body.velocity.y},
                "speed": speed,
                "position": {"x": body.position.x, "y": body.position.y},
                "angle": to_degrees(body.angle),
                "angularVelocity": body.angular_velocity,
            }
        else:
            physics = {
                "velocity": {"x": 0, "y": 0},
                "speed": 0,
                "position": {"x": 0, "y": 0},
                "angle": 0,
                "angularVelocity": 0,
            }

        project_meta = self.project.get("project") or {}
        scope = {
            "time": scene_time,
            "sceneTime": scene_time,
            "globalTime": global_time,
            "layerTime": layer_time,
            "frame": js_round(global_time * self.timeline["fps"]),
            "fps": self.timeline["fps"],
            "duration": self.timeline["duration"],
            "layer": layer_scope,
            "scene": {
                "id": scene["id"],
                "start": scene["start"],
                "duration": scene["duration"],
                "index": scene["index"],
            },
            # 曲の区間（`from: { section }` を書いたシーンだけ入ります）
            "section": _section_scope(scene, global_time),
            "project": {
                "name": project_meta.get("name") or "untitled",
                "width": self.width,
                "height": self.height,
                "fps": self.timeline["fps"],
                "duration": self.timeline["duration"],
                "seed": self.seed,
            },
            "variables": self.project.get("variables") or {},
            "audio": audio,
            "mouse": {"x": 0, "y": 0},
            "seed": self.seed,
            "bpm": bpm,
            "wiggle": wiggle,
            "beat": beat,
            "beatIndex": beat_index,
            "beatPulse": beat_pulse,
            "physics": physics,
        }

        return {
            "time": scene_time,
            "engine": self.engine,
            "scope": scope,
            "seed": self.seed,
            "audio": audio,
            "bpm": self.bpm,
            "fps": self.timeline["fps"],
            "path": layer.get("id"),
        }

    # ── トランスフォーム ──────────────────────────────────────

    def _resolve_transform(self, layer: dict, ctx: dict) -> dict:
        raw = layer.get("transform") or {}
        # group / composition は子を絶対座標で描くので、アンカーの既定は左上。
        # `x: 0, y: 0` が «ずらさない» を意味するようにするためです。
        default_anchor = 0 if layer.get("type") in ("group", "composition") else DEFAULT_TRANSFORM["anchorX"]
        uniform_scale = resolve_number(raw.get("scale"), ctx, 1) if raw.get("scale") is not None else None
        base_scale = 1 if uniform_scale is None else uniform_scale
        transform = {
            "x": resolve_number(raw.get("x"), ctx, DEFAULT_TRANSFORM["x"]),
            "y": resolve_number(raw.get("y"), ctx, DEFAULT_TRANSFORM["y"]),
            "rotation": resolve_number(raw.get("rotation"), ctx, DEFAULT_TRANSFORM["rotation"]),
            "scaleX": resolve_number(raw.get("scaleX"), ctx, base_scale),
            "scaleY": resolve_number(raw.get("scaleY"), ctx, base_scale),
            "skewX": resolve_number(raw.get("skewX"), ctx, 0),
            "skewY": resolve_number(raw.get("skewY"), ctx, 0),
            "opacity": clamp(resolve_number(raw.get("opacity"), ctx, 1), 0, 1),
            "anchorX": resolve_number(raw.get("anchorX"), ctx, default_anchor),
            "anchorY": resolve_number(raw.get("anchorY"), ctx, default_anchor),
            "width": resolve_number(raw.get("width"), ctx, 0) if raw.get("width") is not None else None,
            "height": resolve_number(raw.get("height"), ctx, 0) if raw.get("height") is not None else None,
        }
        if uniform_scale is not None and raw.get("scaleX") is None:
            transform["scaleX"] = uniform_scale
        if uniform_scale is not None and raw.get("scaleY") is None:
            transform["scaleY"] = uniform_scale

        # 3 次元の傾き。書かれているときだけ «板を 3D に置く» 経路へ入る。
        transform["rotationX"] = resolve_number(raw.get("rotationX"), ctx, 0)
        transform["rotationY"] = resolve_number(raw.get("rotationY"), ctx, 0)

        # 2.5D カメラ。レイヤーの z とカメラ位置から «縮尺» と «視差» を出す。
        if self.camera and raw.get("z") is not None:
            transform["z"] = resolve_number(raw.get("z"), ctx, 0)
            # 板を 3D に置く場合は投影を drawPlane 側でまとめて行う。ここで
            # 2.5D の縮尺を先に掛けると «二重に» 遠近が付いてしまう。
            if not is_plane3d(transform):
                self._apply_camera(transform, ctx)
            else:
                self._apply_camera_depth_of_field(transform, ctx)
        return transform

    def _apply_camera(self, transform: dict, ctx: dict) -> None:
        """カメラの位置とレイヤーの z から、縮尺と視差を transform に反映する。

        完全な 3D ではなく «2.5D»（板が z 方向に並んでいるだけ）です。
        イラスト MV のパララックスはこれで足ります。
        """
        camera = self.camera
        cam_x = resolve_number(camera.get("x"), ctx, self.width / 2)
        cam_y = resolve_number(camera.get("y"), ctx, self.height / 2)
        cam_z = resolve_number(camera.get("z"), ctx, -1200)
        fov = clamp(resolve_number(camera.get("fov"), ctx, 45), 1, 179)
        focal = self.height / 2 / math.tan(fov * math.pi / 360)
        distance = transform["z"] - cam_z
        if distance <= 1:
            return  # カメラの後ろ・真上にある板は動かさない
        reference = max(1, resolve_number(camera.get("referenceDistance"), ctx, focal))
        k = reference / distance

        centre_x = self.width / 2
        centre_y = self.height / 2
        transform["x"] = centre_x + (transform["x"] - cam_x) * k
        transform["y"] = centre_y + (transform["y"] - cam_y) * k
        transform["scaleX"] *= k
        transform["scaleY"] *= k

        self._apply_camera_depth_of_field(transform, ctx)

    def _apply_camera_depth_of_field(self, transform: dict, ctx: dict) -> None:
        """被写界深度だけを掛ける（3D に置いた板とも共有する）。"""
        dof = (self.camera or {}).get("dof")
        if not dof or dof.get("enabled") is False:
            return
        focus_z = resolve_number(dof.get("focusZ"), ctx, 0)
        aperture = resolve_number(dof.get("aperture"), ctx, 0.4)
        blur = abs((transform.get("z") or 0) - focus_z) * aperture * 0.02
        if blur > 0.4:
            transform["cameraBlur"] = min(blur, 40)

    def _depth_for(self, destination: Bitmap, layer: dict):
        """深度バッファを描画先ごとに 1 枚だけ用意する。

        3D の板を使うレイヤーが 1 枚も無ければ確保しないので、2D だけの
        プロジェクトには余計なメモリも時間も掛かりません。
        """
        if layer.get("depthTest") is False and layer.get("depthWrite") is False:
            return None
        key = id(destination)
        buffer = self._depth_buffers.get(key)
        if buffer is None:
            buffer = np.full((destination.height, destination.width), np.inf, np.float32)
            self._depth_buffers[key] = buffer
        return {
            "buffer": buffer,
            "z": self._plane_depths,
            "test": layer.get("depthTest") is not False,
            "write": layer.get("depthWrite") is not False,
        }

    def _camera3d_for(self, ctx: dict) -> dict:
        """3D に置いた板を描くために、カメラの位置・向き・基準距離をまとめる。"""
        camera = self.camera or {}
        eye = {
            "x": resolve_number(camera.get("x"), ctx, self.width / 2),
            "y": resolve_number(camera.get("y"), ctx, self.height / 2),
            "z": resolve_number(camera.get("z"), ctx, -1200),
        }
        fov = clamp(resolve_number(camera.get("fov"), ctx, 45), 1, 179)
        focal = self.height / 2 / math.tan(fov * math.pi / 360)
        look_at = None
        if camera.get("lookAt"):
            look_at = {
                "x": resolve_number(camera["lookAt"].get("x"), ctx, self.width / 2),
                "y": resolve_number(camera["lookAt"].get("y"), ctx, self.height / 2),
                "z": resolve_number(camera["lookAt"].get("z"), ctx, 0),
            }
        return {
            "eye": eye,
            "basis": camera_basis(eye, look_at, camera.get("up")),
            "referenceDistance": max(1, resolve_number(camera.get("referenceDistance"), ctx, focal)),
            "centreX": self.width / 2,
            "centreY": self.height / 2,
        }

    def _resolve_modifiers(self, items, ctx: dict) -> dict:
        """変形／エフェクトを «一覧» と «id 引き» の両方で返す。

        ``by_id`` は ``animations`` が ``modifiers.<id>.amount`` のように
        指してくるためのものです。**キーは ``by_id``（snake_case）** に
        しました。JS の ``byId`` はここでしか使われないので、Python 側の
        命名にそろえたほうが読みやすいためです。
        """
        resolved = []
        by_id: dict = {}
        for modifier in items or []:
            if not modifier:
                continue
            entry = resolve_animated(modifier, ctx, None) or {}
            if not isinstance(entry, dict):
                continue
            entry["type"] = modifier.get("type")
            if modifier.get("mask"):
                entry["mask"] = resolve_animated(modifier["mask"], ctx, None)
            resolved.append(entry)
            if modifier.get("id"):
                by_id[modifier["id"]] = entry
        return {"list": resolved, "by_id": by_id}

    # ── 1 フレーム ────────────────────────────────────────────

    def advance_simulation(self, frame_index: int) -> None:
        """指定フレームまでシミュレーションを進める（実体は simulation.py）。"""
        advance_simulation(self, frame_index)

    def render_frame(self, frame_index: int) -> Bitmap:
        """1 フレーム描く。戻り値は実寸の RGBA ビットマップ。"""
        # 連続でないフレームを描くと残像がつながらないので履歴を捨てる
        if self.layer_history and frame_index != self.current_frame + 1:
            self.layer_history.clear()
            self.history_bytes = 0
            self.motion_trails.clear()
            self.frame_stack.clear()
        self.current_frame = frame_index
        advance_simulation(self, frame_index)
        time = frame_index / self.timeline["fps"]
        scale = self.render_scale
        canvas = Bitmap(self.width * scale, self.height * scale)
        background = parse_color(self.timeline.get("background") or "#000000")
        if background[3] > 0:
            canvas.fill((int(background[0]), int(background[1]), int(background[2]), _u8(background[3] * 255)))

        self.layer_bitmaps.clear()
        self.layer_states.clear()
        # 3D の前後関係を «描いた順» ではなく «奥行き» で決めるための深度バッファ。
        # 使うレイヤーが 1 枚も無ければ確保しない。
        self._depth_buffers = {}

        self._frame_canvas = canvas  # "frame": シーンをまたいだ合成先
        self._frame_buffer_count = 0
        # slitScan 用に «過去に描いたフレーム» を渡す
        self._effect_frame_history = self.frame_stack

        base_matrix = Mat2D.scale(Mat2D.identity(), scale, scale)
        for scene in scenes_at(self.timeline, time):
            scene_time = time - scene["start"]
            transition = scene.get("transition")
            target = Bitmap(canvas.width, canvas.height) if transition else canvas
            self._scene_target = target  # "scene": このシーンの描画先
            if scene.get("background"):
                scene_bg = parse_color(scene["background"])
                if scene_bg[3] > 0:
                    region = rasterize_contours(
                        [[0, 0, target.width, 0, target.width, target.height, 0, target.height]],
                        target.width,
                        target.height,
                    )
                    fill_coverage(target, region, scene_bg, 1)
            for layer in scene["layers"]:
                self._render_layer(layer, target, scene, scene_time, time, base_matrix)
            if target is not canvas:
                matte_bitmap = None
                if transition.get("type") == "matte" and transition.get("asset") and self.assets:
                    matte_bitmap = self.assets.get(transition["asset"])
                applied = apply_scene_transition(
                    target,
                    transition,
                    scene_time,
                    scene["duration"],
                    self.seed + scene["index"],
                    {"matte": matte_bitmap},
                )
                composed = target
                if applied.get("blur", 0) > 0:
                    composed = apply_effect(composed, {"type": "blur", "radius": applied["blur"]}, {})
                if applied.get("scale") and abs(applied["scale"] - 1) > 1e-4:
                    # シーンバッファ全体を中心基準で拡大する（ズームトランジション）
                    zoomed = Bitmap(composed.width, composed.height)
                    mesh = Mesh(1, 1, composed.width, composed.height, composed.width, composed.height)
                    matrix = Mat2D.translate(Mat2D.identity(), composed.width / 2, composed.height / 2)
                    matrix = Mat2D.scale(matrix, applied["scale"], applied["scale"])
                    matrix = Mat2D.translate(matrix, -composed.width / 2, -composed.height / 2)
                    mesh.draw(zoomed, composed, matrix, {"clampEdge": True})
                    composed = zoomed
                if applied["alpha"] > 0:
                    draw_bitmap(canvas, composed, applied["offsetX"], applied["offsetY"], applied["alpha"])
                flash = applied.get("flash")
                if flash and flash["amount"] > 0.002:
                    region = rasterize_contours(
                        [[0, 0, canvas.width, 0, canvas.width, canvas.height, 0, canvas.height]],
                        canvas.width,
                        canvas.height,
                    )
                    fill_coverage(canvas, region, flash["color"], clamp(flash["amount"], 0, 1))

        # slitScan を使うプロジェクトだけ、合成済みフレームを積んでおく
        if self._needs_frame_stack:
            self.frame_stack.insert(0, canvas.copy())
            del self.frame_stack[self.frame_stack_limit :]

        return canvas.resize(self.width, self.height) if scale > 1 else canvas

    # ── レイヤー 1 枚 ─────────────────────────────────────────

    def _render_layer(
        self,
        layer: dict,
        target: Bitmap,
        scene: dict,
        scene_time: float,
        global_time: float,
        parent_matrix,
    ) -> None:
        if layer.get("enabled") is False:
            return
        if not is_layer_active(layer, scene_time):
            return
        if layer.get("type") == "audio":
            return
        # トラックマット専用のレイヤーは、それ自体は描かない
        if layer.get("matte") is True:
            return

        # コマ落とし（ポスタリゼーション時間）: このレイヤーだけ低い fps で評価する。
        hold = layer.get("frameHold")
        if hold and hold.get("enabled") is not False and (hold.get("fps") or 0) > 0:
            step = 1 / hold["fps"]
            shifted = math.floor(scene_time / step + 1e-9) * step
            global_time += shifted - scene_time
            scene_time = shifted

        # タイムリマップ（速度ランプ・フリーズ・逆再生）。**時間を曲げるだけ**です。
        # ここで曲げておけば、キーフレーム・式・モジュレーター・動画の再生位置が
        # «まとめて» 追従します。`frameHold` の «後» に掛けるのは、先に出力側の
        # 時刻を粗くしてから曲げると「1 秒に 8 回」が遅い区間でも守られるためです。
        remapped = _remap_time(layer.get("timeRemap"), scene_time)
        if remapped is not None:
            global_time += remapped - scene_time
            scene_time = remapped

        body = self.bodies.get(layer.get("id"))
        ctx = self._context_for(layer, scene, scene_time, global_time, body)
        transform = self._resolve_transform(layer, ctx)
        modifiers = self._resolve_modifiers(layer.get("modifiers"), ctx)
        effects = self._resolve_modifiers(layer.get("effects"), ctx)

        # `animations` は transform.* / modifiers.<id>.* / effects.<id>.* / shape.* を指せる
        animation_target = {
            "transform": transform,
            "modifiers": modifiers["by_id"],
            "effects": effects["by_id"],
        }
        # `shape.*` を指す animations があるときだけ、図形の指定を先に解いて的に載せる。
        # 毎回先に解くと二度手間なので、指しているときだけ。
        animates_shape = any(
            isinstance(a, dict) and isinstance(a.get("property"), str) and a["property"].startswith("shape.")
            for a in (layer.get("animations") or [])
        )
        if animates_shape:
            animation_target["shape"] = resolve_animated(layer.get("shape") or {}, ctx, {}) or {}
        apply_animations(animation_target, layer.get("animations"), ctx)
        if animates_shape:
            # ctx はレイヤーごとに作られるので、入れ子で描いても混ざらない
            ctx["animatedShape"] = animation_target["shape"]

        self._apply_physics(layer, transform, body, ctx, scene_time)
        self._apply_motion_path(layer, transform, ctx)

        content = self._render_content(layer, ctx, transform, scene, scene_time, global_time, target)
        if not content or content["bitmap"].is_empty:
            return

        # 軌跡を記録する（linePath.followLayer から参照される）
        if layer.get("id") in self._tracked_layers:
            trail = self.motion_trails.setdefault(layer["id"], [])
            last = trail[-1] if trail else None
            if last is None or math.hypot(last["x"] - transform["x"], last["y"] - transform["y"]) > 0.5:
                trail.append({"x": transform["x"], "y": transform["y"], "frame": self.current_frame})
                if len(trail) > 600:
                    trail.pop(0)

        self.layer_states[layer["id"]] = {
            "id": layer["id"],
            "type": layer.get("type"),
            "transform": dict(transform),
            "width": content["box_width"],
            "height": content["box_height"],
            "physics": body.state() if body is not None else None,
        }

        bitmap = content["bitmap"]
        origin_x = content.get("origin_x") or 0
        origin_y = content.get("origin_y") or 0
        # `box_*` はメッシュの footprint、`anchor_*` はアンカーが基準にする論理的な
        # 大きさ（文字のように余白が付くと 2 つは違う値になる）。
        box_width = content["box_width"]
        box_height = content["box_height"]
        anchor_width = content.get("anchor_width")
        anchor_width = box_width if anchor_width is None else anchor_width
        anchor_height = content.get("anchor_height")
        anchor_height = box_height if anchor_height is None else anchor_height
        content_scale = content.get("scale") or 1

        # 領域拡張: 描画領域を外へ広げる。回転や変形で内容が切れるのを防ぐ。
        # アンカーは元の内容を基準にしたままにしたいので、origin をずらして相殺する。
        if layer.get("regionExpand"):
            expand = resolve_animated(layer["regionExpand"], ctx, {}) or {}
            grown = expand_region_with(Bitmap, bitmap, expand, content_scale)
            if grown:
                bitmap = grown["bitmap"]
                origin_x -= grown["left"]
                origin_y -= grown["top"]
                box_width += grown["left"] + grown["right"]
                box_height += grown["top"] + grown["bottom"]

        # frameBuffer はマスクをエフェクトの「後」に掛ける。先に切り抜くと、
        # ぼかしが切り口を拾って縁が光ってしまう。
        defer_mask = layer.get("type") == "frameBuffer"

        def apply_own_mask(source: Bitmap) -> Bitmap:
            if not layer.get("mask"):
                return source
            return _apply_layer_mask(
                source,
                resolve_animated(layer["mask"], ctx, None),
                {"assets": self.assets, "layerAlpha": self.layer_bitmaps.get},
            )

        if not defer_mask:
            bitmap = apply_own_mask(bitmap)

        modifier_result = apply_modifiers(
            bitmap,
            modifiers["list"],
            {
                "time": scene_time,
                "fps": self.timeline["fps"],
                "seed": self.seed,
                "assets": self.assets,
                "meshResolution": self.mesh_resolution,
                "renderScale": content_scale,
                "boxWidth": box_width,
                "boxHeight": box_height,
                "plugins": self.plugins,
                "layerAlpha": self.layer_bitmaps.get,
            },
        )
        bitmap = modifier_result["bitmap"]
        mesh = modifier_result["mesh"]
        # ここだけ **camelCase** で受けます（apply_modifiers の戻り値がその綴り）。
        origin_x -= modifier_result["offsetX"]
        origin_y -= modifier_result["offsetY"]
        box_width = modifier_result["boxWidth"]
        box_height = modifier_result["boxHeight"]

        # ソフトチェーンの物理はレイヤーを点列に沿って曲げる
        chain = self.soft_chains.get(layer.get("id"))
        if chain is not None:
            _apply_soft_chain_to_mesh(mesh, chain, transform)

        # 画素のエフェクトとエフェクトグラフは、幾何変形の後ろで走る
        if effects["list"] or layer.get("effectGraph") or transform.get("cameraBlur"):
            if not mesh.is_identity():
                baked = bake_mesh(mesh, bitmap, Bitmap, content_scale)
                bitmap = baked["bitmap"]
                origin_x -= baked["offsetX"]
                origin_y -= baked["offsetY"]
                box_width = baked["width"]
                box_height = baked["height"]
            effect_ctx = {
                "time": scene_time,
                "fps": self.timeline["fps"],
                "seed": self.seed,
                "assets": self.assets,
                "plugins": self.plugins,
                "layerAlpha": self.layer_bitmaps.get,
                # 時間方向の加工（slitScan）が使う過去フレーム。新しい順に並ぶ。
                "frameHistory": self._effect_frame_history,
            }
            for effect in effects["list"]:
                if effect.get("enabled") is False:
                    continue
                bitmap = apply_effect(bitmap, effect, effect_ctx)
            if transform.get("cameraBlur"):
                bitmap = apply_effect(bitmap, {"type": "blur", "radius": transform["cameraBlur"]}, effect_ctx)
            if layer.get("effectGraph"):
                bitmap = apply_effect_graph(
                    bitmap, resolve_animated(layer["effectGraph"], ctx, None), effect_ctx
                )
            mesh = Mesh.grid(box_width, box_height, self.mesh_resolution, bitmap.width, bitmap.height)

        if defer_mask:
            bitmap = apply_own_mask(bitmap)
            spec = resolve_animated(layer.get("frameBuffer") or {}, ctx, {}) or {}
            if spec.get("clear") and bitmap.width == target.width and bitmap.height == target.height:
                # 取り込んだぶんを元から消す。**透明な黒へ寄せます** — 書き出しに
                # アルファは残らないので、アルファだけ削っても «見た目は消えて
                # いない» ことになってしまいます。
                keep = 1 - bitmap.data[..., 3].astype(np.float32) / 255
                taken = bitmap.data[..., 3] > 0
                scaled = target.data.astype(np.float32) * keep[..., None]
                target.data[...] = np.where(taken[..., None], scaled.astype(np.uint8), target.data)

        self.layer_bitmaps[layer["id"]] = bitmap

        anchor_offset_x = -transform["anchorX"] * anchor_width - origin_x
        anchor_offset_y = -transform["anchorY"] * anchor_height - origin_y
        chain_matrix = Mat2D.multiply(
            parent_matrix, self._parent_chain_matrix(layer, scene, scene_time, global_time)
        )
        draw_options = {
            # 照明は «足す» のが既定。矩形として乗せると黒い板になってしまう。
            "blend": layer.get("blend") or ("screen" if layer.get("type") == "spotlight" else "normal"),
            "clampEdge": False,
        }

        def draw_once(destination: Bitmap, active_transform: dict, alpha: float) -> None:
            # 3 次元に傾けた板は、4 隅を投影してテクスチャ三角形として描く。
            # 2D のアフィン変換では «奥ほど狭くなる» 台形が作れないため。
            if is_plane3d(active_transform):
                plane = project_plane(
                    active_transform,
                    {
                        "left": anchor_offset_x,
                        "top": anchor_offset_y,
                        "width": box_width,
                        "height": box_height,
                    },
                    self._camera3d_for(ctx),
                )
                self._plane_depths = plane["depths"]
                if not plane["visible"]:
                    return
                draw_plane(
                    destination,
                    bitmap,
                    plane["corners"],
                    {
                        "alpha": alpha,
                        "blend": draw_options["blend"],
                        "doubleSided": layer.get("doubleSided") is not False,
                        "facing": plane["facing"],
                        "depth": self._depth_for(destination, layer),
                    },
                )
                return
            matrix = Mat2D.multiply(chain_matrix, Mat2D.from_transform(active_transform))
            matrix = Mat2D.translate(matrix, anchor_offset_x, anchor_offset_y)
            repeater = layer.get("repeater")
            if repeater and (repeater.get("count") or 0) > 1:
                self._draw_repeated(
                    destination, bitmap, mesh, matrix, repeater, alpha, draw_options,
                    anchor_offset_x, anchor_offset_y,
                )
            else:
                mesh.draw(destination, bitmap, matrix, {**draw_options, "alpha": alpha})

        # トラックマットは «マットを掛ける前のレイヤー» が要るので別バッファへ描く
        matte = layer.get("trackMatte")
        destination = Bitmap(target.width, target.height) if matte else target

        # 残像（エコー）: 過去の時刻のトランスフォームで同じ内容を重ねる。
        echo = layer.get("echo")
        if echo and (echo.get("count") or 0) > 0:
            count = int(clamp(js_round(echo["count"]), 1, 32))
            delay = echo.get("delay", 0.06) if echo.get("delay") is not None else 0.06
            step = clamp(echo.get("opacity", 0.5) if echo.get("opacity") is not None else 0.5, 0, 1)
            for i in range(count, 0, -1):
                past = scene_time - delay * i
                if past < (layer.get("localStart") or 0) - 1e-6:
                    continue
                echo_ctx = self._context_for(layer, scene, past, global_time - delay * i, body)
                echo_transform = self._resolve_transform(layer, echo_ctx)
                apply_animations(
                    {
                        "transform": echo_transform,
                        "modifiers": modifiers["by_id"],
                        "effects": effects["by_id"],
                    },
                    layer.get("animations"),
                    echo_ctx,
                )
                self._apply_physics(layer, echo_transform, body, echo_ctx, past)
                self._apply_motion_path(layer, echo_transform, echo_ctx)
                echo_transform["rotation"] += (echo.get("rotation") or 0) * i
                echo_transform["x"] += (echo.get("offsetX") or 0) * i
                echo_transform["y"] += (echo.get("offsetY") or 0) * i
                scale_step = (echo.get("scale", 1) or 1) ** i
                echo_transform["scaleX"] *= scale_step
                echo_transform["scaleY"] *= scale_step
                draw_once(destination, echo_transform, transform["opacity"] * (step**i))

        # 残像（フレーム履歴版）: 過去フレームの «描画結果そのもの» を重ねる。
        frame_echo = layer.get("frameEcho")
        use_frame_echo = (
            frame_echo and frame_echo.get("enabled") is not False and (frame_echo.get("count") or 0) > 0
        )
        if use_frame_echo:
            draw_frame_echo(self, destination, layer, frame_echo, draw_options, self.current_frame)

        motion_blur = layer.get("motionBlur") or (self.project.get("render") or {}).get("motionBlur")
        samples = 1
        if motion_blur and motion_blur.get("enabled") is not False:
            samples = max(1, js_round(motion_blur.get("samples", 6) or 6))
        if samples > 1:
            # トランスフォームだけをサブフレームで取る。安いうえに、モーション
            # ブラーが本来見せたい «動き・回転・拡大» はこれで出る。
            shutter = clamp(motion_blur.get("shutter", 0.5) if motion_blur.get("shutter") is not None else 0.5, 0, 1)
            span = shutter / self.timeline["fps"]
            for i in range(samples):
                offset = (i / (samples - 1) - 0.5) * span
                sub_ctx = self._context_for(layer, scene, scene_time + offset, global_time + offset, body)
                sub_transform = self._resolve_transform(layer, sub_ctx)
                apply_animations(
                    {
                        "transform": sub_transform,
                        "modifiers": modifiers["by_id"],
                        "effects": effects["by_id"],
                    },
                    layer.get("animations"),
                    sub_ctx,
                )
                self._apply_physics(layer, sub_transform, body, sub_ctx, scene_time)
                self._apply_motion_path(layer, sub_transform, sub_ctx)
                draw_once(destination, sub_transform, transform["opacity"] / samples)
        else:
            draw_once(destination, transform, transform["opacity"])

        if matte:
            self._apply_track_matte(destination, matte, scene, scene_time, global_time, parent_matrix)
            draw_bitmap(target, destination, 0, 0, 1, layer.get("blend") or "normal")

        # フレーム履歴に今フレームの描画結果を残す（frameEcho 用）
        if use_frame_echo:
            push_frame_history(
                self,
                layer,
                frame_echo,
                {
                    "frame": self.current_frame,
                    "bitmap": bitmap,
                    "mesh": mesh,
                    "matrix": Mat2D.translate(
                        Mat2D.multiply(chain_matrix, Mat2D.from_transform(transform)),
                        anchor_offset_x,
                        anchor_offset_y,
                    ),
                    "opacity": transform["opacity"],
                },
            )

    def _draw_repeated(
        self, target, bitmap, mesh, base_matrix, repeater, alpha, options, anchor_offset_x, anchor_offset_y
    ) -> None:
        """同じ内容を «ずらしながら» 何度も重ねる（シェイプリピーター）。"""
        count = min(200, max(1, js_round(repeater.get("count", 1) or 1)))
        offset = repeater.get("offset") or {}
        for i in range(count):
            factor = count - 1 - i if repeater.get("reverse") else i
            matrix = Mat2D.translate(base_matrix, -anchor_offset_x, -anchor_offset_y)
            matrix = Mat2D.translate(matrix, (offset.get("x") or 0) * factor, (offset.get("y") or 0) * factor)
            matrix = Mat2D.rotate(matrix, to_radians((offset.get("rotation") or 0) * factor))
            scale = (offset.get("scale", 1) or 1) ** factor
            matrix = Mat2D.scale(matrix, scale, scale)
            matrix = Mat2D.translate(matrix, anchor_offset_x, anchor_offset_y)
            step_alpha = alpha * ((offset.get("opacity", 1) or 1) ** factor)
            if step_alpha <= 0.002:
                continue
            mesh.draw(target, bitmap, matrix, {**options, "alpha": step_alpha})

    def _apply_track_matte(self, buffer, matte, scene, scene_time, global_time, parent_matrix) -> None:
        """バッファのアルファに、別レイヤーのアルファ（か輝度）を掛ける。

        マットのレイヤーは «単独で» 描き直します。そうしないと別々に動かせません。
        """
        matte_layer = find_layer(scene["layers"], matte.get("layer"))
        if matte_layer is None:
            logger.warn(f'trackMatte layer "{matte.get("layer")}" was not found; the matte is ignored')
            return
        matte_buffer = Bitmap(buffer.width, buffer.height)
        self._render_layer(
            {**matte_layer, "trackMatte": None, "matte": False},
            matte_buffer,
            scene,
            scene_time,
            global_time,
            parent_matrix,
        )
        use_luma = (matte.get("type") or "alpha") == "luma"
        data = matte_buffer.data.astype(np.float32)
        if use_luma:
            weight = (
                (0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]) / 255
            ) * (data[..., 3] / 255)
        else:
            weight = data[..., 3] / 255
        if matte.get("invert") is True:
            weight = 1 - weight
        buffer.data[..., 3] = np.clip(buffer.data[..., 3].astype(np.float32) * weight, 0, 255).astype(np.uint8)

    def _parent_chain_matrix(self, layer, scene, scene_time, global_time, depth: int = 0):
        """`layer.parent` から受け継ぐ変換（親をたどって掛け合わせる）。"""
        if not layer.get("parent") or depth > 16:
            return Mat2D.identity()
        parent = find_layer(scene["layers"], layer["parent"])
        if parent is None:
            logger.warn(f'layer "{layer["id"]}" references unknown parent "{layer["parent"]}"')
            return Mat2D.identity()
        body = self.bodies.get(parent.get("id"))
        ctx = self._context_for(parent, scene, scene_time, global_time, body)
        transform = self._resolve_transform(parent, ctx)
        apply_animations({"transform": transform}, parent.get("animations"), ctx)
        self._apply_physics(parent, transform, body, ctx, scene_time)
        self._apply_motion_path(parent, transform, ctx)
        own = Mat2D.from_transform(transform)
        return Mat2D.multiply(
            self._parent_chain_matrix(parent, scene, scene_time, global_time, depth + 1), own
        )

    def _apply_motion_path(self, layer: dict, transform: dict, ctx: dict) -> None:
        """パスに沿って動かす。`autoOrient` で進行方向へ向く。"""
        spec = (layer.get("transform") or {}).get("motionPath") or layer.get("motionPath")
        if not spec:
            return
        points = []
        for p in spec.get("points") or []:
            if isinstance(p, dict):
                points.append([p.get("x") or 0, p.get("y") or 0])
            else:
                points.append([p[0] if len(p) > 0 else 0, p[1] if len(p) > 1 else 0])
        if len(points) < 2:
            return
        path = [*points, points[0]] if spec.get("closed") else points
        default_progress = ctx["time"] / max(1e-6, self.timeline["duration"])
        progress = clamp(resolve_number(spec.get("progress"), ctx, default_progress), 0, 1)
        x, y = sample_polyline(path, progress)
        transform["x"] = x + (spec.get("offsetX") or 0)
        transform["y"] = y + (spec.get("offsetY") or 0)
        if spec.get("autoOrient"):
            delta = 0.002
            ax, ay = sample_polyline(path, clamp(progress - delta, 0, 1))
            bx, by = sample_polyline(path, clamp(progress + delta, 0, 1))
            transform["rotation"] = to_degrees(math.atan2(by - ay, bx - ax)) + (spec.get("orientOffset") or 0)

    def _apply_physics(self, layer: dict, transform: dict, body, ctx: dict, scene_time: float) -> None:
        if body is None:
            return
        control = layer.get("physicsControl") or (layer.get("physics") or {}).get("control") or {}
        mode = control.get("mode") or ("physics" if body.type == "dynamic" else "animation")
        physics_x = body.position.x
        physics_y = body.position.y
        physics_rotation = to_degrees(body.angle)
        if mode == "animation":
            return
        if mode == "blend":
            weight = control.get("physicsWeight")
            if weight is None:
                weight = 1 - (control.get("animationWeight", 0.5) or 0.5)
            weight = clamp(weight, 0, 1)
            transform["x"] = lerp(transform["x"], physics_x, weight)
            transform["y"] = lerp(transform["y"], physics_y, weight)
            transform["rotation"] = lerp(transform["rotation"], physics_rotation, weight)
            return
        if mode == "override":
            if within_range(control, scene_time):
                return
            transform["x"] = physics_x
            transform["y"] = physics_y
            transform["rotation"] = physics_rotation
            return
        # "follow" / "physics" / それ以外
        transform["x"] = physics_x
        transform["y"] = physics_y
        # `body.fixedRotation` は Python の Body から読めないので、
        # 作ったときに覚えた集合を見ます（simulation.py の冒頭を参照）。
        if layer.get("id") not in self._fixed_rotation:
            transform["rotation"] = physics_rotation

    # ── レイヤーの «中身» ─────────────────────────────────────

    def _render_content(
        self, layer: dict, ctx: dict, transform: dict, scene: dict, scene_time: float,
        global_time: float, target: Bitmap,
    ):
        scale = self.render_scale
        kind = layer.get("type")

        if kind == "image":
            bitmap = self.assets.get(layer.get("asset")) if self.assets else None
            if bitmap is None:
                logger.verbose(f'layer "{layer.get("id")}" has no image asset; skipped')
                return None
            # ⚠ **書かなかった側には «素材の原寸» が入ります。比率は補われません。**
            # `{"height": 400}` とだけ書いても幅は素材のまま（960px など）になるので、
            # 縦横比の違う素材を渡すと潰れます。比率を保ちたいときは
            # `transform.scale` を使い、width / height は書かないでください。
            return {
                "bitmap": bitmap,
                "box_width": transform["width"] if transform["width"] is not None else bitmap.width,
                "box_height": transform["height"] if transform["height"] is not None else bitmap.height,
                "origin_x": 0,
                "origin_y": 0,
                "scale": 1,
            }

        if kind == "video":
            meta = self.assets.describe(layer.get("asset")) if self.assets else None
            if not meta or not meta.get("source"):
                logger.verbose(f'layer "{layer.get("id")}" has no video asset; skipped')
                return None
            source = self.videos.get(layer.get("asset"))
            if source is None:
                from movo.renderer.video import VideoSource

                source = VideoSource(meta["source"], cache=self.cache, fps=self.timeline["fps"])
                self.videos[layer["asset"]] = source
            local_time = (scene_time - (layer.get("localStart") or 0)) * (
                layer.get("timeScale", 1) if layer.get("timeScale") is not None else 1
            ) + (layer.get("offset") or 0)
            frame = source.frame_at(local_time)
            if frame is None:
                return None
            return {
                "bitmap": frame,
                "box_width": transform["width"] if transform["width"] is not None else frame.width,
                "box_height": transform["height"] if transform["height"] is not None else frame.height,
                "origin_x": 0,
                "origin_y": 0,
                "scale": 1,
            }

        if kind == "text":
            return self._render_text_content(layer, ctx, scene_time, scale)

        if kind == "shape":
            shape_spec = ctx.get("animatedShape")
            if shape_spec is None:
                shape_spec = resolve_animated(layer.get("shape") or {}, ctx, {}) or {}
            else:
                shape_spec = dict(shape_spec)
            # レイヤー直下の `trim` は `shape.trim` の別名。「線が描かれていく」演出は
            # レイヤーの性質として書きたくなるので、両方の書き方を許す。
            if layer.get("trim") and shape_spec.get("trim") is None:
                shape_spec["trim"] = resolve_animated(layer["trim"], ctx, {})
            rendered = render_shape(shape_spec, scale, {"assets": self.assets})
            return {
                "bitmap": rendered["bitmap"],
                "box_width": rendered["bitmap"].width / scale,
                "box_height": rendered["bitmap"].height / scale,
                "anchor_width": rendered["box_width"],
                "anchor_height": rendered["box_height"],
                "origin_x": rendered["origin_x"],
                "origin_y": rendered["origin_y"],
                "scale": scale,
            }

        if kind == "group":
            buffer = Bitmap(self.width * scale, self.height * scale)
            base = Mat2D.scale(Mat2D.identity(), scale, scale)
            for child in layer.get("children") or []:
                self._render_layer(child, buffer, scene, scene_time, global_time, base)
            return {
                "bitmap": buffer,
                "box_width": self.width,
                "box_height": self.height,
                "origin_x": 0,
                "origin_y": 0,
                "scale": scale,
            }

        if kind == "composition":
            return self._render_composition(layer, scene, scene_time, global_time, scale)

        # ── 組み込み／生成レイヤー ───────────────────────────
        builtin = _builtin_layers()
        if kind == "character":
            return builtin["render_character"](self, layer, ctx, scene, scene_time, global_time)
        if kind == "particle":
            return builtin["render_particle_layer"](self, layer, ctx, transform)
        if kind == "shader":
            return builtin["render_shader"](self, layer, ctx, transform, scene_time)
        if kind == "fractalNoise":
            return builtin["render_fractal_noise"](self, layer, ctx, transform, scene_time)
        if kind == "frameBuffer":
            return builtin["render_frame_buffer"](self, layer, ctx, target)
        if kind == "waveform":
            return builtin["render_waveform"](self, layer, ctx, transform, global_time)
        if kind == "speedLines":
            return builtin["render_speed_lines"](self, layer, ctx, transform, scene_time)
        if kind in (
            "starfield", "linePath", "neonPath", "metaball", "waterSurface",
            "primitive3d", "spotlight", "mesh", "shapeAnim",
        ):
            return builtin["render_generator"](self, layer, ctx, transform, scene_time, target)

        if kind == "custom":
            handler = self._plugin("layerType", layer.get("layerType") or layer.get("kind") or "custom")
            if handler is None:
                logger.warn(
                    f'no plugin provides the custom layer type '
                    f'"{layer.get("layerType") or layer.get("kind")}"; skipped'
                )
                return None
            result = handler(
                {"layer": layer, "ctx": ctx, "renderer": self, "Bitmap": Bitmap,
                 "transform": transform, "time": scene_time}
            )
            if not result or not result.get("bitmap"):
                return None
            return {
                "bitmap": result["bitmap"],
                "box_width": result.get("box_width", result.get("boxWidth")) or result["bitmap"].width,
                "box_height": result.get("box_height", result.get("boxHeight")) or result["bitmap"].height,
                "origin_x": result.get("origin_x", result.get("originX")) or 0,
                "origin_y": result.get("origin_y", result.get("originY")) or 0,
                "scale": result.get("scale") or 1,
            }

        logger.warn(f'unknown layer type "{kind}"; skipped')
        return None

    def _render_text_content(self, layer: dict, ctx: dict, scene_time: float, scale: int):
        resolved_text = resolve_animated(
            {"text": layer.get("text"), "style": layer.get("style"), "font": layer.get("font")}, ctx, None
        )
        styled = resolve_text_style(layer, resolved_text or {})
        content = styled["content"]
        style = styled["style"]

        # カウンター（数字が動く演出）
        if layer.get("counter"):
            counter = resolve_animated(layer["counter"], ctx, {}) or {}
            layer_length = max(1e-6, (layer.get("localEnd") or 1) - (layer.get("localStart") or 0))
            raw = counter.get("progress")
            if raw is None:
                raw = (scene_time - (layer.get("localStart") or 0)) / layer_length
            eased = get_easing(counter.get("easing") or "linear")(clamp(raw, 0, 1))
            content = format_counter(counter, eased)

        # `fit.maxWidth: "80%"` の基準。組版側からは画面の寸法が見えないので、
        # ここで入れておきます。入れないと «% を書いても黙って効かない» という
        # いちばん困る形になります。
        if style.get("fit") and style["fit"].get("basis") is None:
            basis = style.get("maxWidth")
            if basis is None:
                basis = style.get("width")
            if basis is None:
                basis = (layer.get("transform") or {}).get("width")
            if basis is None:
                basis = self.width
            style["fit"] = {**style["fit"], "basis": basis}

        stroke = style.get("stroke")
        shadow = style.get("shadow")
        scaled_style = {
            **style,
            "size": (style.get("size") or 48) * scale,
            "letterSpacing": (style.get("letterSpacing") or 0) * scale,
            "maxWidth": style["maxWidth"] * scale if style.get("maxWidth") else None,
            # ドットの網目も «論理画素» で書きたい。倍率を掛けないと、
            # 高品質（superSample > 1）で書き出したときだけ網が細かくなる。
            "pixelGrid": style["pixelGrid"] * scale if style.get("pixelGrid") else None,
            "stroke": {**stroke, "width": (stroke.get("width") or 0) * scale} if stroke else None,
            "shadow": (
                {
                    **shadow,
                    "blur": (shadow.get("blur") or 0) * scale,
                    "offsetX": (shadow.get("offsetX") or 0) * scale,
                    "offsetY": (shadow.get("offsetY") or 0) * scale,
                }
                if shadow
                else None
            ),
        }
        # ランダムフォントと書き順は «時刻で変わる» ので、レイヤー相対の時刻を渡す
        local_time = scene_time - (layer.get("localStart") or 0)
        if scaled_style.get("randomFont"):
            scaled_style["time"] = local_time
        if layer.get("strokeOrder"):
            scaled_style["strokeOrder"] = resolve_animated(layer["strokeOrder"], ctx, {}) or {}
            scaled_style["time"] = local_time

        # パス上の文字（円周・弧・折れ線）
        if layer.get("textPath"):
            path_spec = resolve_animated(layer["textPath"], ctx, {}) or {}
            scaled_path = {
                **path_spec,
                "radius": (path_spec.get("radius", 200) if path_spec.get("radius") is not None else 200) * scale,
                "firstMargin": (path_spec.get("firstMargin") or 0) * scale,
                "points": [
                    [p[0] * scale, p[1] * scale] if isinstance(p, (list, tuple)) else p
                    for p in (path_spec.get("points") or [])
                ]
                or None,
            }
            on_path = render_text_on_path(content, scaled_style, self.font_manager, scaled_path)
            if on_path:
                return {
                    "bitmap": on_path["bitmap"],
                    "box_width": on_path["bitmap"].width / scale,
                    "box_height": on_path["bitmap"].height / scale,
                    "anchor_width": on_path["bitmap"].width / scale,
                    "anchor_height": on_path["bitmap"].height / scale,
                    "origin_x": 0,
                    "origin_y": 0,
                    "scale": scale,
                }

        animator = layer.get("textAnimator")
        if animator is None and isinstance(resolved_text, dict):
            animator = resolved_text.get("textAnimator")
        if animator:
            rendered = render_animated_text(
                content, scaled_style, self.font_manager, _scale_animator(animator, scale), local_time
            )
        else:
            rendered = render_text(content, scaled_style, self.font_manager)

        # カラオケ風の歌詞塗り
        if layer.get("karaoke"):
            karaoke = resolve_animated(layer["karaoke"], ctx, {}) or {}
            rendered["bitmap"] = apply_karaoke_fill(
                rendered["bitmap"],
                karaoke,
                {
                    "offset_x": rendered["offset_x"],
                    "width": max(1, rendered["layout"]["width"]),
                    "base_color": style.get("color"),
                    # 色を変えたラン（`text.runs`）も塗り替えの対象にします。
                    # 地の色 1 つだけを渡すと、**強調した語だけ塗り残されます。**
                    "base_colors": None if karaoke.get("keepRunColors") is True else _run_colors_of(style),
                },
            )

        # 文字に追従する枠。文字数が変わっても手で調整しなくてよくなる。
        if layer.get("textBox"):
            box_spec = resolve_animated(layer["textBox"], ctx, {}) or {}
            box_shadow = box_spec.get("shadow")
            box_stroke = box_spec.get("stroke")
            scaled_box = {
                **box_spec,
                "padding": [v * scale for v in resolve_padding(box_spec.get("padding", 16))],
                "radius": (box_spec.get("radius") or 0) * scale,
                "stroke": (
                    {**box_stroke, "width": (box_stroke.get("width") or 0) * scale} if box_stroke else None
                ),
                "shadow": (
                    {
                        **box_shadow,
                        "blur": (box_shadow.get("blur") or 0) * scale,
                        "offsetX": (box_shadow.get("offsetX") or 0) * scale,
                        "offsetY": (box_shadow.get("offsetY") or 0) * scale,
                    }
                    if box_shadow
                    else None
                ),
            }
            boxed = draw_text_box(rendered["bitmap"], rendered, scaled_box)
            return {
                "bitmap": boxed["bitmap"],
                "box_width": boxed["bitmap"].width / scale,
                "box_height": boxed["bitmap"].height / scale,
                # アンカーは «枠» を基準にする
                "anchor_width": boxed["box_width"] / scale,
                "anchor_height": boxed["box_height"] / scale,
                "origin_x": boxed["offset_x"] / scale,
                "origin_y": boxed["offset_y"] / scale,
                "scale": scale,
            }

        return {
            "bitmap": rendered["bitmap"],
            # メッシュは «余白込みのビットマップ» を論理画素で 1:1 に覆う…
            "box_width": rendered["bitmap"].width / scale,
            "box_height": rendered["bitmap"].height / scale,
            # …一方アンカーは «文字の塊» を指す
            "anchor_width": rendered["layout"]["width"] / scale,
            "anchor_height": rendered["layout"]["height"] / scale,
            "origin_x": rendered["offset_x"] / scale,
            "origin_y": rendered["offset_y"] / scale,
            "scale": scale,
        }

    def _render_composition(self, layer: dict, scene: dict, scene_time: float, global_time: float, scale: int):
        composition = (self.project.get("compositions") or {}).get(layer.get("composition"))
        if not composition:
            logger.warn(
                f'composition "{layer.get("composition")}" is not declared; '
                f'layer "{layer.get("id")}" is skipped'
            )
            return None
        comp_width = composition.get("width") or self.width
        comp_height = composition.get("height") or self.height
        buffer = Bitmap(js_round(comp_width * scale), js_round(comp_height * scale))
        if composition.get("background"):
            bg = parse_color(composition["background"])
            if bg[3] > 0:
                buffer.fill((int(bg[0]), int(bg[1]), int(bg[2]), _u8(bg[3] * 255)))
        comp_duration = composition.get("duration") or self.timeline["duration"]
        comp_time = scene_time - (layer.get("localStart") or 0) + (layer.get("offset") or 0)
        if layer.get("loop") and comp_duration > 0:
            comp_time = math.fmod(math.fmod(comp_time, comp_duration) + comp_duration, comp_duration)
        comp_scene = {
            "id": f'{scene["id"]}/{layer.get("composition")}',
            "index": scene["index"],
            "start": 0,
            "duration": comp_duration,
            "end": comp_duration,
            "layers": [],
        }
        children = layer.get("children")
        if not children:
            children = _prepare_composition_layers(composition, comp_duration)
            layer["children"] = children
        base = Mat2D.scale(Mat2D.identity(), scale, scale)
        for child in children:
            self._render_layer(child, buffer, comp_scene, comp_time, global_time, base)
        return {
            "bitmap": buffer,
            "box_width": comp_width,
            "box_height": comp_height,
            "origin_x": 0,
            "origin_y": 0,
            "scale": scale,
        }

    def describe(self) -> dict:
        return {
            "renderer": self.kind,
            "superSample": self.render_scale,
            "meshResolution": self.mesh_resolution,
            "bodies": len(self.bodies),
            "softChains": len(self.soft_chains),
            "particleSystems": len(self.particles),
            "frameHistoryLimit": self.frame_history_limit,
            "frameHistoryBytes": self.history_bytes,
            "fonts": self.font_manager.describe() if hasattr(self.font_manager, "describe") else None,
        }


# ══════════════════════════════════════════════════════════════════
# 補助
# ══════════════════════════════════════════════════════════════════


class _CallableScope(dict):
    """式から `layer.x` とも `layer("id").x` とも書けるようにする入れ物。

    JS 版は関数オブジェクトにプロパティを貼っています。Python の式エンジンは
    «呼び出し» と «属性/添字» の両方を扱うので、辞書を継いだうえで
    ``__call__`` を足しました。
    """

    __slots__ = ("_lookup",)

    def __init__(self, lookup, current: dict) -> None:
        super().__init__(current or {})
        self._lookup = lookup

    def __call__(self, layer_id: str):
        return self._lookup(layer_id)


def _u8(value: float) -> int:
    return int(min(255, max(0, round(value))))


def _builtin_layers() -> dict:
    """組み込み／生成レイヤーの入口を遅延で引く。

    ``layers_builtin`` は ``index`` を **import しません**（renderer は第 1 引数で
    渡ります）が、こちらから先に読むと «移植が終わっていないときに
    レンダラー全体が読めない» ことになります。使うときだけ引きます。
    """
    cache = _builtin_layers._cache
    if cache is not None:
        return cache

    def missing(name):
        def _skip(*_args, **_kwargs):
            logger.warn(f"{name} レイヤーはまだ移植されていません — 飛ばします（後で繋ぐ）")
            return None

        return _skip

    entries = {}
    try:
        from movo.renderer import layers_builtin as module

        for name in (
            "render_character", "render_particle_layer", "render_shader",
            "render_fractal_noise", "render_frame_buffer", "render_waveform",
            "render_speed_lines", "render_generator",
        ):
            entries[name] = getattr(module, name, None) or missing(name)
    except Exception as error:  # noqa: BLE001 - 未移植でも他のレイヤーは描けるようにする
        logger.verbose(f"layers_builtin を読めませんでした: {error}")
        for name in (
            "render_character", "render_particle_layer", "render_shader",
            "render_fractal_noise", "render_frame_buffer", "render_waveform",
            "render_speed_lines", "render_generator",
        ):
            entries[name] = missing(name)
    _builtin_layers._cache = entries
    return entries


_builtin_layers._cache = None


def _scale_animator(animator: dict, scale: int) -> dict:
    """テキストアニメーターの «px の値» は論理画素なので、SSAA 用に倍にする。"""
    if scale == 1:
        return animator
    source = animator.get("from") or {}
    return {
        **animator,
        "from": {**source, "x": (source.get("x") or 0) * scale, "y": (source.get("y") or 0) * scale},
    }


def _prepare_composition_layers(composition: dict, duration: float) -> list[dict]:
    """合成の中のレイヤーに «時刻と順序» を付ける（タイムラインの簡易版）。"""
    layers = composition.get("layers") or []
    out = []
    for index, layer in enumerate(layers):
        entry = dict(layer)
        entry["id"] = layer.get("id") or f'{composition.get("id") or "comp"}-{layer.get("type")}-{index}'
        entry["order"] = index
        entry["zIndex"] = index if layer.get("zIndex") is None else layer["zIndex"]
        entry["localStart"] = layer.get("start") or 0
        end = layer.get("end")
        if end is None:
            end = (layer.get("start") or 0) + layer["duration"] if layer.get("duration") is not None else duration
        entry["localEnd"] = end
        entry["children"] = (
            _prepare_composition_layers({"layers": layer["layers"], "id": layer.get("id")}, duration)
            if layer.get("layers")
            else None
        )
        out.append(entry)
    return out


def apply_scene_transition(
    buffer: Bitmap, transition: dict, scene_time: float, duration: float, seed: int = 0,
    options: dict | None = None,
) -> dict:
    """シーンのトランジション（仕様 6 章 / scene.transition）。

    シーンバッファを合成するときの «不透明度とずれ» を返します。ワイプ・虹彩・
    ディゾルブはバッファのアルファを直接書き換えます。

    **画素ごとのループは全部 NumPy にしてあります。** JS 版は二重 for ですが、
    1920x1080 の 1 パスを Python で回すと 1 フレームあたり数秒かかります。
    """
    options = options or {}
    in_length = transition.get("in")
    if in_length is None:
        in_length = transition.get("duration", 0.5) if transition.get("duration") is not None else 0.5
    out_length = transition.get("out")
    if out_length is None:
        out_length = transition.get("duration", 0.5) if transition.get("duration") is not None else 0.5
    kind = transition.get("type") or "fade"
    easing = get_easing(transition.get("easing")) if transition.get("easing") else (lambda t: t)
    progress_in = clamp(scene_time / in_length, 0, 1) if in_length > 0 else 1
    progress_out = clamp((duration - scene_time) / out_length, 0, 1) if out_length > 0 else 1
    entering = progress_in < 1
    progress = easing(min(progress_in, progress_out))
    if progress >= 1:
        return {"alpha": 1, "offsetX": 0, "offsetY": 0}

    direction = transition.get("direction") or "left"
    height, width = buffer.data.shape[0], buffer.data.shape[1]
    alpha = buffer.data[..., 3]

    if kind == "matte":
        # グレースケール画像 1 枚で «任意のワイプ» を作る。マット画像を差し替える
        # だけで、インク染み・ブラインド・時計ワイプ・砕け・放射が全部作れます。
        # **暗い画素から先に抜けます**（紙が黒から燃える、が直感に合うため）。
        softness = clamp(transition.get("softness", 0.12) if transition.get("softness") is not None else 0.12, 0.001, 1)
        matte = options.get("matte")
        edge = progress * (1 + softness)
        if matte is not None:
            # マット画像は解像度が違って構いません。比率で引きます。
            mx = np.minimum(matte.width - 1, (np.arange(width) / width * matte.width).astype(np.int64))
            my = np.minimum(matte.height - 1, (np.arange(height) / height * matte.height).astype(np.int64))
            sample = matte.data[my[:, None], mx[None, :]].astype(np.float64)
            level = (sample[..., 0] * 0.2126 + sample[..., 1] * 0.7152 + sample[..., 2] * 0.0722) / 255
        else:
            # 画像を用意しないときは生成マット。値ノイズなので «砕けて抜ける» 見た目。
            noise_scale = (transition.get("generator") or {}).get("scale") or 40
            ys, xs = np.mgrid[0:height, 0:width]
            level = np.clip((value_noise_2d(xs / noise_scale, ys / noise_scale, seed) + 1) / 2, 0, 1)
        if transition.get("invert") is True:
            level = 1 - level
        weight = np.clip((edge - level) / softness, 0, 1)
        _multiply_alpha(alpha, weight)
        return {"alpha": 1, "offsetX": 0, "offsetY": 0}

    if kind == "slide":
        distance = 1 - progress
        sign = 1 if entering else -1
        dx = 0
        dy = 0
        if direction == "left":
            dx = -width * distance * sign
        elif direction == "right":
            dx = width * distance * sign
        elif direction == "up":
            dy = -height * distance * sign
        elif direction == "down":
            dy = height * distance * sign
        return {"alpha": 1, "offsetX": dx, "offsetY": dy}

    if kind == "wipe":
        softness = max(1, (transition.get("softness", 0.05) if transition.get("softness") is not None else 0.05) * max(width, height))
        horizontal = direction in ("left", "right")
        extent = width if horizontal else height
        edge = progress * (extent + softness)
        if horizontal:
            position = np.arange(width) if direction == "left" else (width - 1 - np.arange(width))
            weight = np.clip((edge - position) / softness, 0, 1)[None, :]
        else:
            position = np.arange(height) if direction == "up" else (height - 1 - np.arange(height))
            weight = np.clip((edge - position) / softness, 0, 1)[:, None]
        _multiply_alpha(alpha, np.broadcast_to(weight, (height, width)))
        return {"alpha": 1, "offsetX": 0, "offsetY": 0}

    if kind == "jaws":
        # ギザギザの歯が上下（左右）から噛み合って画面を閉じる。
        teeth = max(1, js_round(transition.get("teeth", 12) or 12))
        depth = clamp(transition.get("depth", 0.4) if transition.get("depth") is not None else 0.4, 0, 1)
        shape = transition.get("shape") or "spikes"
        horizontal = (transition.get("direction") or "vertical") == "horizontal"
        span = width if horizontal else height
        across = height if horizontal else width
        close = 1 - progress
        along = np.arange(across, dtype=np.float64)
        t = along / max(1, across - 1) * teeth
        phase = t - np.floor(t)
        if shape == "waves":
            tooth = 0.5 - 0.5 * np.cos(phase * TAU)
        elif shape == "blocks":
            tooth = np.where(phase < 0.5, 0.0, 1.0)
        else:
            tooth = 1 - np.abs(phase * 2 - 1)
        bite = close * span * (0.5 + depth * (tooth - 0.5))
        across_pos = np.arange(span, dtype=np.float64)
        # visible[along, across_position]
        visible = (across_pos[None, :] > bite[:, None]) & (across_pos[None, :] < span - bite[:, None])
        mask = visible if horizontal else visible.T
        alpha[...] = np.where(mask, alpha, 0)
        return {"alpha": 1, "offsetX": 0, "offsetY": 0}

    if kind == "iris":
        cx = (transition.get("centerX", 0.5) if transition.get("centerX") is not None else 0.5) * width
        cy = (transition.get("centerY", 0.5) if transition.get("centerY") is not None else 0.5) * height
        max_radius = math.hypot(max(cx, width - cx), max(cy, height - cy))
        radius = progress * max_radius
        softness = max(1, (transition.get("softness", 0.08) if transition.get("softness") is not None else 0.08) * max_radius)
        ys, xs = np.mgrid[0:height, 0:width]
        distance = np.hypot(xs - cx, ys - cy)
        _multiply_alpha(alpha, np.clip((radius - distance) / softness, 0, 1))
        return {"alpha": 1, "offsetX": 0, "offsetY": 0}

    if kind == "dissolve":
        ys, xs = np.mgrid[0:height, 0:width]
        n = (value_noise_2d(xs * 0.35, ys * 0.35, seed) + 1) / 2
        alpha[...] = np.where(n <= progress, alpha, 0)
        return {"alpha": 1, "offsetX": 0, "offsetY": 0}

    if kind == "zoom":
        # 入りは大きく、抜けはさらに大きくしながらフェードする。
        target = transition.get("scale", 1.5) if transition.get("scale") is not None else 1.5
        factor = lerp(target, 1, progress) if entering else lerp(1, target, 1 - progress)
        return {
            "alpha": progress,
            "offsetX": 0,
            "offsetY": 0,
            "scale": factor,
            "blur": (transition.get("blur") or 0) * (1 - progress),
        }

    if kind == "flash":
        # 白（指定色）で飛ばしてから戻る。
        return {
            "alpha": 1,
            "offsetX": 0,
            "offsetY": 0,
            "flash": {"color": transition.get("color") or "#ffffff", "amount": 1 - progress},
        }

    # "fade" とそれ以外
    return {"alpha": progress, "offsetX": 0, "offsetY": 0}


def _multiply_alpha(alpha: np.ndarray, weight: np.ndarray) -> None:
    """アルファ面に重みを掛ける（重み 1 の画素は触らない）。

    JS 版は `weight >= 1` を `continue` で飛ばしています。同じ結果になるよう
    NumPy でも «掛けたあと 1 のところだけ元に戻す» ではなく最初から除きます。
    こうしておくと丸め誤差で 1 画素ずれることがありません。
    """
    keep = weight >= 1
    scaled = np.clip(alpha.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    alpha[...] = np.where(keep, alpha, scaled)


def _apply_layer_mask(bitmap: Bitmap, mask, ctx: dict) -> Bitmap:
    """レイヤー単位のマスクでアルファを掛ける。

    マスク場は 128x128 で作って引き伸ばします（JS 版と同じ）。
    引くところは **NumPy の一括サンプリング**です。
    """
    if not mask:
        return bitmap
    resolution = 128
    field = build_mask_field(mask, resolution, resolution, {**ctx, "selfBitmap": bitmap})
    if field is None:
        return bitmap
    out = bitmap.copy()
    v = (np.arange(bitmap.height, dtype=np.float64) + 0.5) / bitmap.height
    u = (np.arange(bitmap.width, dtype=np.float64) + 0.5) / bitmap.width
    grid_u, grid_v = np.meshgrid(u, v)
    weight = np.clip(sample_field(field, resolution, resolution, grid_u, grid_v), 0, 1)
    out.data[..., 3] = np.clip(bitmap.data[..., 3].astype(np.float32) * weight, 0, 255).astype(np.uint8)
    return out


def _apply_soft_chain_to_mesh(mesh: Mesh, chain, transform: dict) -> None:
    """メッシュをソフトチェーンに沿って曲げる。

    レイヤーの縦軸が紐に沿い、横軸は紐に垂直なままになります。
    Python 版の ``chain.points`` は ``(N, 5)`` の NumPy 配列なので、
    頂点ごとの for ではなく **一括の索引引き** で書いています。
    """
    points = chain.points
    if points.shape[0] < 2:
        return
    origin_x, origin_y = chain.origin[0], chain.origin[1]
    count = points.shape[0]
    along = np.clip(mesh.v0, 0, 1) * (count - 1)
    index = np.minimum(count - 2, np.floor(along).astype(np.int64))
    t = along - index
    ax = points[index, 0]
    ay = points[index, 1]
    bx = points[index + 1, 0]
    by = points[index + 1, 1]
    px = ax + (bx - ax) * t
    py = ay + (by - ay) * t
    tx = bx - ax
    ty = by - ay
    length = np.hypot(tx, ty)
    length[length == 0] = 1
    tx = tx / length
    ty = ty / length
    offset = (mesh.u0 - 0.5) * mesh.width
    mesh.x[...] = px - origin_x + (-ty) * offset + transform["anchorX"] * mesh.width
    mesh.y[...] = py - origin_y + tx * offset


def _section_scope(scene: dict, global_time: float) -> dict:
    """式から見える «曲の区間» を作る。

    `from: { section }` を書いたシーンには `_section` が貼ってあります。書いて
    いないシーンでは、シーン自身を 1 区間とみなします。そうしておけば
    `section.progress` はどのシーンでも書けて、**曲を渡していない JSON でも
    式が壊れません**（未定義を読ませると式ごと落ちます）。
    """
    marked = scene.get("_section")
    start = marked["start"] if marked else (scene.get("start") or 0)
    end = marked["end"] if marked else (scene.get("start") or 0) + (scene.get("duration") or 0)
    span = max(1e-6, end - start)
    elapsed = global_time - start
    return {
        "label": (marked or {}).get("label") or scene.get("id") or "",
        "energy": (marked or {}).get("energy") or 0,
        "start": start,
        "end": end,
        "duration": end - start,
        "progress": min(1, max(0, elapsed / span)),
        "remaining": max(0, end - global_time),
    }


def _remap_time(spec, time: float):
    """タイムリマップ。出力の時刻から «素材の時刻» を引く。

    `keyframes` は「出力の何秒で、素材の何秒を見せるか」の対応表です。同じ値を
    2 つ並べればフリーズフレーム、値を下げれば逆再生になります。曲げないときは
    ``None`` を返します。
    """
    if not spec or spec.get("enabled") is False:
        return None
    keyframes = spec.get("keyframes")
    if not isinstance(keyframes, (list, tuple)) or not keyframes:
        return None
    value = sample_keyframes(keyframes, time, extrapolate=spec.get("extrapolate") or "hold")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _run_colors_of(style: dict):
    """テキストのランに出てくる色をぜんぶ集める。

    カラオケ塗りは «地の色に近い画素» を塗り替えます。色を変えたランがあると、
    その語だけ地の色と違うので塗り残されます。塗る対象の色を全部渡すためのもの。
    """
    colors = [style.get("color")]
    for run in style.get("runs") or []:
        if run and run.get("color") and run["color"] not in colors:
            colors.append(run["color"])
    return colors if len(colors) > 1 else None


__all__ = ["DEFAULT_TRANSFORM", "RENDERER_KINDS", "Renderer", "apply_scene_transition"]
