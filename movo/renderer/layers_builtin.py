"""組み込みレイヤー種別の «中身を作る» 処理。

キャラクター・パーティクル・波形・集中線・フラクタルノイズ・シェーダー・
フレームバッファ、そして生成レイヤーへの受け渡しをここにまとめています。

どれも ``{"bitmap", "box_width", "box_height", "origin_x", "origin_y", "scale"}``
（描かないときは ``None``）を返し、あとは Renderer が通常のレイヤーと同じ経路
（変形 → エフェクト → マスク → 合成）に載せます。

**状態は Renderer が持ったままなので、第 1 引数で受け取っています。**
ここから ``movo.renderer.index`` を import すると循環するので、絶対にしません。

## 移植にあたって変えたところ（JS 版との差）

* JS は ``fractalNoise`` と ``shader`` を «画素ごとの二重 for» で舐めていました。
  Python で同じことをすると 1280x720 の 1 パスで 720 ミリ秒かかるので、
  ノイズは :func:`movo.core.rng.fbm2d_grid`（Numba）に、色の割り当ては
  NumPy の一括演算に置き換えています。
* 集中線の «背骨» の分割も NumPy の一括演算です。乱数の呼ぶ順だけは
  JS と 1 回もずらしていないので、同じ種からは同じ絵が出ます。
* パーティクルは Python 側が «粒ごとのオブジェクト» ではなく
  «NumPy の配列の束» なので、描画は :func:`movo.renderer.particles.render_particles`
  に丸ごと任せ、ここは «レイヤー種別の入口» だけを持ちます。
"""

from __future__ import annotations

import math

import numpy as np

from movo.animation.resolver import resolve_animated
from movo.cli.console import logger
from movo.core.bitmap import Bitmap
from movo.core.math import TAU, clamp, js_round
from movo.core.rng import fbm2d_grid
from movo.expression import evaluate as _evaluate_ast
from movo.expression import to_number as _to_number
from movo.renderer.effects import Random
from movo.renderer.helpers import hash_code, mix_css, rect_contour_flat
from movo.renderer.particles import render_particles as _draw_particles
from movo.renderer.raster import (
    fill_coverage,
    fill_coverage_with,
    parse_color,
    rasterize_contours,
)

_U32 = 0xFFFFFFFF


# ── 小さな道具 ──────────────────────────────────────────────────

def _opt(spec: dict, key: str, fallback):
    """JS の ``spec.key ?? fallback``。

    **Python の ``or`` では代用できません。** ``0`` や ``""`` も «偽» として
    落ちてしまい、``gap: 0`` や ``curve: 0`` が既定値に化けます。
    """
    if not isinstance(spec, dict):
        return fallback
    value = spec.get(key)
    return fallback if value is None else value


def _spec_of(source, ctx) -> dict:
    """``resolveAnimated(layer.foo ?? {}, ctx, {}) ?? {}`` と同じ。

    キーフレームやモジュレータを «その時刻の値» に潰した辞書を返します。
    JSON 由来なのでキーは camelCase のままです。
    """
    resolved = resolve_animated(source if source is not None else {}, ctx, {})
    return resolved if isinstance(resolved, dict) else {}


# 色を混ぜる ``mix_css``・角丸の輪郭 ``rect_contour_flat``・レイヤー ID の
# ハッシュ ``hash_code`` は :mod:`movo.renderer.helpers` に移植済みなので、
# ここでは持ちません（JS 版も helpers.js から取っていました）。
# ``rect_contour_flat`` は ``raster.rect_contour`` とは別物です。あちらは角を
# **8 分割**、こちらは JS 版と同じ **6 分割**で、波形のバーが何十個も並ぶと
# «丸みの見え方» がはっきり変わるため、使い分ける必要があります。


def _lerp_colors(t: np.ndarray, start, end) -> np.ndarray:
    """``t``（``(h, w)`` の 0..1）を 2 色の間に写して RGBA の uint8 にする。

    **ここが «画素ごとの for» を潰した本体です。** 丸めは ``np.rint``（偶数丸め）
    で、JS の ``Uint8ClampedArray`` への代入と同じになります。
    """
    sr, sg, sb, sa = start
    er, eg, eb, ea = end
    out = np.empty(t.shape + (4,), np.uint8)
    for index, (a, b) in enumerate(((sr, er), (sg, eg), (sb, eb))):
        out[..., index] = np.rint(np.clip(a + (b - a) * t, 0, 255)).astype(np.uint8)
    out[..., 3] = np.rint(np.clip((sa + (ea - sa) * t) * 255.0, 0, 255)).astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════
# キャラクター（rig）
# ══════════════════════════════════════════════════════════════════

def render_character(renderer, layer, ctx, scene, scene_time, global_time):
    """キャラクター（rig）レイヤー。

    ⚠ **rig の «置き場所» が JS 版と違います。**
    JS は ``packages/character/src/rig.js`` ですが、Python 版に
    ``movo/character/`` はありません。``build_rig`` / ``resolve_rig_pose`` /
    ``normalize_motion`` は **``movo.physics`` に移植済み** です
    （``movo/physics/rig.py``）。«character パッケージが無い ＝ rig が未移植»
    と読んで丸ごと飛ばしていたので、繋ぎ直しました。移植先のパッケージ名が
    変わったときは、まず ``__all__`` を grep してください。

    リグは «根元が画面の中央» のバッファへ描き、アンカーはその原点を指します
    （``anchor_width`` / ``anchor_height`` を 0 にして ``origin_*`` を中央に置く）。
    こうしておくと ``transform.x/y`` が «リグの根元をどこに置くか» になります。
    """
    del scene, global_time  # JS 版と引数をそろえるために残す（今は使わない）
    from movo.core.math import Mat2D
    from movo.deformer import apply_modifiers
    from movo.physics import build_rig, normalize_motion, resolve_rig_pose

    scale = renderer.render_scale
    rig = renderer.rigs.get(layer["id"])
    if rig is None:
        spec = layer.get("rig") or (renderer.project.get("characters") or {}).get(layer.get("character"))
        if not spec:
            logger.warn(f'character layer "{layer["id"]}" has no rig; skipped')
            return None
        rig = build_rig(spec)
        renderer.rigs[layer["id"]] = rig
    pose = resolve_rig_pose(rig, ctx, {"motion": normalize_motion(layer.get("motion"))})

    buffer = Bitmap(js_round(renderer.width * scale), js_round(renderer.height * scale))
    center_x = (renderer.width / 2) * scale
    center_y = (renderer.height / 2) * scale
    ordered = sorted(rig["order"], key=lambda part: part.get("zIndex") or 0)

    for part in ordered:
        entry = pose.get(part.get("id"))
        if not entry:
            continue
        bitmap = None
        if renderer.assets is not None:
            bitmap = renderer.assets.get(part.get("asset"))
            if bitmap is None:
                bitmap = renderer.assets.get(f'{layer.get("character")}.{part.get("id")}')
        if bitmap is None:
            continue
        pivot = part.get("pivot") or [0.5, 0.5]
        part_modifiers = renderer._resolve_modifiers(entry.get("modifiers"), ctx)
        result = apply_modifiers(
            bitmap,
            part_modifiers["list"],
            {
                "time": scene_time,
                "seed": renderer.seed,
                "assets": renderer.assets,
                "meshResolution": max(4, js_round(renderer.mesh_resolution / 2)),
                "renderScale": 1,
                "boxWidth": bitmap.width,
                "boxHeight": bitmap.height,
                "plugins": renderer.plugins,
            },
        )
        matrix = Mat2D.translate(Mat2D.identity(), center_x, center_y)
        matrix = Mat2D.scale(matrix, scale, scale)
        matrix = Mat2D.multiply(matrix, entry["matrix"])
        matrix = Mat2D.translate(
            matrix,
            -pivot[0] * bitmap.width - result["offsetX"],
            -pivot[1] * bitmap.height - result["offsetY"],
        )
        result["mesh"].draw(
            buffer, result["bitmap"], matrix, {"alpha": entry["opacity"], "clampEdge": False}
        )

    return {
        "bitmap": buffer,
        "box_width": renderer.width,
        "box_height": renderer.height,
        # アンカーは «リグの根元»（バッファの中央）を指す
        "anchor_width": 0,
        "anchor_height": 0,
        "origin_x": renderer.width / 2,
        "origin_y": renderer.height / 2,
        "scale": scale,
    }


# ══════════════════════════════════════════════════════════════════
# パーティクル
# ══════════════════════════════════════════════════════════════════

def render_particle_layer(renderer, layer, ctx, transform):
    """パーティクルレイヤーの入口。

    JS 版の ``renderParticles`` に相当しますが、**粒を 1 個ずつ回す部分は
    ここにはありません。** Python の :class:`movo.renderer.particles.ParticleSystem`
    は粒ごとのオブジェクトではなく «NumPy の配列の束» で、描画も
    :func:`movo.renderer.particles.render_particles` が Numba で
    «粒の囲む矩形の中だけ» を塗るようにできています。ここはその呼ぶ側です。

    名前が ``particles.render_particles`` とぶつかるので関数名を
    ``render_particle_layer`` にし、下で ``render_particles`` の別名も貼っています。
    """
    scale = renderer.render_scale
    system = renderer.particles.get(layer.get("id"))
    if system is None:
        return None

    emitter = _spec_of(layer.get("emitter") if layer.get("emitter") is not None else layer.get("particles"), ctx)
    sprite = renderer.assets.get(emitter["asset"]) if emitter.get("asset") else None
    buffer = _draw_particles(
        system,
        emitter,
        renderer.width,
        renderer.height,
        transform,
        scale,
        sprite,
    )
    return {
        "bitmap": buffer,
        "box_width": renderer.width,
        "box_height": renderer.height,
        # 粒はワールド座標で進むので、レイヤーの «原点» は rig と同じく
        # バッファの中心に置きます。anchor を 0 にしておくと、変形が
        # «中心まわり» になって画面外へ飛びません。
        "anchor_width": 0,
        "anchor_height": 0,
        "origin_x": renderer.width / 2,
        "origin_y": renderer.height / 2,
        "scale": scale,
    }


#: JS 版と同じ名前でも呼べるようにしておく（Renderer 側がどちらで呼んでもよい）。
render_particles = render_particle_layer


# ══════════════════════════════════════════════════════════════════
# 波形（オーディオビジュアライザ）
# ══════════════════════════════════════════════════════════════════

def render_waveform(renderer, layer, ctx, transform, global_time):
    """音の波形レイヤー。

    ``style: "bars"`` は棒グラフ、``"wave"`` は生波形、``"mirror"`` は中心線で
    対称にした棒グラフです。値は **書き出す音そのもの**から取るので、出来上がった
    動画の音と必ず合います。

    JS は «バー 1 本ごとに全画面ぶんの被覆率バッファを作って塗る» ので、
    512 本だと画面 512 枚ぶんを触ります。ここでは輪郭を 1 度にラスタライズし、
    色は ``endColor`` があるときだけ «列 → バー番号» の一括写像で乗せます
    （:func:`movo.renderer.raster.fill_coverage_with` は NumPy で色を作る入口です）。
    """
    scale = renderer.render_scale
    spec = _spec_of(layer.get("waveform"), ctx)
    width = max(4, js_round(_opt(transform, "width", renderer.width * 0.6)))
    height = max(4, js_round(_opt(transform, "height", 200)))
    bitmap = Bitmap(js_round(width * scale), js_round(height * scale))
    count = int(clamp(js_round(_opt(spec, "bars", 48)), 2, 512))
    style = _opt(spec, "style", "bars")
    color = _opt(spec, "color", "#ffffff")
    end_color = _opt(spec, "endColor", None)
    gap = _opt(spec, "gap", 0.25)
    gain = _opt(spec, "gain", 1)
    window_seconds = _opt(spec, "window", 0.12)
    radius = _opt(spec, "radius", 0)
    values = sample_audio_window(renderer, global_time, count, window_seconds)

    gap_k = clamp(gap, 0, 0.9)
    bar_width = (width / count) * (1 - gap_k)
    contours = []
    for i in range(count):
        value = clamp(float(values[i]) * gain, 0, 1)
        x = (i * width) / count + ((width / count) * gap_k) / 2
        if style == "wave":
            centre = height / 2
            amplitude = value * (height / 2)
            following = values[i + 1] if i + 1 < count else values[i]
            nxt = clamp(float(following) * gain, 0, 1) * (height / 2)
            thickness = max(1, _opt(spec, "thickness", 3))
            contours.append([
                x * scale, (centre - amplitude) * scale,
                (x + width / count) * scale, (centre - nxt) * scale,
                (x + width / count) * scale, (centre - nxt + thickness) * scale,
                x * scale, (centre - amplitude + thickness) * scale,
            ])
        elif style == "mirror":
            centre = height / 2
            amplitude = max(1, value * (height / 2))
            contours.append(rect_contour_flat(
                x * scale, (centre - amplitude) * scale,
                bar_width * scale, amplitude * 2 * scale, radius * scale,
            ))
        else:
            bar_height = max(1, value * height)
            contours.append(rect_contour_flat(
                x * scale, (height - bar_height) * scale,
                bar_width * scale, bar_height * scale, radius * scale,
            ))

    region = rasterize_contours(contours, bitmap.width, bitmap.height)
    if end_color is None:
        # 全部同じ色なので 1 回で塗れる（既定の経路。ここが一番速い）。
        fill_coverage(bitmap, region, color, 1)
    else:
        # 端から端へ色が変わる場合。バーは x で綺麗に分かれているので、
        # «列 → バー番号 → 色» の引きだけで画素ごとの色が決まります。
        table = np.empty((count, 4), np.float64)
        for i in range(count):
            table[i] = mix_css(color, end_color, i / max(1, count - 1))

        def shade(xs, _ys, _table=table, _count=count, _width=width, _scale=scale):
            index = np.clip((xs / _scale) * _count / _width, 0, _count - 1).astype(np.int64)
            return _table[index]

        fill_coverage_with(bitmap, region, shade, 1)

    return {
        "bitmap": bitmap,
        "box_width": width,
        "box_height": height,
        "origin_x": 0,
        "origin_y": 0,
        "scale": scale,
    }


# ══════════════════════════════════════════════════════════════════
# 集中線（スピード線）
# ══════════════════════════════════════════════════════════════════

def render_speed_lines(renderer, layer, ctx, transform, scene_time):
    """集中線。線端の形・湾曲・扇状の範囲・二重化に対応します。

    ``innerRadius`` が AviUtl の «中心幅»、``density`` が «濃さ» に相当します。
    innerRadius をキーフレームで動かすとカットイン演出になります。
    既定値は JS 版と同じ見た目になるようにしてあります（後方互換）。

    **乱数を引く順番は JS 版と 1 回もずらしていません。** 線ごとに
    「間引き → 角度 → 内側の半径 → 太さ」の順で引くところまで同じなので、
    同じ種からは同じ絵が出ます。線の «背骨» の分割だけは NumPy の一括演算です
    （ここは乱数を引かないので、まとめても数列がずれません）。
    """
    scale = renderer.render_scale
    spec = _spec_of(layer.get("speedLines"), ctx)
    width = max(8, js_round(_opt(transform, "width", renderer.width)))
    height = max(8, js_round(_opt(transform, "height", renderer.height)))
    bitmap = Bitmap(js_round(width * scale), js_round(height * scale))
    count = int(clamp(js_round(_opt(spec, "count", 140)), 4, 2000))
    inner_radius = clamp(_opt(spec, "innerRadius", 0.35), 0, 2)
    outer_radius = clamp(_opt(spec, "outerRadius", 1.25), 0.05, 4)
    thickness = max(0.05, _opt(spec, "thickness", 1))
    density = clamp(_opt(spec, "density", 0.5), 0, 1)
    jitter = clamp(_opt(spec, "jitter", 0.6), 0, 1)
    speed = _opt(spec, "speed", 0)
    cx = _opt(spec, "centerX", 0.5) * width * scale
    cy = _opt(spec, "centerY", 0.5) * height * scale
    diagonal = math.hypot(width, height) * scale * 0.5
    # speed を入れると時間で本数のパターンが回る（線が流れて見える）。
    rotation_seed = math.floor(scene_time * abs(speed) * 10)

    style = _opt(spec, "style", "taperInner")
    curve = _opt(spec, "curve", 0)
    vertical = spec.get("direction") == "vertical"
    wedge = _opt(spec, "wedge", None)
    # 途中で太さが変わる線と曲がる線は、節を入れないと形にならない。
    # taperInner と uniform は直線 1 本で足りるので分割しない（従来どおり軽い）。
    needs_segments = curve != 0 or style in ("taperBoth", "dotted")
    if needs_segments:
        segments = int(clamp(js_round(_opt(spec, "segments", 24 if style == "dotted" else 10)), 2, 64))
    else:
        segments = 1

    # 分割の位置は全部の線で同じなので、1 度だけ作って使い回します。
    t_axis = np.arange(segments + 1, dtype=np.float64) / segments

    def width_at(t: np.ndarray, half_width: float) -> np.ndarray:
        """線端の太さ。taperInner は中心側を尖らせる（従来どおり）。"""
        if style == "uniform":
            return np.full_like(t, half_width)
        if style == "taperBoth":
            return half_width * np.sin(np.clip(t, 0, 1) * math.pi)
        if style == "dotted":
            # 破線状。t を刻んで «点» の間だけ太さを持たせる
            return np.where(np.mod(t * 8, 1.0) < 0.5, half_width, 0.0)
        return half_width * t  # taperInner（既定）

    def build_contours(seed_salt: int, angle_offset: float) -> list[np.ndarray]:
        """1 層ぶんの輪郭を作る。二重集中線はこれを 2 回呼ぶ。"""
        base_seed = int(_opt(spec, "seed", 31))
        random = Random((base_seed ^ ((rotation_seed * 2654435761) & _U32) ^ seed_salt) & _U32)
        contours: list[np.ndarray] = []
        for i in range(count):
            if random() > density * 1.6:
                continue
            if wedge:
                # 扇状：指定した角度の範囲にだけ線を出す
                start = (_opt(wedge, "startAngle", -40) * math.pi) / 180
                end = (_opt(wedge, "endAngle", 40) * math.pi) / 180
                angle = start + ((i + random() * jitter) / count) * (end - start)
            elif vertical:
                # 横方向のカットイン：線が上下から集まる
                side = -1 if i % 2 == 0 else 1
                angle = ((random() - 0.5) * 2 * 0.9 + side * 1.4) * math.pi * 0.5
            else:
                angle = (i / count) * TAU + (random() - 0.5) * (TAU / count) * jitter * 2
            angle += scene_time * speed + angle_offset

            inner = diagonal * inner_radius * (1 - jitter * 0.35 * random())
            outer = diagonal * outer_radius
            half_width = (thickness * (0.4 + random() * 1.6) * scale) / 2

            # 中心から外へ向かう «背骨» を作り、その両側に幅を付ける。
            # ここは NumPy の一括演算（1 本あたり segments+1 点をまとめて計算）。
            radius = inner + (outer - inner) * t_axis
            # curve で線を湾曲させる（外へ行くほど角度がずれる）
            bent = angle + curve * t_axis * t_axis
            cos_b = np.cos(bent)
            sin_b = np.sin(bent)
            px = cx + cos_b * radius
            py = cy + sin_b * radius
            w = width_at(t_axis, half_width)
            nx = -sin_b * w
            ny = cos_b * w

            n = segments + 1
            contour = np.empty(n * 4, np.float64)
            contour[0 : 2 * n : 2] = px + nx
            contour[1 : 2 * n : 2] = py + ny
            # 帰りは «逆向き» に辿って閉じた輪郭にする（JS 版と同じ並び）
            contour[2 * n :: 2] = (px - nx)[::-1]
            contour[2 * n + 1 :: 2] = (py - ny)[::-1]
            contours.append(contour)
        return contours

    double = _opt(spec, "doubleLayer", None)
    if double:
        # 二重集中線：ずらした色違いの層を先に敷く
        offset_angle = (_opt(double, "offset", 6) * math.pi) / 180
        contours = build_contours(7919, offset_angle)
        if contours:
            region = rasterize_contours(contours, bitmap.width, bitmap.height)
            fill_coverage(bitmap, region, _opt(double, "color", "#ff4d6d"),
                          clamp(_opt(double, "opacity", 0.5), 0, 1))

    contours = build_contours(0, 0)
    if contours:
        region = rasterize_contours(contours, bitmap.width, bitmap.height)
        fill_coverage(bitmap, region, _opt(spec, "color", "#ffffff"), 1)

    return {
        "bitmap": bitmap,
        "box_width": width,
        "box_height": height,
        "origin_x": 0,
        "origin_y": 0,
        "scale": scale,
    }


# ══════════════════════════════════════════════════════════════════
# 音の窓
# ══════════════════════════════════════════════════════════════════

def sample_audio_window(renderer, time: float, count: int, window_seconds: float) -> np.ndarray:
    """``time`` のまわりの音を ``count`` 個の RMS に束ねる（無ければ包絡線で代用）。

    JS は «標本 1 個ずつ» の三重ループでした。窓が 1 秒を超えると数十万回まわるので、
    ここでは ``(count, per)`` の添字表を 1 度だけ組んで NumPy の一括演算にしています。
    範囲外の添字は «使わなかった» として数から外すので、JS の ``continue`` と同じ結果です。
    """
    count = int(count)
    audio = getattr(renderer, "audio", None)
    if audio is not None and getattr(audio, "length", 0) > 0:
        sample_rate = getattr(audio, "sample_rate", None) or getattr(audio, "sampleRate", 48000)
        total = max(count, js_round(window_seconds * sample_rate))
        start = js_round(time * sample_rate) - (total // 2)
        per = max(1, total // count)

        index = start + np.arange(count, dtype=np.int64)[:, None] * per + np.arange(per, dtype=np.int64)[None, :]
        valid = (index >= 0) & (index < audio.length)
        # 範囲外は 0 番を読むが、あとで valid で落とすので値は使われない。
        safe = np.where(valid, index, 0)

        channels = list(getattr(audio, "channels", []) or [])
        mono = np.zeros(index.shape, np.float64)
        for channel in channels:
            mono += np.asarray(channel, np.float64)[safe]
        mono /= max(1, len(channels))

        squared = np.where(valid, mono * mono, 0.0)
        used = valid.sum(axis=1)
        total_energy = squared.sum(axis=1)
        values = np.zeros(count, np.float32)
        alive = used > 0
        values[alive] = np.sqrt(total_energy[alive] / used[alive]) * 2.2
        return values

    # 音源が無い：包絡線から «それらしい形» を作って、レイヤーが読めるようにする。
    audio_state = renderer._audio_at(time)
    bands = audio_state.get("bands") or []
    level = audio_state.get("level", 0)
    i = np.arange(count, dtype=np.float64)
    band_index = np.floor((i / count) * 3).astype(np.int64)
    band_table = np.array(
        [bands[k] if k < len(bands) and bands[k] is not None else level for k in range(3)],
        np.float64,
    )
    values = np.clip(band_table[np.clip(band_index, 0, 2)] * (0.6 + 0.4 * np.sin(i * 1.7 + time * 4)), 0, 1)

    # 音源も包絡線も無いときは全部 0 になり、バーが消えて「壊れている」ように見える。
    # 素材が無いときのプレースホルダと同じ考えで、決定的な待機パターンを出しておく。
    if getattr(renderer, "audio", None) is None and getattr(renderer, "audio_envelope", None) is None:
        if not getattr(renderer, "_idle_waveform_warned", False):
            renderer._idle_waveform_warned = True
            logger.verbose("音源が無いため波形は待機パターンを表示します（audio を追加すると音に同期します）")
        project = renderer.project if isinstance(renderer.project, dict) else {}
        bpm = _opt(project.get("project") or {}, "bpm", 120)
        beat = (time * bpm) / 60
        pulse = 0.45 + 0.35 * math.pow(max(0.0, math.sin(beat * math.pi)), 3)
        u = i / max(1, count - 1)
        # 低域が高く高域が低い、それらしい傾き＋固定ノイズ
        tilt = 0.35 + 0.65 * np.power(1 - np.abs(u * 2 - 1), 1.3)
        ripple = 0.5 + 0.5 * np.sin(i * 2.4 + time * 5.1) * np.sin(i * 0.7 + time * 1.9)
        values = np.clip(pulse * tilt * (0.55 + 0.45 * ripple), 0, 1)

    return values.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# フレームバッファ（調整レイヤー）
# ══════════════════════════════════════════════════════════════════

def render_frame_buffer(renderer, layer, ctx, target):
    """フレームバッファ（調整レイヤー）。

    このレイヤーに到達した時点で描かれているものをそのまま content にします。
    あとは通常のレイヤーと同じ経路なので、変形・エフェクト・マスク・ブレンドが
    そのまま使えます。「下にあるものだけをぼかす」「一部だけ歪ませる」といった
    調整レイヤー的な使い方ができます。

    :param target: このレイヤーを描き込む先（＝すでに描かれた絵）
    """
    spec = _spec_of(layer.get("frameBuffer"), ctx)
    source = _opt(spec, "source", "below")
    # below : いま描き込んでいる先（グループの中ならそのグループのバッファ）
    # scene : このシーンの描画先
    # frame : シーンをまたいだ合成済みフレーム
    # ``or`` ではなく «None かどうか» で見ます。Bitmap に真偽の決まりは無いので
    # 今は同じ結果ですが、あとで «空のときは偽» を足されたら静かに壊れるためです。
    if source == "frame":
        from_bitmap = getattr(renderer, "_frame_canvas", None)
    elif source == "scene":
        from_bitmap = getattr(renderer, "_scene_target", None)
    else:
        from_bitmap = target
    if from_bitmap is None:
        from_bitmap = target
    if from_bitmap is None:
        return None

    # 取り込みはフレーム 1 枚ぶんのコピーなので、数が増えると重い。
    # 設定ミスで大量に置かれたときに気付けるよう一度だけ警告する。
    renderer._frame_buffer_count = (getattr(renderer, "_frame_buffer_count", 0) or 0) + 1
    if renderer._frame_buffer_count > 32 and not getattr(renderer, "_frame_buffer_warned", False):
        renderer._frame_buffer_warned = True
        logger.warn("1 フレームに frameBuffer が 32 枚を超えています。フレームのコピーが増えて重くなります")

    scale = renderer.render_scale
    return {
        # JS は new Bitmap + data.set でしたが、Python は copy() が同じことを
        # 1 回の memcpy でやります（Bitmap に clone() はありません）。
        "bitmap": from_bitmap.copy(),
        "box_width": from_bitmap.width / scale,
        "box_height": from_bitmap.height / scale,
        "origin_x": 0,
        "origin_y": 0,
        "scale": scale,
    }


# ══════════════════════════════════════════════════════════════════
# フラクタルノイズ
# ══════════════════════════════════════════════════════════════════

def render_fractal_noise(renderer, layer, ctx, transform, scene_time):
    """フラクタルノイズのレイヤー（雲・煙・霧）。

    ``evolution`` は 3 次元目として渡すので、模様がスクロールするのではなく
    「湧いて変わる」動きになります。``scrollX`` / ``scrollY`` を足すと流れます。
    シードと時刻が同じなら必ず同じ模様です。

    **JS の «画素ごとに fbm2D を呼ぶ二重 for» は使いません。**
    :func:`movo.core.rng.fbm2d_grid` が同じ式を Numba でまとめて解き
    （1280x720 で 32 秒 → 52 ミリ秒）、色の割り当ては NumPy の一括演算です。
    """
    spec = _spec_of(layer.get("fractalNoise") if layer.get("fractalNoise") is not None else layer.get("noise"), ctx)
    width = max(1, js_round(_opt(transform, "width", renderer.width)))
    height = max(1, js_round(_opt(transform, "height", renderer.height)))
    resolution = int(clamp(js_round(_opt(spec, "resolution", 256)), 8, 2048))
    step_x = max(1, js_round(width / resolution))
    step_y = max(1, js_round(height / resolution))
    grid_width = math.ceil(width / step_x)
    grid_height = math.ceil(height / step_y)

    seed = (int(renderer.seed) ^ hash_code(layer.get("id") or "fractalNoise")
            ^ js_round(_opt(spec, "seed", 0))) & _U32
    # ここの scale は «ノイズの細かさ»。renderer.render_scale とは別物なので
    # 名前を分けています（JS では両方 scale で紛らわしかった）。
    freq_x = _opt(spec, "scale", 0.004)
    freq_y = _opt(spec, "scaleY", freq_x)
    evolution = _opt(spec, "evolution", 0)
    try:
        evolution = float(evolution)
        if not math.isfinite(evolution):
            evolution = 0.0
    except (TypeError, ValueError):
        evolution = 0.0
    scroll_x = _opt(spec, "scrollX", 0) * scene_time
    scroll_y = _opt(spec, "scrollY", 0) * scene_time
    contrast = _opt(spec, "contrast", 1)
    brightness = _opt(spec, "brightness", 0)
    kind = _opt(spec, "type", "fbm")

    # nx = (x * stepX + scrollX) * freqX を «起点と刻み» に読み替えて一括で解く
    raw = fbm2d_grid(
        grid_width,
        grid_height,
        x0=scroll_x * freq_x,
        y0=scroll_y * freq_y,
        dx=step_x * freq_x,
        dy=step_y * freq_y,
        seed=seed,
        z=evolution,
        octaves=_opt(spec, "octaves", 5),
        lacunarity=_opt(spec, "lacunarity", 2),
        gain=_opt(spec, "gain", 0.5),
        type=kind,
    )
    # fbm は [-1,1]、turbulent / ridged は [0,1] なので 0..1 に揃える
    if kind == "fbm":
        raw = raw * 0.5 + 0.5
    t = np.clip((raw - 0.5) * contrast + 0.5 + brightness, 0, 1)

    bitmap = Bitmap(grid_width, grid_height)
    bitmap.data[...] = _lerp_colors(t, parse_color(_opt(spec, "colorA", "#000000")),
                                    parse_color(_opt(spec, "colorB", "#ffffff")))
    return {
        "bitmap": bitmap,
        "box_width": width,
        "box_height": height,
        "origin_x": 0,
        "origin_y": 0,
        # 粗いグリッドで作ったので、あとで «元の大きさ» に伸ばしてもらう
        "scale": bitmap.width / width,
    }


# ══════════════════════════════════════════════════════════════════
# シェーダー（式で作る絵）
# ══════════════════════════════════════════════════════════════════

def _mentions(node, names: frozenset) -> bool:
    """構文木のどこかに ``names`` の綴りが現れるか（安全側に倒した判定）。

    式エンジンは «木をたどる評価器» なので、識別子が本当に変数として使われて
    いるかを厳密に見るには節の種類を知る必要があります。ここは
    **«入っていたら使われているものとして扱う»** だけの判定にしてあります。
    取りこぼす方向（使っていないのに使っていると見なす）にしか外れないので、
    速い経路に誤って乗ることはありません。
    """
    if isinstance(node, str):
        return node in names
    if isinstance(node, (tuple, list)):
        return any(_mentions(child, names) for child in node)
    return False


_PIXEL_VARS = frozenset({"u", "v", "x", "y"})


def render_shader(renderer, layer, ctx, transform, scene_time):
    """式で作る手続き的レイヤー：``shader.expression`` が返す 0..1 を
    ``shader.colorA`` → ``shader.colorB`` に写します。

    ## なぜ «式の評価» だけ NumPy にできないか

    Movo の式エンジンは ``eval`` を使わない自前のサンドボックスで、評価器が
    演算のたびに ``to_number()`` を通します。``to_number()`` は NumPy の配列を
    «数ではないもの» と見て 0.0 にするので、``u`` に配列を入れて 1 回で解く、
    という手は **黙って真っ黒な絵を出す**ことになります。評価器は他のエージェント
    が持っているファイルなので、ここからは触りません。

    そこで次の 2 段構えにしています。

    * 式が ``u`` / ``v`` / ``x`` / ``y`` を使っていない → **1 回だけ評価**して
      NumPy で一様に塗る（``"0.5"`` や ``"wave(time)"`` はここに乗ります）。
    * 使っている → 格子ごとに評価する。ただし **構文木は 1 度だけ組み**、
      スコープの辞書も 1 個を使い回します（毎回 ``{**ctx.scope, ...}`` を
      作り直すと、それだけで倍かかります）。

    どちらの場合も **色を作るところは NumPy の一括演算**です（``_lerp_colors``）。
    格子は画面ではなく ``shader.resolution``（既定 128）の粗さなので、
    既定値では 128x72 ≒ 9 千回の評価で済みます。
    """
    spec = layer.get("shader") or {}
    width = max(1, js_round(_opt(transform, "width", renderer.width)))
    height = max(1, js_round(_opt(transform, "height", renderer.height)))
    resolution = int(clamp(js_round(_opt(spec, "resolution", 128)), 8, 1024))
    step_x = max(1, js_round(width / resolution))
    step_y = max(1, js_round(height / resolution))
    grid_width = math.ceil(width / step_x)
    grid_height = math.ceil(height / step_y)
    source = _opt(spec, "expression", "0.5")

    engine = renderer.engine
    scope = dict(ctx.get("scope") or {})
    scope["time"] = scene_time
    layer_id = layer.get("id")

    try:
        ast = engine.compile(source)
    except Exception as err:  # 構文が壊れている
        logger.warn(f'shader layer "{layer_id}" failed: {err}')
        return None

    if not _mentions(ast, _PIXEL_VARS):
        # 画素に依らない式。1 回解いて全面同じ値にする。
        scope.update({"u": 0.0, "v": 0.0, "x": 0.0, "y": 0.0})
        try:
            value = _to_number(_evaluate_ast(ast, scope, engine.functions))
        except Exception as err:
            logger.warn(f'shader layer "{layer_id}" failed: {err}')
            return None
        t = np.full((grid_height, grid_width), clamp(value, 0, 1), np.float64)
    else:
        if grid_width * grid_height > 65536 and not getattr(renderer, "_shader_grid_warned", False):
            renderer._shader_grid_warned = True
            logger.warn(
                f'shader レイヤー "{layer_id}" の格子が {grid_width}x{grid_height} あります。'
                "式は 1 マスずつ解くので、resolution を下げると軽くなります"
            )
        t = np.empty((grid_height, grid_width), np.float64)
        # 分母は式の中で毎回同じなので外に出す
        inv_u = 1.0 / max(1, grid_width - 1)
        inv_v = 1.0 / max(1, grid_height - 1)
        functions = engine.functions
        for gy in range(grid_height):
            scope["v"] = gy * inv_v
            scope["y"] = gy * step_y
            row = t[gy]
            for gx in range(grid_width):
                scope["u"] = gx * inv_u
                scope["x"] = gx * step_x
                try:
                    row[gx] = _to_number(_evaluate_ast(ast, scope, functions))
                except Exception as err:
                    logger.warn(f'shader layer "{layer_id}" failed: {err}')
                    return None
        np.clip(t, 0, 1, out=t)

    bitmap = Bitmap(grid_width, grid_height)
    bitmap.data[...] = _lerp_colors(t, parse_color(_opt(spec, "colorA", "#000000")),
                                    parse_color(_opt(spec, "colorB", "#ffffff")))
    return {
        "bitmap": bitmap,
        "box_width": width,
        "box_height": height,
        "origin_x": 0,
        "origin_y": 0,
        "scale": bitmap.width / width,
    }


# ══════════════════════════════════════════════════════════════════
# 生成レイヤー（星空・線・ネオン・メタボール・水面・3D・図形アニメ）
# ══════════════════════════════════════════════════════════════════
#
# 実装は layers_generator.py にあります。ここは «同じ入口から呼べる» ように
# 名前を通すだけです。移植の途中でまだファイルが無いことがあるので、
# import に失敗したら «呼ばれたときにもう一度試す» 包みに差し替えます
# （そうしないと、あとからファイルが入っても再起動するまで直りません）。

try:  # pragma: no cover - ファイルが揃っていれば単なる再輸出
    from movo.renderer.layers_generator import render_generator
except ImportError:  # pragma: no cover - 移植の途中だけ通る
    _generator_impl = None
    _generator_warned = False

    def render_generator(*args, **kwargs):
        """``layers_generator`` が入るまでの繋ぎ。入っていれば本物へ委ねます。"""
        global _generator_impl, _generator_warned
        if _generator_impl is None:
            try:
                from movo.renderer.layers_generator import render_generator as impl
            except ImportError:
                if not _generator_warned:
                    _generator_warned = True
                    logger.warn("生成レイヤーは movo/renderer/layers_generator.py が未移植のため飛ばします")
                return None
            _generator_impl = impl
        return _generator_impl(*args, **kwargs)


__all__ = [
    "render_character",
    "render_frame_buffer",
    "render_fractal_noise",
    "render_generator",
    "render_particle_layer",
    "render_particles",
    "render_shader",
    "render_speed_lines",
    "render_waveform",
    "sample_audio_window",
]
