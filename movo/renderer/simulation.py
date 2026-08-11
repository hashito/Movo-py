"""シミュレーションの準備と進行。

移植元: ``packages/renderer/src/simulation.js``

物理ボディ・ソフトチェーン・パーティクルの «作る» と «進める» をここに
まとめています。描画（Renderer）と切り離してあるのは、時間を進める処理と
絵を作る処理が混ざると追いにくいためです。状態は Renderer が持ったままなので、
各関数は第 1 引数で受け取ります。

## JS 版との食い違いと、その直し方

1. **``new Body({...})`` → ``Body(**options)``**
   Python 版の ``Body`` / ``SoftChain`` は «辞書 1 つ» ではなくキーワード引数を
   取ります（``World`` だけは辞書）。呼ぶ側をそちらに合わせました。移植済みの
   ``movo/physics`` を触るほうが影響範囲が広いためです。
2. **``body.fixedRotation`` が読めない**
   Python の ``Body`` は ``__slots__`` を持ち、値は NumPy の «列» に入って
   います。``fixed_rotation`` の getter は無く、属性を後から生やすこともでき
   ません。そこで **レンダラー側に ``_fixed_rotation`` という集合を持たせて**
   覚えます（``physics.fixedRotation`` は JSON にしか無い静的な指定なので、
   作るときに 1 回覚えれば足ります）。
3. **``chain.points`` は «点オブジェクトの配列» ではなく ``(N, 5)`` の NumPy 配列**
   列は ``x, y, px, py, pinned``。巻き戻しの ``point.px = point.x`` は
   スライス代入 1 行になります。
4. **``system.particles`` が無い**（粒は «配列の束»）
   ``shapeTarget`` の引き寄せは粒ごとの for を NumPy の一括演算に書き換えて
   います。粒 1 万個 × 4500 フレームを Python の for で回すと終わりません。
"""

from __future__ import annotations

import math

import numpy as np

from movo.animation.resolver import resolve_number
from movo.cli.console import logger
from movo.core.math import clamp, js_round, smoothstep, to_radians
from movo.physics import Body, SoftChain, create_shape

# ⚠ **`ParticleSystem` は 2 つあります。**
# `movo.physics`（soft.py）と `movo.renderer.particles` の両方に移植されました。
# JS 版は physics のものを import していますが、**描画側の
# `render_particles(system, …)` は renderer 版の `render()` を呼びます**
# （粒を «配列の束» で返すのは renderer 版だけ）。physics 版を渡すと
# `__init__() takes 1 positional argument` で落ちます（実際に落ちました）。
# 描く側と組になっているほうへそろえます。
from movo.renderer.particles import ParticleSystem
from movo.renderer.helpers import hash_code, within_range
from movo.renderer.particle_presets import resolve_preset
from movo.renderer.text import render_text, resolve_text_style
from movo.renderer.text_extras import create_random
from movo.timeline import find_layer, is_layer_active

# SoftChain の points の列（movo/physics/soft.py と同じ並び）
_SOFT_X, _SOFT_Y, _SOFT_PX, _SOFT_PY, _SOFT_PINNED = range(5)


def register_scene_bodies(renderer, scene: dict, world, config: dict) -> None:
    """シーンの中の «動くもの» を世界に登録する。"""

    def walk(layers: list) -> None:
        for layer in layers:
            if layer.get("children"):
                walk(layer["children"])
            if layer.get("type") == "particle":
                renderer.particles[layer["id"]] = create_particle_system(renderer, layer)
                continue
            physics = layer.get("physics")
            if not physics or physics.get("type") == "none":
                continue
            if physics.get("type") == "softChain":
                chain = create_soft_chain(renderer, layer, physics)
                renderer.soft_chains[layer["id"]] = chain
                world.add_soft_body(chain)
                continue
            body = create_body(renderer, layer, physics)
            if body is not None:
                world.add_body(body)
                renderer.bodies[layer["id"]] = body

    walk(scene["layers"])

    for constraint in config.get("constraints") or []:
        body_a = renderer.bodies.get(constraint.get("bodyA"))
        body_b = renderer.bodies.get(constraint.get("bodyB"))
        if body_a is None or body_b is None:
            logger.warn(
                f'constraint "{constraint.get("id") or constraint.get("type")}" '
                "references an unknown body; skipped"
            )
            continue
        world.add_constraint({**constraint, "bodyA": body_a, "bodyB": body_b})


def content_size_of(renderer, layer: dict) -> dict:
    """物理の «形» を作るための、レイヤーの見かけの大きさ。

    キーフレームで動く指定（辞書）は «静的な数» ではないので既定値に落とします。
    物理の形は最初に 1 回だけ決まるものなので、そこで時刻を持ち出さないためです。
    """
    transform = layer.get("transform") or {}

    def static_number(value, fallback):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback

    if layer.get("type") in ("image", "video"):
        bitmap = renderer.assets.get(layer.get("asset")) if renderer.assets else None
        return {
            "width": static_number(transform.get("width"), bitmap.width if bitmap else 100),
            "height": static_number(transform.get("height"), bitmap.height if bitmap else 100),
            "bitmap": bitmap,
        }
    if layer.get("type") == "shape":
        shape = layer.get("shape") or {}
        radius = shape.get("radius")
        doubled = radius * 2 if isinstance(radius, (int, float)) and not isinstance(radius, bool) else None
        return {
            "width": static_number(shape.get("width") if shape.get("width") is not None else doubled, 100),
            "height": static_number(shape.get("height") if shape.get("height") is not None else doubled, 100),
            "bitmap": None,
        }
    return {
        "width": static_number(transform.get("width"), 100),
        "height": static_number(transform.get("height"), 100),
        "bitmap": None,
    }


def create_body(renderer, layer: dict, physics: dict):
    size = content_size_of(renderer, layer)
    transform = layer.get("transform") or {}
    shape_spec = physics.get("shape") or {"type": "rectangle"}
    shape = create_shape(
        shape_spec,
        {
            "bitmap": renderer.assets.get(shape_spec["asset"]) if shape_spec.get("asset") else size["bitmap"],
            "width": size["width"],
            "height": size["height"],
            "alphaOutline": renderer.alpha_outline,
        },
    )
    x = transform.get("x") if isinstance(transform.get("x"), (int, float)) else 0
    y = transform.get("y") if isinstance(transform.get("y"), (int, float)) else 0
    rotation = transform.get("rotation") if isinstance(transform.get("rotation"), (int, float)) else 0
    velocity = physics.get("velocity") or {}
    # None は «指定なし» なので渡さない（Body 側の既定値をそのまま効かせるため）。
    options = {
        "id": layer["id"],
        "bodyType": physics.get("bodyType") or "dynamic",
        "shape": shape,
        "x": x,
        "y": y,
        "angle": to_radians(rotation),
        "velocityX": velocity.get("x", 0),
        "velocityY": velocity.get("y", 0),
        "angularVelocity": physics.get("angularVelocity", 0),
        "userData": {"layerId": layer["id"]},
    }
    for key in (
        "mass",
        "friction",
        "restitution",
        "linearDamping",
        "angularDamping",
        "gravityScale",
        "fixedRotation",
        "sensor",
        "collisionGroup",
        "collisionMask",
    ):
        if physics.get(key) is not None:
            options[key] = physics[key]
    body = Body(**options)
    # `body.fixedRotation` を読めないぶん、レンダラー側で覚えておく（冒頭の 2 番）。
    if physics.get("fixedRotation"):
        renderer._fixed_rotation.add(layer["id"])
    # 巻き戻し用の初期状態。Body は __slots__ なので属性を生やせず、
    # レンダラー側の辞書に置く。
    renderer._initial_states[layer["id"]] = {
        "x": body.position.x,
        "y": body.position.y,
        "angle": body.angle,
        "vx": body.velocity.x,
        "vy": body.velocity.y,
        "av": body.angular_velocity,
    }
    return body


def create_soft_chain(renderer, layer: dict, physics: dict) -> SoftChain:
    size = content_size_of(renderer, layer)
    transform = layer.get("transform") or {}
    options = {
        "id": layer["id"],
        "segments": physics.get("segments", 8),
        "length": physics.get("length") if physics.get("length") is not None else size["height"],
        "angle": physics.get("angle", 90),
        "origin": {
            "x": transform.get("x") if isinstance(transform.get("x"), (int, float)) else 0,
            "y": transform.get("y") if isinstance(transform.get("y"), (int, float)) else 0,
        },
    }
    for key in ("stiffness", "damping", "gravityScale"):
        if physics.get(key) is not None:
            options[key] = physics[key]
    return SoftChain(**options)


def create_particle_system(renderer, layer: dict) -> ParticleSystem:
    emitter = layer.get("emitter") or layer.get("particles") or {}
    transform = layer.get("transform") or {}
    # preset は雨・雪・桜吹雪などの定番設定。個別指定で上書きできる。
    preset_defaults = {}
    if emitter.get("preset"):
        preset_defaults = resolve_preset(emitter["preset"], renderer.width, renderer.height) or {}
        if not preset_defaults:
            from movo.renderer.particle_presets import list_particle_presets

            logger.warn(
                f'unknown particle preset "{emitter["preset"]}"; '
                f"available: {', '.join(list_particle_presets())}"
            )
        else:
            preset_defaults = dict(preset_defaults)
            # プリセットの寸法・速度は 1080p 基準で書いてあるので解像度に合わせる。
            # 利用者が emitter に直接書いた値はそのプロジェクトの座標系なので触らない。
            scale = renderer.height / 1080
            if scale != 1:
                if isinstance(preset_defaults.get("size"), (int, float)):
                    preset_defaults["size"] = max(0.5, preset_defaults["size"] * scale)
                if isinstance(preset_defaults.get("speed"), (int, float)):
                    preset_defaults["speed"] *= scale
            # 0 秒時点で画面が空っぽにならないよう、寿命ぶんだけ空回ししておく
            if preset_defaults.get("prewarm") is None:
                preset_defaults["prewarm"] = (preset_defaults.get("lifetime") or 2) * 0.9

    options = {
        "id": layer["id"],
        "seed": (renderer.seed ^ hash_code(layer["id"])) & 0xFFFFFFFF,
        "x": transform.get("x") if isinstance(transform.get("x"), (int, float)) else 0,
        "y": transform.get("y") if isinstance(transform.get("y"), (int, float)) else 0,
        **preset_defaults,
        **emitter,
    }
    # 位置や rate は式・キーフレームで書けるので、**辞書のまま渡すと
    # ParticleSystem が数として扱えません。** 静的な数以外は既定値に落とし、
    # 毎フレーム `update_driven_bodies` が今の時刻の値を書き戻します。
    for key in ("x", "y", "rate", "speed", "direction", "width", "height"):
        if key in options and not isinstance(options[key], (int, float)):
            options[key] = 0 if key in ("x", "y", "width", "height") else None
            if options[key] is None:
                del options[key]
    system = ParticleSystem(options)
    # shapeTarget 用。粒が生まれた順に «目標点の番号» を配る。
    system._shape_cursor = 0
    system._shape_progress = 0.0
    system._shape_target = None
    system._shape_index = np.zeros(0, np.int64)
    return system


# ------------------------------------------------------------------
# emitter.shapeTarget — 素材の形に粒を集める
# ------------------------------------------------------------------


def _shape_source_for(renderer, spec: dict):
    """目標点の «元絵» を用意する。

    `asset` は画像素材、`text` は文字列、`layer` は他のテキストレイヤー。
    文字を対象にできるのは、歌詞が «粒で書かれて散る» のが定番だからです。
    """
    if spec.get("asset"):
        bitmap = renderer.assets.get(spec["asset"]) if renderer.assets else None
        if bitmap is None:
            logger.warn(f'shapeTarget references an unknown asset "{spec["asset"]}"; ignored')
            return None
        return {"bitmap": bitmap, "key": f'asset:{spec["asset"]}'}

    content = spec.get("text")
    style = spec.get("style")
    if spec.get("layer"):
        found = None
        for scene in renderer.timeline["scenes"]:
            found = find_layer(scene["layers"], spec["layer"])
            if found:
                break
        if not found:
            logger.warn(f'shapeTarget references an unknown layer "{spec["layer"]}"; ignored')
            return None
        # アニメーションは «形» に効かないので、書いてあるままの値で 1 枚だけ描く
        resolved = resolve_text_style(
            found, {"text": found.get("text"), "style": found.get("style"), "font": found.get("font")}
        )
        content = resolved["content"]
        style = {**resolved["style"], **(spec.get("style") or {})}
    if not content:
        return None

    text_style = {"size": 240, "color": "#ffffff", "align": "center", **(style or {})}
    rendered = render_text(str(content), text_style, renderer.font_manager)
    key = f'text:{content}:{text_style.get("family") or ""}:{text_style.get("size")}'
    return {"bitmap": rendered["bitmap"], "key": key}


def _sample_shape_points(bitmap, count: int, threshold: float, channel: str, seed: int):
    """元絵から目標点を «決定的に» サンプリングする。

    返すのは中心を原点とした -0.5〜0.5 の相対座標です。ワールド座標にしないのは、
    位置や大きさをキーフレームで動かしてもサンプリングし直さずに済ませるため。

    **候補集めは NumPy です。** 3000x1750 の素材は全画素 500 万件になるので、
    Python の二重ループでは 1 回でも数秒かかります。
    """
    if count <= 0:
        return None
    data = bitmap.data
    height, width = data.shape[0], data.shape[1]
    stride = max(1, int(math.sqrt((width * height) / max(1, count * 60))))
    sub = data[::stride, ::stride]
    alpha = sub[:, :, 3].astype(np.float64) / 255
    if channel == "luma":
        value = (
            sub[:, :, 0] * 0.299 + sub[:, :, 1] * 0.587 + sub[:, :, 2] * 0.114
        ) / 255 * alpha
    else:
        value = alpha
    ys, xs = np.nonzero(value >= threshold)
    total = xs.size
    if total == 0:
        return None
    candidates_x = xs * stride
    candidates_y = ys * stride

    # 同じ点を何度も引くと «形» がまばらになるので、部分 Fisher-Yates で
    # 重複なしに選ぶ。**乱数を引く順まで JS 版と同じ**にしたいので、ここだけは
    # Python のループのまま（点数は粒の数ぶん＝数千件なので実害はない）。
    random = create_random(seed & 0xFFFFFFFF)
    order = np.arange(total)
    picks = min(count, total)
    for i in range(picks):
        j = i + int(random() * (total - i))
        order[i], order[j] = order[j], order[i]

    points = np.empty(count * 2, np.float64)
    for i in range(count):
        index = int(order[i % picks])
        # 間引いた格子がそのまま «縞» に見えないよう、格子の中で散らす
        jx = candidates_x[index] + random() * stride
        jy = candidates_y[index] + random() * stride
        points[i * 2] = jx / width - 0.5
        points[i * 2 + 1] = jy / height - 0.5
    return points


def _resolve_shape_target(renderer, system, spec: dict, ctx: dict, transform: dict):
    progress = clamp(resolve_number(spec.get("progress"), ctx, 1), 0, 1)
    strength = max(0.0, resolve_number(spec.get("strength"), ctx, 260))
    threshold = clamp(resolve_number(spec.get("threshold"), ctx, 0.5), 0, 1)
    channel = "luma" if spec.get("channel") == "luma" else "alpha"
    seed = js_round(spec.get("seed", 7) or 7)
    count = system.max_particles

    cache = renderer._shape_targets
    # 鍵に count・seed・threshold を含めるので、素材が同じなら 1 回しか走らない
    probe = (
        f'{spec.get("asset") or ""}|{spec.get("layer") or ""}|{spec.get("text") or ""}'
        f"|{count}|{seed}|{threshold}|{channel}"
    )
    if probe not in cache:
        source = _shape_source_for(renderer, spec)
        cache[probe] = (
            {
                "points": _sample_shape_points(source["bitmap"], count, threshold, channel, seed),
                "width": source["bitmap"].width,
                "height": source["bitmap"].height,
            }
            if source
            else None
        )
    entry = cache[probe]
    if not entry or entry["points"] is None:
        return None

    return {
        "points": entry["points"],
        "count": count,
        "centreX": resolve_number(spec.get("x"), ctx, transform.get("x", 0)),
        "centreY": resolve_number(spec.get("y"), ctx, transform.get("y", 0)),
        "spanX": resolve_number(spec.get("width"), ctx, entry["width"]),
        "spanY": resolve_number(spec.get("height"), ctx, entry["height"]),
        "strength": strength,
        "progress": progress,
        "release": resolve_number(spec.get("release"), ctx, system.speed),
    }


def _apply_shape_target(system, config: dict, dt: float) -> None:
    """粒を目標点へ引き寄せる。1 ステップぶん。**NumPy の一括演算です。**

    «粒の index → サンプル点» は生まれた順に配って粒に持たせます。毎フレーム
    割り当て直すと、粒が入れ替わるたびに形がちらついてしまうためです。
    Python 版は粒がオブジェクトではないので、**割り当てを別の配列
    （``system._shape_index``）に持ちます。**
    """
    n = system.count
    if n == 0:
        return
    count = config["count"]
    assigned = system._shape_index
    if assigned.size < n:
        # 新しく生まれたぶんだけ番号を配る（生まれた順は配列の末尾に積まれる）
        extra = n - assigned.size
        start = system._shape_cursor
        new = (np.arange(start, start + extra) % max(1, count)).astype(np.int64)
        assigned = np.concatenate([assigned, new])
        system._shape_cursor = int((start + extra) % max(1, count))
    elif assigned.size > n:
        # 粒が死んで詰められたぶんは、前から詰め直す（ParticleSystem は
        # 生き残りを前へ詰めるので、同じ操作をこちらにも掛ける必要がある）
        assigned = assigned[:n]
    system._shape_index = assigned

    points = config["points"]
    progress = config["progress"]
    strength = config["strength"]
    release = config["release"]

    # 張り付いている間に速度は消えるので、progress を戻しただけでは «また散る»
    # にならない。緩んだぶんを外向きの勢いに変えて、粒を解き放つ。
    loosened = max(0.0, (system._shape_progress or 0.0) - progress)
    system._shape_progress = progress
    if loosened > 0 and release > 0:
        # 向きは粒ごとの種から決める（乱数を回さないので巻き戻しても同じ）
        angle = system.p_seed[:n] * math.pi * 2 + assigned * 0.61803
        system.p_vx[:n] += np.cos(angle) * release * loosened
        system.p_vy[:n] += np.sin(angle) * release * loosened
    if progress <= 0 or strength <= 0:
        return

    # progress が 1 に近づいたら位置そのものを寄せる。バネのままだと重力ぶん
    # 下にずれ続けて «形» が崩れるので、最後だけ張り付かせている。
    snap = smoothstep(0.85, 1, progress)
    pull = strength * progress
    # 引く力に見合った減衰。入れないと目標のまわりで延々と振動する。
    damping = 1 / (1 + 2 * math.sqrt(pull) * dt)
    target_x = config["centreX"] + points[assigned * 2] * config["spanX"]
    target_y = config["centreY"] + points[assigned * 2 + 1] * config["spanY"]
    system.p_vx[:n] = (system.p_vx[:n] + (target_x - system.p_x[:n]) * pull * dt) * damping
    system.p_vy[:n] = (system.p_vy[:n] + (target_y - system.p_y[:n]) * pull * dt) * damping
    if snap > 0:
        system.p_x[:n] += (target_x - system.p_x[:n]) * snap
        system.p_y[:n] += (target_y - system.p_y[:n]) * snap
        system.p_vx[:n] *= 1 - snap
        system.p_vy[:n] *= 1 - snap


def reset_simulation(renderer) -> None:
    """すべてを 0 フレーム目の状態へ戻す（巻き戻し）。"""
    for layer_id, body in renderer.bodies.items():
        initial = renderer._initial_states.get(layer_id)
        if not initial:
            continue
        body.position.x = initial["x"]
        body.position.y = initial["y"]
        body.angle = initial["angle"]
        body.velocity.x = initial["vx"]
        body.velocity.y = initial["vy"]
        body.angular_velocity = initial["av"]
    for chain in renderer.soft_chains.values():
        # 前フレームの位置＝今の位置にする（ヴァーレ積分なので速度が消える）
        chain.points[:, _SOFT_PX] = chain.points[:, _SOFT_X]
        chain.points[:, _SOFT_PY] = chain.points[:, _SOFT_Y]
    for layer_id, system in renderer.particles.items():
        system.reset()
        # 目標点の配り直し。粒を捨てたのにカーソルだけ進んでいると、
        # 巻き戻したときに «同じ時刻なのに違う形» になってしまう。
        system._shape_cursor = 0
        system._shape_progress = 0.0
        system._shape_index = np.zeros(0, np.int64)
        system.warmup(world_for_layer(renderer, layer_id), 1 / renderer.timeline["fps"])
    for world in renderer.worlds.values():
        world.time = 0.0
        world.step_count = 0
    # 履歴も破棄する（巻き戻して描き直したときに古い残像が混ざらないように）
    renderer.layer_history.clear()
    renderer.history_bytes = 0
    renderer.simulated_frame = -1


def advance_simulation(renderer, frame_index: int) -> None:
    """`frame_index` まで世界を進める。

    順番でないフレームを描くとシミュレーションをやり直すので、**まとめて
    描いたときと 1 枚だけ描いたときで結果が変わりません**（決定性）。
    """
    if not renderer.worlds and not renderer.particles:
        return
    # 最初の 1 回だけ prewarm を消化する
    if not renderer._warmed_up:
        renderer._warmed_up = True
        for layer_id, system in renderer.particles.items():
            system.warmup(world_for_layer(renderer, layer_id), 1 / renderer.timeline["fps"])
    if frame_index < renderer.simulated_frame:
        reset_simulation(renderer)
    fps = renderer.timeline["fps"]
    while renderer.simulated_frame < frame_index:
        next_frame = renderer.simulated_frame + 1
        time = next_frame / fps
        dt = 1 / fps
        update_driven_bodies(renderer, time)
        for scene_id, world in renderer.worlds.items():
            scene = next((s for s in renderer.timeline["scenes"] if s["id"] == scene_id), None)
            if scene is None:
                continue
            if time < scene["start"] or time > scene["end"] + dt:
                continue
            steps = max(1, js_round(dt / world.time_step))
            for _ in range(steps):
                world.step(world.time_step)
        for layer_id, system in renderer.particles.items():
            world = world_for_layer(renderer, layer_id)
            system.step(dt, world)
            # 引き寄せは «進めたあと» に効かせる。step の中の重力・抗力と喧嘩しない。
            if system._shape_target:
                _apply_shape_target(system, system._shape_target, dt)
        renderer.simulated_frame = next_frame


def world_for_layer(renderer, layer_id: str):
    for scene in renderer.timeline["scenes"]:
        if find_layer(scene["layers"], layer_id):
            return renderer.worlds.get(scene["id"])
    return next(iter(renderer.worlds.values()), None)


def update_driven_bodies(renderer, time: float) -> None:
    """アニメーションで «動かされる側» の物理を、進める前に書き戻す。"""
    for scene in renderer.timeline["scenes"]:
        scene_time = time - scene["start"]

        def walk(layers: list) -> None:
            for layer in layers:
                if layer.get("children"):
                    walk(layer["children"])
                layer_id = layer.get("id")
                if not (
                    layer_id in renderer.bodies
                    or layer_id in renderer.soft_chains
                    or layer_id in renderer.particles
                ):
                    continue
                ctx = renderer._context_for(layer, scene, scene_time, time, None)
                transform = renderer._resolve_transform(layer, ctx)

                body = renderer.bodies.get(layer_id)
                if body is not None:
                    control = layer.get("physicsControl") or (layer.get("physics") or {}).get("control") or {}
                    mode = control.get("mode") or ("physics" if body.type == "dynamic" else "animation")
                    if mode == "animation" or (mode == "override" and within_range(control, scene_time)):
                        body.position.x = transform["x"]
                        body.position.y = transform["y"]
                        body.angle = to_radians(transform["rotation"])
                        body.velocity.x = 0
                        body.velocity.y = 0
                        body.angular_velocity = 0
                    elif mode == "follow":
                        stiffness = control.get("followStiffness", 20)
                        damping = control.get("followDamping", 4)
                        body.apply_force(
                            (transform["x"] - body.position.x) * stiffness * body.mass
                            - body.velocity.x * damping * body.mass,
                            (transform["y"] - body.position.y) * stiffness * body.mass
                            - body.velocity.y * damping * body.mass,
                        )
                    elif body.type == "kinematic":
                        body.velocity.x = (transform["x"] - body.position.x) * renderer.timeline["fps"]
                        body.velocity.y = (transform["y"] - body.position.y) * renderer.timeline["fps"]

                chain = renderer.soft_chains.get(layer_id)
                if chain is not None:
                    chain.set_origin(transform["x"], transform["y"])
                    wind = (layer.get("physics") or {}).get("wind")
                    if wind:
                        chain.set_wind(
                            resolve_number(wind.get("x"), ctx, 0), resolve_number(wind.get("y"), ctx, 0)
                        )

                system = renderer.particles.get(layer_id)
                if system is not None:
                    system.emitter["x"] = transform["x"]
                    system.emitter["y"] = transform["y"]
                    emitter = layer.get("emitter") or layer.get("particles") or {}
                    if emitter.get("rate") is not None:
                        system.rate = resolve_number(emitter["rate"], ctx, system.rate)
                    if emitter.get("direction") is not None:
                        system.direction = resolve_number(emitter["direction"], ctx, system.direction)
                    if emitter.get("speed") is not None:
                        system.speed = resolve_number(emitter["speed"], ctx, system.speed)
                    # 形状ターゲットはここで «今の時刻の値» にしておき、step の直後に効かせる
                    system._shape_target = (
                        _resolve_shape_target(renderer, system, emitter["shapeTarget"], ctx, transform)
                        if emitter.get("shapeTarget")
                        else None
                    )
                    if not is_layer_active(layer, scene_time):
                        system.rate = 0

        walk(scene["layers"])


__all__ = [
    "advance_simulation",
    "content_size_of",
    "create_body",
    "create_particle_system",
    "create_soft_chain",
    "register_scene_bodies",
    "reset_simulation",
    "update_driven_bodies",
    "world_for_layer",
]
