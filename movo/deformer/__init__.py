"""movo-deformer — 変形の並び（JS 版 packages/deformer/src/index.js の移植）。

変形は **書かれた順に厳密に**適用します（仕様 9 章）。幾何変形は 1 枚の
メッシュに溜め、画素エフェクトが出てきたところで一度メッシュを焼き、
エフェクトを掛けてから新しいメッシュを張り直します。これで
`bend → wave → blur → colour → draw` が説明どおりに動きつつ、
**要らない中間ラスタライズを避けられます**。
"""

from __future__ import annotations

from ._compat import js_round, warn
from .deformers import apply_deform, deformers, has_deformer, list_deformers
from .mask import MASK_TYPES, build_mask_field, point_in_polygon, sample_field
from .mesh import Mesh, bake_mesh

MASK_RESOLUTION = 64


def apply_modifiers(bitmap, modifiers, ctx: dict | None = None) -> dict:
    """レイヤーの中身に変形とエフェクトを順に掛ける。

    :param bitmap: レイヤーの中身
    :param modifiers: 解決済みの変形／エフェクトの並び
    :param ctx: ``time`` / ``seed`` / ``assets`` / ``meshResolution`` /
        ``renderScale`` / ``plugins`` / ``layerAlpha`` / ``boxWidth`` / ``boxHeight``
    :returns: ``{"bitmap", "mesh", "offsetX", "offsetY", "boxWidth", "boxHeight"}``
    """
    ctx = ctx or {}
    resolution = max(2, js_round(ctx.get("meshResolution", 20)))
    scale = max(1, ctx.get("renderScale", 1))
    current = bitmap
    offset_x = 0
    offset_y = 0
    # メッシュの位置は «論理画素»（拡大前）、UV は元画像の画素です。
    # こうしておくと、超解像で描いた素材が余分な解像度を保ったまま変形されます。
    box_width = ctx.get("boxWidth") or current.width
    box_height = ctx.get("boxHeight") or current.height
    mesh = Mesh.grid(box_width, box_height, resolution, current.width, current.height)

    plugins = ctx.get("plugins") or {}
    effects = _effects_module()

    for modifier in (modifiers or []):
        if not modifier or modifier.get("enabled") is False or not modifier.get("type"):
            continue
        kind = modifier["type"]
        plugin_deformer = plugins["deformer"](kind) if plugins.get("deformer") else None
        if has_deformer(kind) or plugin_deformer:
            mask_field = None
            if modifier.get("mask"):
                mask_field = build_mask_field(
                    modifier["mask"], MASK_RESOLUTION, MASK_RESOLUTION,
                    {"assets": ctx.get("assets"), "selfBitmap": current, "layerAlpha": ctx.get("layerAlpha")},
                )
            deform_ctx = dict(ctx)
            deform_ctx["maskField"] = mask_field
            deform_ctx["sourceBitmap"] = current
            try:
                (plugin_deformer or deformers[kind])(mesh, modifier, deform_ctx)
            except Exception as error:  # 1 つの変形で «動画ごと» 落とさない
                warn(f'deformer "{kind}" failed and was skipped: {error}')
            continue

        plugin_effect = plugins["effect"](kind) if plugins.get("effect") else None
        if (effects is not None and effects.has_effect(kind)) or plugin_effect:
            if not mesh.is_identity():
                from movo.core.bitmap import Bitmap

                baked = bake_mesh(mesh, current, Bitmap, scale)
                current = baked["bitmap"]
                offset_x += baked["offsetX"]
                offset_y += baked["offsetY"]
                box_width = baked["width"]
                box_height = baked["height"]
            try:
                current = (plugin_effect or effects.apply_effect)(current, modifier, ctx)
            except Exception as error:
                warn(f'effect "{kind}" failed and was skipped: {error}')
            # エフェクトは «同じ大きさの別の画像» を返すことがあるので、
            # メッシュはそちらを指すように張り直します。
            mesh = Mesh.grid(box_width, box_height, resolution, current.width, current.height)
            continue

        warn(f'unknown modifier type "{kind}" — skipped')

    return {
        "bitmap": current,
        "mesh": mesh,
        "offsetX": offset_x,
        "offsetY": offset_y,
        "boxWidth": box_width,
        "boxHeight": box_height,
    }


def _effects_module():
    """renderer のエフェクトを «あれば» 使う（まだ移植中の環境でも動くように）。"""
    try:
        from movo.renderer import effects  # type: ignore

        return effects
    except Exception:
        return None


def describe_deformers() -> dict:
    """`movo list deformers` が出す一覧。"""
    return {
        "bend": "Bend the layer around an axis (axis, amount, origin)",
        "twist": "Rotate around a centre with radial falloff (angle, center, radius)",
        "wave": "Sinusoidal ripple (axis, amplitude, frequency, speed, phase)",
        "skew": "Shear horizontally and vertically (x, y)",
        "perspective": "Move the four corners (corners.topLeft ... bottomRight)",
        "bulge": "Expand or contract inside a radius (center, radius, strength)",
        "pinch": "Bulge with an inverted sign (center, radius, strength)",
        "sphereize": "Spherical lens distortion (center, radius, strength)",
        "ripple": "Concentric travelling waves (center, amplitude, frequency, speed)",
        "meshWarp": "Free-form lattice warp (columns, rows, points[])",
        "pathDeform": "Lay the layer along a curve (path[[x,y], ...])",
        "displacement": "Displace using another image (mapAsset, amountX, amountY)",
        "turbulentDisplace": "Displace with fractal noise, no source image (amount, scale, octaves, evolution, mode)",
        "melt": "Drip downwards, column by column (progress, amount, columns, randomness, angle)",
        "handDrawn": "Wobble the outline like a hand drawing (amount, scale, interval, roughness)",
        "curveDeform": "Curve the top and bottom edges independently (axis, topCurve, bottomCurve, twist)",
    }


__all__ = [
    "Mesh", "bake_mesh", "deformers", "has_deformer", "list_deformers",
    "build_mask_field", "sample_field", "point_in_polygon", "apply_deform",
    "apply_modifiers", "describe_deformers", "MASK_TYPES", "MASK_RESOLUTION",
]
