"""JS 版と «数値そのもの» を突き合わせる（変形とマスク）。

基準の作り直しかた:

    node tests/data/parity_deformer.mjs > tests/data/parity_deformer.json

見ているもの:

  - 16 種すべての変形の **全頂点**（x と y）
  - マスク 13 通りの **64x64 の重み場すべて**（膨張・ぼかし・反転・不透明度込み）
  - `sampleField` の双一次補間
  - 値ノイズと fbm そのもの（**乱数の «元» がずれていないか**）

ノイズは 32 ビットのハッシュなので、**1 ビットずれれば値が丸ごと変わります**。
「だいたい合っている」では通らない検査になっていて、そこが狙いです。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from movo.deformer import Mesh, build_mask_field, deformers, sample_field
from movo.deformer._compat import fbm2d, value_noise_1d

GOLDEN = json.loads((Path(__file__).parent / "data" / "parity_deformer.json").read_text("utf-8"))

CASES = {
    "bend": {"amount": 0.6, "axis": "x", "origin": 0.4},
    "bend-y": {"type": "bend", "amount": -0.35, "axis": "y", "origin": 0.55},
    "twist": {"angle": 66, "center": {"x": 0.45, "y": 0.52}, "radius": 0.8, "falloff": 1.4},
    "wave": {"amplitude": 17.5, "frequency": 2.5, "speed": 3, "phase": 0.2, "axis": "both"},
    "skew": {"x": 0.3, "y": -0.2, "originX": 0.4, "originY": 0.6},
    "perspective": {"corners": {
        "topLeft": [0.05, 0.1], "topRight": {"x": 0.95, "y": -0.02},
        "bottomLeft": [-0.1, 1.05], "bottomRight": [1.2, 0.9],
    }},
    "bulge": {"strength": 0.7, "center": {"x": 0.4, "y": 0.6}, "radius": 0.45},
    "pinch": {"strength": 0.4, "centerX": 0.55, "centerY": 0.35, "radius": 0.5},
    "sphereize": {"strength": 0.65, "radius": 0.55},
    "ripple": {"amplitude": 12, "frequency": 3.5, "speed": 1.7, "radius": 0.9},
    "meshWarp": {"columns": 3, "rows": 2, "points": [
        {"column": 1, "row": 1, "offsetX": 12, "offsetY": -8},
        {"column": 3, "row": 0, "x": -6, "y": 14},
        {"col": 0, "row": 2, "offsetX": 5, "offsetY": 5},
    ]},
    "pathDeform": {"path": [[0, 0.2], [0.3, 0.7], [0.6, 0.15], [1, 0.6]], "strength": 0.85},
    "turbulentDisplace": {"amount": 22, "scale": 0.02, "octaves": 4, "evolution": 1.3, "seed": 4242, "mode": "turbulent"},
    "turbulentDisplace-fbm": {"type": "turbulentDisplace", "amountX": 10, "amountY": 30, "scale": 0.013,
                              "octaves": 2, "gain": 0.65, "lacunarity": 2.3, "evolution": -0.7, "seed": 7,
                              "mode": "fbm", "offsetX": 13, "offsetY": -21},
    "turbulentDisplace-ridged": {"type": "turbulentDisplace", "amount": 15, "scale": 0.03, "octaves": 3,
                                 "seed": 99, "mode": "ridged"},
    "melt": {"progress": 0.62, "amount": 240, "columns": 17, "randomness": 0.55, "angle": 100, "seed": 12},
    "handDrawn": {"amount": 6.5, "scale": 0.04, "interval": 2, "roughness": 0.8, "seed": 31},
    "handDrawn-smooth": {"type": "handDrawn", "amount": 3, "scale": 0.09, "interval": 5, "roughness": 0.3, "seed": 5},
    "curveDeform": {"axis": "x", "topCurve": 0.3, "bottomCurve": -0.15, "twist": 0.22},
    "curveDeform-y": {"type": "curveDeform", "axis": "y", "topCurve": -0.4, "bottomCurve": 0.25, "twist": -0.1},
}

MASKS = {
    "rectangle": {"type": "rectangle", "x": 0.4, "y": 0.55, "width": 0.5, "height": 0.6, "rotation": 25},
    "ellipse": {"type": "ellipse", "x": 0.5, "y": 0.5, "width": 0.8, "height": 0.4, "rotation": -15},
    "sector": {"type": "sector", "x": 0.5, "y": 0.5, "startAngle": -120, "endAngle": 60,
               "innerRadius": 0.1, "outerRadius": 0.45},
    "sector-full": {"type": "sector", "startAngle": 0, "endAngle": 360},
    "diagonal": {"type": "diagonal", "angle": -35, "center": 0.45, "width": 0.4},
    "polygon": {"type": "polygon", "points": [[0.1, 0.1], [0.9, 0.2], [0.7, 0.8], [0.2, 0.6]]},
    "path": {"type": "path", "path": [[0.1, 0.2], [0.5, 0.8], [0.9, 0.3]], "thickness": 0.18},
    "path-closed": {"type": "path", "path": [[0.2, 0.2], [0.8, 0.25], [0.5, 0.75]], "closed": True, "thickness": 0.08},
    "ellipse-feather": {"type": "ellipse", "width": 0.5, "height": 0.5, "feather": 0.3},
    "ellipse-expand": {"type": "ellipse", "width": 0.3, "height": 0.3, "expand": 0.08},
    "ellipse-shrink": {"type": "ellipse", "width": 0.6, "height": 0.6, "expand": -0.05},
    "rect-invert-opacity": {"type": "rectangle", "width": 0.4, "height": 0.4, "invert": True, "opacity": 0.65},
    "rect-all": {"type": "rectangle", "width": 0.5, "height": 0.35, "rotation": 10, "expand": 0.04,
                 "feather": 0.2, "invert": True, "opacity": 0.8},
}


def fresh_mesh() -> Mesh:
    return Mesh.grid(320, 180, 8, 640, 360)


@pytest.mark.parametrize("label", list(CASES))
def test_deformer_matches_js(label):
    params = CASES[label]
    mesh = fresh_mesh()
    deformers[params.get("type", label)](mesh, params, {"time": 1.35, "fps": 24})
    got = np.concatenate([mesh.x, mesh.y])
    want = np.array(GOLDEN["deformers"][label], np.float64)
    assert got.shape == want.shape
    # JSON に 9 桁で書いてあるぶんの丸めだけを許します。
    assert np.allclose(got, want, rtol=0, atol=2e-9 + np.abs(want) * 1e-9), (
        f"{label}: 最大差 {np.abs(got - want).max()}"
    )


def test_masked_deform_matches_js():
    """マスク越しの部分適用。**重み 0.999 の場合分け**までここで見ます。"""
    mesh = fresh_mesh()
    field = build_mask_field(
        {"type": "ellipse", "x": 0.45, "y": 0.5, "width": 0.7, "height": 0.9, "rotation": 20, "feather": 0.15},
        64, 64, {},
    )
    deformers["twist"](mesh, {"angle": 90, "radius": 1}, {"maskField": field})
    got = np.concatenate([mesh.x, mesh.y])
    want = np.array(GOLDEN["deformers"]["twist-masked"], np.float64)
    assert np.allclose(got, want, rtol=0, atol=1e-6), f"最大差 {np.abs(got - want).max()}"


@pytest.mark.parametrize("label", list(MASKS))
def test_mask_field_matches_js(label):
    field = build_mask_field(MASKS[label], 64, 64, {})
    want = np.array(GOLDEN["masks"][label], np.float64)
    assert field is not None
    got = np.asarray(field, np.float64)
    assert got.shape == want.shape
    # 場は float32 で持つので、JS 版（float32）と同じ精度で比べます。
    assert np.allclose(got, want, rtol=0, atol=1e-6), f"{label}: 最大差 {np.abs(got - want).max()}"


def test_sample_field_matches_js():
    field = build_mask_field({"type": "ellipse", "width": 0.6, "height": 0.4, "feather": 0.2}, 64, 64, {})
    u = np.arange(41, dtype=np.float64) / 40
    v = 1 - u * 0.7
    got = sample_field(field, 64, 64, u, v)
    want = np.array(GOLDEN["sampleField"], np.float64)
    assert np.allclose(got, want, rtol=0, atol=1e-6)


def test_fbm2d_matches_js():
    """**32 ビットのハッシュ**なので、1 ビットずれれば値が丸ごと変わります。"""
    got = []
    for i in range(60):
        x = (i % 10) * 1.37 - 6
        y = (i // 10) * 2.11 - 4
        for kind in ("fbm", "turbulent", "ridged"):
            got.append(float(fbm2d(x, y, {"seed": 1234, "z": 0.75, "octaves": 4,
                                          "lacunarity": 2.1, "gain": 0.55, "type": kind})))
    assert np.allclose(got, GOLDEN["noise"]["fbm2D"], rtol=0, atol=1e-12)


def test_value_noise_1d_matches_js():
    got = [float(value_noise_1d(i * 0.37, 77)) for i in range(-20, 20)]
    assert np.allclose(got, GOLDEN["noise"]["valueNoise1D"], rtol=0, atol=1e-12)
