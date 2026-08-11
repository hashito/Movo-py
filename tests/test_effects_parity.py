"""JS 版と «同じ絵» が出るかを画素で確かめるテスト。

## どうやって確かめたか

1. JS 版（`packages/renderer/src/effects.js` ほか）に、下と同じ入力画像と
   同じパラメーターを与えて描かせる
2. 出てきた RGBA をそのまま SHA-256 に通す
3. Python 版の出力の SHA-256 と突き合わせる

つまり **1 画素でも違えば落ちます。** 差の «大きさ» ではなく «有無» を見るので、
丸めが 1 段ずれただけでも気付けます。

## 一致していないもの（承知のうえ）

下の {@link KNOWN_DRIFT} に挙げた 3 件だけは一致しません。**どれも原因が
分かっていて、エフェクトの式の間違いではありません。**

| 件名 | 差 | 原因 |
| --- | --- | --- |
| `kaleidoscope` `polar_inv` | それぞれ 1 画素で 1 | `np.arctan2` が libm と 1 ULP 違い、ちょうど «.5» の丸めの向きが変わる |
| `hexTile_kaleido` | 2 画素で 41 | 同上（`np.sin`）。こちらは最近傍で拾うので、隣の画素を掴むと色が大きく変わる |

**どれも NumPy と libm の三角関数の差**で、どちらが «正しい» という話では
ありません。1120 画素のうち 4 画素です。エフェクトの式そのものは一致しています。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from movo.core.bitmap import Bitmap
from movo.renderer.effects import effects, has_effect, list_effects

#: 入力画像の大きさ。小さすぎると «縁の扱い» を試せないので、透明な枠を
#: 取れるだけの大きさにしてあります。
WIDTH = 40
HEIGHT = 28

#: エフェクトに渡す ctx。時刻とシードを固定して «同じ絵» を出させます。
CTX = {"seed": 12345, "time": 0.4, "fps": 30}

#: JS 版と一致しないもの（上の表を参照）。**増やすときは理由を必ず書くこと。**
KNOWN_DRIFT = {
    "kaleidoscope": "np.arctan2 が libm と 1 ULP 違う",
    "polar_inv": "np.arctan2 が libm と 1 ULP 違う",
    "hexTile_kaleido": "np.sin が libm と 1 ULP 違い、最近傍で隣の画素を拾う",
}


def make_image(width: int = WIDTH, height: int = HEIGHT, tweak: int = 0) -> Bitmap:
    """試験用の画像。**透明な枠と半透明の点を混ぜてあります。**

    真っ黒や単色だと «透明な縁の色が滲む» 類の間違いが出ません。JS 版の
    同名の関数と 1 画素も違わない画を作ります。
    """
    ys, xs = np.mgrid[0:height, 0:width]
    data = np.zeros((height, width, 4), np.uint8)
    data[..., 0] = (xs * 7 + ys * 13 + tweak * 31) % 256
    data[..., 1] = (xs * xs + ys * 3 + 40 + tweak * 17) % 256
    data[..., 2] = ((xs * 5) ^ (ys * 11)) % 256
    edge = (xs < 3) | (ys < 2) | (xs > width - 4) | (ys > height - 3)
    data[..., 3] = np.where(edge, 0, np.where((xs + ys) % 5 == 0, 90, 255))
    return Bitmap(width, height, data)


#: JS 版に食わせた «名前 → パラメーター»。名前に `_` が付いているものは
#: 同じエフェクトの別の枝（`type` で本当の名前を指しています）。
CASES = [
    ('opacity', {'amount': 0.6}),
    ('blur', {'radius': 3}),
    ('blur_low', {'type': 'blur', 'radius': 5, 'quality': 'low'}),
    ('directionalBlur', {'radius': 6, 'angle': 33}),
    ('sharpen', {'amount': 1.4}),
    ('colorAdjust', {'brightness': 0.08, 'contrast': 0.3, 'saturation': 0.4, 'hue': 25, 'gamma': 1.2}),
    ('colorAdjust_plain', {'type': 'colorAdjust', 'brightness': -0.1, 'contrast': 0.5}),
    ('tint', {'color': '#ff8800', 'amount': 0.45}),
    ('grayscale', {'amount': 0.8}),
    ('invert', {'amount': 0.7}),
    ('threshold', {'level': 0.45}),
    ('pixelate', {'size': 6}),
    ('pixelate_odd', {'type': 'pixelate', 'size': 7}),
    ('glow', {'radius': 4, 'intensity': 1.2}),
    ('glow_color', {'type': 'glow', 'radius': 3, 'color': '#88ffcc'}),
    ('dropShadow', {'offsetX': 4, 'offsetY': 3, 'radius': 3, 'color': 'rgba(0,0,0,0.6)'}),
    ('stroke', {'width': 2, 'color': '#ff0066'}),
    ('chromaKey', {'color': '#00ff00', 'tolerance': 0.4, 'softness': 0.2}),
    ('vignette', {'amount': 0.7, 'radius': 0.4, 'softness': 0.5}),
    ('noise', {'amount': 0.25}),
    ('noise_color', {'type': 'noise', 'amount': 0.2, 'monochrome': False}),
    ('bloom', {'threshold': 0.4, 'radius': 5, 'intensity': 0.9}),
    ('duotone', {'shadow': '#101a3b', 'highlight': '#ffd166', 'amount': 0.9}),
    ('posterize', {'levels': 5}),
    ('emboss', {'amount': 0.8}),
    ('edgeDetect', {'amount': 0.9}),
    ('mirror', {'axis': 'x'}),
    ('mirror_y', {'type': 'mirror', 'axis': 'y', 'flip': True}),
    ('kaleidoscope', {'segments': 5, 'rotation': 20}),
    ('scanlines', {'spacing': 3, 'amount': 0.5}),
    ('chromaticAberration', {'amount': 2.5}),
    ('lensDistortion', {'strength': 0.35}),
    ('roundCorners', {'radius': 7}),
    ('feather', {'size': 5}),
    ('gradientMap', {'stops': [{'offset': 0, 'color': '#001133'}, {'offset': 0.5, 'color': '#ff5522'}, {'offset': 1, 'color': '#ffffcc'}]}),
    ('radialBlur', {'amount': 60, 'samples': 8}),
    ('spinBlur', {'angle': 12, 'samples': 7}),
    ('glitch', {'amount': 0.6, 'blocks': 6, 'colorShift': 4, 'seed': 3}),
    ('rasterScroll', {'amplitude': 5, 'frequency': 3, 'speed': 1.5}),
    ('rasterScroll_rand', {'type': 'rasterScroll', 'amplitude': 4, 'random': 0.8, 'axis': 'vertical'}),
    ('diffusion', {'strength': 60, 'diffusion': 20}),
    ('lightStreak', {'length': 12, 'angle': 30, 'threshold': 0.4, 'intensity': 0.9}),
    ('lensFlare', {'x': 0.4, 'y': 0.35, 'size': 0.5, 'rings': 2, 'streaks': 4}),
    ('rimLight', {'angle': -50, 'width': 3}),
    ('innerGlow', {'size': 5}),
    ('halftone', {'dotSize': 5, 'angle': 30}),
    ('halftone_square', {'type': 'halftone', 'dotSize': 6, 'shape': 'square'}),
    ('mangaize', {'levels': 4, 'edge': 0.8, 'dotSize': 4}),
    ('polar', {'mode': 'rectToPolar'}),
    ('polar_inv', {'type': 'polar', 'mode': 'polarToRect'}),
    ('tile', {'columns': 3, 'rows': 2}),
    ('tile_mirror', {'type': 'tile', 'columns': 2, 'rows': 3, 'mirror': True, 'scrollX': 0.3}),
    ('peripheralBlur', {'radius': 0.4, 'blur': 4, 'light': 0.3}),
    ('letterbox', {'ratio': 1.9}),
    ('gradientOverlay', {'stops': [{'offset': 0, 'color': '#ff7ad9'}, {'offset': 1, 'color': '#3ad6ff'}], 'angle': 60, 'opacity': 0.6}),
    ('gradientOverlay_screen', {'type': 'gradientOverlay', 'blend': 'screen', 'angle': -120, 'opacity': 0.4}),
    ('luminanceKey', {'threshold': 0.4, 'softness': 0.2}),
    ('colorKey', {'color': '#204080', 'tolerance': 0.3}),
    ('pixelSort', {'threshold': 0.35, 'amount': 0.8, 'seed': 9}),
    ('pixelSort_v', {'type': 'pixelSort', 'axis': 'vertical', 'threshold': 0.5}),
    ('reflection', {'position': 0.6, 'opacity': 0.5, 'blur': 2}),
    ('dither', {'levels': 3, 'pattern': 'bayer4'}),
    ('dither_8', {'type': 'dither', 'levels': 4, 'pattern': 'bayer8', 'amount': 0.8}),
    ('dither_fs', {'type': 'dither', 'levels': 3, 'pattern': 'floydSteinberg'}),
    ('dither_palette', {'type': 'dither', 'pattern': 'bayer2', 'palette': ['#000000', '#ff0000', '#00ff88', '#ffffff']}),
    ('misregistration', {'amount': 1}),
    ('misregistration_jitter', {'type': 'misregistration', 'amount': 0.8, 'jitter': 2, 'seed': 4}),
    ('retroFilm', {'seed': 5}),
    ('lightLeak', {'intensity': 0.9, 'angle': 25, 'position': 0.4}),
    ('colorama', {'cycles': 2, 'phase': 0.15}),
    ('colorama_hue', {'type': 'colorama', 'source': 'hue', 'amount': 0.7}),
    ('leaveColor', {'hue': 200, 'tolerance': 40, 'softness': 20}),
    ('monochrome', {'color': '#ffcc88', 'contrast': 0.3}),
    ('bevel', {'size': 5, 'strength': 1}),
    ('bevel_emboss', {'type': 'bevel', 'style': 'emboss', 'size': 4}),
    ('directionalLight', {'intensity': 0.8, 'ambient': 0.3}),
    ('longShadow', {'length': 14, 'angle': 40}),
    ('longShadow_front', {'type': 'longShadow', 'length': 10, 'behind': False}),
    ('graphicPen', {'spacing': 4, 'layers': 2}),
    ('hexTile', {'size': 12, 'mode': 'mirror', 'outline': 2}),
    ('hexTile_kaleido', {'type': 'hexTile', 'size': 10, 'mode': 'kaleido'}),
    ('shatter', {'progress': 0.4, 'pieces': 8, 'seed': 7}),
    ('shatter_grid', {'type': 'shatter', 'pattern': 'grid', 'progress': 0.3, 'pieces': 12}),
    ('shatter_radial', {'type': 'shatter', 'pattern': 'radial', 'progress': 0.25, 'pieces': 9}),
    ('objectSplit', {'columns': 4, 'rows': 3, 'progress': 0.5, 'offset': {'x': 20, 'y': 10, 'rotation': 30}, 'seed': 11}),
    ('slice', {'count': 5, 'angle': 15, 'offset': 8, 'seed': 2}),
    ('curves', {'rgb': [[0, 0], [0.25, 0.18], [0.75, 0.82], [1, 1]], 'blue': [[0, 0.04], [1, 0.94]]}),
    ('colorWheels', {'lift': 0.05, 'gamma': {'r': 1.1, 'b': 0.9}, 'gain': [1.05, 1, 0.95]}),
    ('hslSecondary', {'select': {'hue': [180, 260]}, 'softness': 0.2, 'shift': {'hue': -20, 'sat': 0.4}}),
    ('lut_identity', {'type': 'lut', 'amount': 1, '__identityLut': 5}),
    # slitScan だけは «過去のフレーム» が要るので、ctx に履歴を組み立てて渡します
    ('slitScan', {'axis': 'y', 'span': 0.1, '__frameHistory': 4}),
]

#: JS 版の出力の SHA-256（先頭 32 桁）。**手で書き換えないでください。**
#: 作り直すときは JS 版を走らせて取り直します。
JS_DIGESTS = {
    'opacity': '124216ead38aa97c5f3cfa06ca41355e',
    'blur': 'e70135f1874b7ed48b869c668789a389',
    'blur_low': '19e40b45a4f9cd4a857020333fb8bec9',
    'directionalBlur': '17a76eb7bdf14b0c83cb278ebc91c7c3',
    'sharpen': '4d0b24adeca37098f4cdfd7cd1f2c643',
    'colorAdjust': 'c036a69d2a83c694cd023461e6b3d50e',
    'colorAdjust_plain': 'c60249e145c84320ea8666400859d2b5',
    'tint': 'ec39cc76382bcc744da2ab2ffcfa44ce',
    'grayscale': 'cd494b298443da31f8f553e17c714912',
    'invert': '7f10d8c02341641aa6c9d99e4ab2fc14',
    'threshold': 'fe296a13ba828abc0e9c28b90c946f22',
    'pixelate': 'ba7e792d090343cd3cd535099db80ac7',
    'pixelate_odd': '35c5d824175ac1caacc6912c08c25463',
    'glow': 'a81f833162d07c316db6daab5e91eb9d',
    'glow_color': '5b30f328681c6ba11cc61572fe227ad0',
    'dropShadow': '1cea062237c54a861dd6e69c1705034d',
    'stroke': '340df7601717b3a326b4723de2dee605',
    'chromaKey': 'cc380828667acf1e01b22ee064822e08',
    'vignette': 'fc10c313871de02f3e9d02e1a6743d09',
    'noise': 'dffdc0b6ba479f0518d7c3c2c16ea99e',
    'noise_color': '46ebbafeb6a56b9441277bf716800ad0',
    'bloom': 'fb75fd301c8284f47e600973fee95cd8',
    'duotone': 'f5d6c97146c401f201513c9c01d17992',
    'posterize': '9e721c10b5b72a82cf6a77cebb513ee3',
    'emboss': '3fceb0bbd030d02d0b1e3f5113a8d8a6',
    'edgeDetect': '3efac88101c8262144a64803e78d378e',
    'mirror': 'd793d20d4e8a69c21d26d2ac7cbef477',
    'mirror_y': '07fe685a0d01827e62886f380dee2c15',
    'kaleidoscope': '39e7243b0f099aa3bcb66aaf02423e27',
    'scanlines': '4ca368c5add0cbc1d6616cfeff9c7de8',
    'chromaticAberration': '877a87e6224613d0da802a7ab1afbed3',
    'lensDistortion': '7dcfe0f7e0dd2e64bfc942ce4565dc86',
    'roundCorners': '59fb3509919bbe7b0ac97cff2a5b383b',
    'feather': '45d1be390bb56ad7bd6109706baff76b',
    'gradientMap': 'e27ebce14a701b33542698fb2759e1c1',
    'radialBlur': 'ee758e136a0c1a2a0fe80abfbd25a89a',
    'spinBlur': 'c0926942f14d615b2ce31577e53c2d84',
    'glitch': '7692b057c1de5a8a4e5e8ae1ae24bdc3',
    'rasterScroll': 'bde031e7f0c32b5a853ff998e7ea938a',
    'rasterScroll_rand': '0f94c7bdca923f8cbc2961e4c7dba5c0',
    'diffusion': '47183ceec9656fe1e38c9b510d00ea24',
    'lightStreak': '180908b82c7be2a07170d5c9fca4b4cb',
    'lensFlare': '6410b979237f172fd3703c0463263c39',
    'rimLight': '742e86bb550c0ff4ff202cad44bad3fa',
    'innerGlow': '0a41df2d091e9ee50c512606258f7ae9',
    'halftone': 'a274e9edae18817e4963677d51223105',
    'halftone_square': 'ef0dcb190f6bb760ddef90503d860fc0',
    'mangaize': 'f43d198cf86d4e18a1e1936fc631267e',
    'polar': 'a9085114a96b34321d5863192b1c182b',
    'polar_inv': 'b040c8328ece942bc24e3fd27af941a1',
    'tile': '1e6e55f2da28fcc3060024036e7a0b80',
    'tile_mirror': '055abd244df1002ea259978d137668d0',
    'peripheralBlur': '01a7d4da551f4d839001ff2643f31094',
    'letterbox': '69668aabbf43c1e418e252ab54a99a21',
    'gradientOverlay': '80132cc08ef24bcce7729fbc44ecf79e',
    'gradientOverlay_screen': '0a4a71091de07469c2248324cdd2aa74',
    'luminanceKey': 'f16f79b620961bd627a7dd1782545fff',
    'colorKey': '60385e46680727b6e1ebff68a6e24473',
    'pixelSort': '303887556eab0f61217a0d436f1f439f',
    'pixelSort_v': '4e01fa15ce9c95369b14b48a4d95cefa',
    'reflection': '487c6aaeb77fe5539156df7b77630361',
    'dither': '4f6bed89ac96d25729eaa00be43a9611',
    'dither_8': '3d0883198926cf092e3323a46abfebf2',
    'dither_fs': 'cd742523cd205de053b8fb881b3d5f59',
    'dither_palette': '43d7981295ab521b8ddb879e60c84d9a',
    'misregistration': '0a688cf2385e633d52fac53b3a3aa228',
    'misregistration_jitter': 'acc7e74fca94b550cff9a68dfa2f4654',
    'retroFilm': 'e36a9e9cb317a8d7bfc5250abf3b013e',
    'lightLeak': 'f8ef12e17a69a2266229c6767355620e',
    'colorama': '19d5e618979e3802caff29e539ce6825',
    'colorama_hue': '6431c6d599a203361820e15e5b4fb454',
    'leaveColor': '5896b861df15767870f98abac813da64',
    'monochrome': 'c56fb24d9d9549d5d47cfb58dfcf0dc0',
    'bevel': 'd7a624017ec60b603441c19ccbbb8055',
    'bevel_emboss': '06cdd9be08d6673bb63382068c7636ca',
    'directionalLight': 'edf84df13ebb4e4e4c32baf35d71eda4',
    'longShadow': 'ccf6872a917b17f425aedcec1ded0ff8',
    'longShadow_front': 'e62f2374a762ef59b56e7caf46a4c237',
    'graphicPen': '2abae7154dabc9fa2f431038fde27b24',
    'hexTile': 'c4c45bc3481645a137325e1c29ad6aad',
    'hexTile_kaleido': '1bcf50c967eab9ad4cec805306a4fb1d',
    'shatter': '214eb17dab9f7f122e8a491c38647166',
    'shatter_grid': '214eb17dab9f7f122e8a491c38647166',
    'shatter_radial': '214eb17dab9f7f122e8a491c38647166',
    'objectSplit': '5f35b5c2941e579d6f4b534f79b30645',
    'slice': '49b92aa6e5970badb7d9b409a1d3f8f9',
    'curves': 'd93573099d12bf0d409b190382786e9f',
    'colorWheels': '49ec526111179bc9e8cd3cb0b29c0354',
    'hslSecondary': 'd2120ea03815ac75b9965aae2c8b979e',
    'slitScan': '96779c90dd41ecd884e3e15d7c0ad59d',
    'lut_identity': '59fb3509919bbe7b0ac97cff2a5b383b',
}


def _run(name: str, params: dict) -> Bitmap:
    kind = params.get("type", name)
    args = dict(params)
    ctx = dict(CTX)
    if args.get("__identityLut"):
        from movo.core.lut import identity_lut

        args["lut"] = identity_lut(args["__identityLut"])
    if args.get("__frameHistory"):
        ctx["frameHistory"] = [make_image(tweak=i + 1) for i in range(args["__frameHistory"])]
    return effects[kind](make_image(), args, ctx)


@pytest.mark.parametrize(("name", "params"), CASES, ids=[c[0] for c in CASES])
def test_matches_js(name, params):
    """JS 版と 1 画素も違わないこと。"""
    if name in KNOWN_DRIFT:
        pytest.xfail(KNOWN_DRIFT[name])
    result = _run(name, params)
    digest = hashlib.sha256(result.data.tobytes()).hexdigest()[:32]
    assert digest == JS_DIGESTS[name], f"{name} の絵が JS 版と違います"


@pytest.mark.parametrize(("name", "params"), CASES, ids=[c[0] for c in CASES])
def test_shape_and_dtype(name, params):
    """大きさと型が崩れていないこと（一致しない 4 件もここは通ります）。"""
    result = _run(name, params)
    assert result.data.shape == (HEIGHT, WIDTH, 4)
    assert result.data.dtype == np.uint8


def test_all_69_effects_are_registered():
    """**JS 版と同じ 69 種**が揃っていること（`movo list effects` 相当）。"""
    names = list_effects()
    assert len(names) == 69, f"エフェクトが {len(names)} 種しかありません"
    assert names == sorted(names)
    assert names == JS_EFFECT_NAMES


def test_every_effect_has_a_case():
    """**すべてのエフェクトに絵の比較がある**こと（移植漏れの見張り）。"""
    covered = {params.get("type", name) for name, params in CASES}
    missing = sorted(set(JS_EFFECT_NAMES) - covered)
    assert missing == [], f"絵を比べていないエフェクトがあります: {missing}"


def test_unknown_effect_is_not_registered():
    assert not has_effect("thereIsNoSuchEffect")


#: `node packages/cli/bin/movo.js list effects` が出す一覧そのまま。
JS_EFFECT_NAMES = [
    "bevel", "bloom", "blur", "chromaKey",
    "chromaticAberration", "colorAdjust", "colorKey", "colorWheels",
    "colorama", "curves", "diffusion", "directionalBlur",
    "directionalLight", "dither", "dropShadow", "duotone",
    "edgeDetect", "emboss", "feather", "glitch",
    "glow", "gradientMap", "gradientOverlay", "graphicPen",
    "grayscale", "halftone", "hexTile", "hslSecondary",
    "innerGlow", "invert", "kaleidoscope", "leaveColor",
    "lensDistortion", "lensFlare", "letterbox", "lightLeak",
    "lightStreak", "longShadow", "luminanceKey", "lut",
    "mangaize", "mirror", "misregistration", "monochrome",
    "noise", "objectSplit", "opacity", "peripheralBlur",
    "pixelSort", "pixelate", "polar", "posterize",
    "radialBlur", "rasterScroll", "reflection", "retroFilm",
    "rimLight", "roundCorners", "scanlines", "sharpen",
    "shatter", "slice", "slitScan", "spinBlur",
    "stroke", "threshold", "tile", "tint",
    "vignette",
]
