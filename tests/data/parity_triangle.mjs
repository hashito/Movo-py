// テクスチャ付き三角形（3D プレーン）を JS 版で描いて RGBA を JSON で吐きます。
//
//     node tests/data/parity_triangle.mjs > tests/data/parity_triangle.json
//
// 下地のアルファをわざと場所ごとに変えてあります。下地が透明だと合成の式の
// `cb * da * (1 - sa)` の項が丸ごと消えてしまい、**掛ける順の違いが出ません**。
// JS 版の置き場は人によって違うので、環境変数で渡せるようにしてある。
//   MOVO_JS_ROOT=/path/to/Movo node tests/data/<this file>
const ROOT = `file://${process.env.MOVO_JS_ROOT ?? '/path/to/Movo'}/packages`;

const { Bitmap } = await import(`${ROOT}/core/src/bitmap.js`);
const { drawTexturedTriangle } = await import(`${ROOT}/renderer/src/raster.js`);

// tests/test_effects_parity.py の make_image と同じ式（整数なので完全に再現できます）
function makeImage(width, height, tweak = 0) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = (x * 7 + y * 13 + tweak * 31) % 256;
      data[i + 1] = (x * x + y * 3 + 40 + tweak * 17) % 256;
      data[i + 2] = ((x * 5) ^ (y * 11)) % 256;
      const edge = x < 3 || y < 2 || x > width - 4 || y > height - 3;
      data[i + 3] = edge ? 0 : (x + y) % 5 === 0 ? 90 : 255;
    }
  }
  return new Bitmap(width, height, data);
}

// 下地。アルファは 40..255（透明にはしない）。
function makeDest(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = (x * 3 + y * 5 + 17) % 256;
      data[i + 1] = (x * 11 + y * 2 + 90) % 256;
      data[i + 2] = ((x * 13) ^ (y * 7)) % 256;
      data[i + 3] = 40 + ((x * 9 + y * 6) % 216);
    }
  }
  return new Bitmap(width, height, data);
}

const W = 64;
const H = 48;
const texture = makeImage(40, 28);
const dst = makeDest(W, H);

// プレーンの 4 隅（TL, TR, BR, BL）。plane3d.drawPlane と同じ組み方ですが、
// 射影の計算は通さず **三角形 2 枚の合成だけ** を見ます。
const TL = { x: 4.3, y: 3.7, u: 0, v: 0 };
const TR = { x: 57.1, y: 9.2, u: 40, v: 0 };
const BR = { x: 52.6, y: 42.4, u: 40, v: 28 };
const BL = { x: 7.8, y: 38.1, u: 0, v: 28 };

const options = { alpha: 0.9, blend: 'normal', clampEdge: true };
drawTexturedTriangle(dst, texture, TL, TR, BR, options);
drawTexturedTriangle(dst, texture, TL, BR, BL, options);

process.stdout.write(JSON.stringify({ width: W, height: H, data: [...dst.data] }));
