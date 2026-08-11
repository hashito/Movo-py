// JS 版が出す GIF のバイト列を吐く。Python 版の基準。
//   node tests/data/parity_gif.mjs > tests/data/parity_gif.json
// JS 版の置き場は人によって違うので、環境変数で渡せるようにしてある。
//   MOVO_JS_ROOT=/path/to/Movo node tests/data/<this file>
const ROOT = `file://${process.env.MOVO_JS_ROOT ?? '/path/to/Movo'}/packages`;
const { encodeGif, buildPalette } = await import(`${ROOT}/exporters/src/gif.js`);
const { Bitmap } = await import(`${ROOT}/core/src/bitmap.js`);

function frame(width, height, k) {
  const bitmap = new Bitmap(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      bitmap.data[i] = (x * 8 + k * 40) % 256;
      bitmap.data[i + 1] = (y * 10) % 256;
      bitmap.data[i + 2] = (x * y + k * 17) % 256;
      bitmap.data[i + 3] = x < 4 && y < 4 ? 0 : 255;
    }
  }
  return bitmap;
}

const out = {};
{
  const frames = [frame(32, 24, 0), frame(32, 24, 1), frame(32, 24, 2)];
  out.palette64 = buildPalette(frames, 64).map((c) => [...c]);
  out.small = [...encodeGif(frames, { fps: 10, colors: 64 })];
  out.smallOpaque = [...encodeGif(frames, { fps: 12, colors: 16, transparent: false, loop: 3 })];
}
{
  // 大きめ・色数多め。符号長が伸びる経路と、サブブロックの分割を通す。
  const frames = [frame(120, 90, 0), frame(120, 90, 5)];
  out.large = [...encodeGif(frames, { fps: 24, colors: 200 })];
}
process.stdout.write(JSON.stringify(out));
