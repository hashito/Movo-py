// JS 版の «変形後の頂点» と «マスクの重み場» を吐く。Python 版の基準。
//   node tests/data/parity_deformer.mjs > tests/data/parity_deformer.json
// JS 版の置き場は人によって違うので、環境変数で渡せるようにしてある。
//   MOVO_JS_ROOT=/path/to/Movo node tests/data/<this file>
const ROOT = `file://${process.env.MOVO_JS_ROOT ?? '/path/to/Movo'}/packages`;
const { Mesh } = await import(`${ROOT}/deformer/src/mesh.js`);
const { deformers } = await import(`${ROOT}/deformer/src/deformers.js`);
const { buildMaskField, sampleField } = await import(`${ROOT}/deformer/src/mask.js`);
const { fbm2D, valueNoise1D } = await import(`${ROOT}/core/src/rng.js`);

const out = { deformers: {}, masks: {}, noise: {}, sampleField: [] };

const CASES = {
  bend: { amount: 0.6, axis: 'x', origin: 0.4 },
  'bend-y': { type: 'bend', amount: -0.35, axis: 'y', origin: 0.55 },
  twist: { angle: 66, center: { x: 0.45, y: 0.52 }, radius: 0.8, falloff: 1.4 },
  wave: { amplitude: 17.5, frequency: 2.5, speed: 3, phase: 0.2, axis: 'both' },
  skew: { x: 0.3, y: -0.2, originX: 0.4, originY: 0.6 },
  perspective: { corners: { topLeft: [0.05, 0.1], topRight: { x: 0.95, y: -0.02 }, bottomLeft: [-0.1, 1.05], bottomRight: [1.2, 0.9] } },
  bulge: { strength: 0.7, center: { x: 0.4, y: 0.6 }, radius: 0.45 },
  pinch: { strength: 0.4, centerX: 0.55, centerY: 0.35, radius: 0.5 },
  sphereize: { strength: 0.65, radius: 0.55 },
  ripple: { amplitude: 12, frequency: 3.5, speed: 1.7, radius: 0.9 },
  meshWarp: { columns: 3, rows: 2, points: [{ column: 1, row: 1, offsetX: 12, offsetY: -8 }, { column: 3, row: 0, x: -6, y: 14 }, { col: 0, row: 2, offsetX: 5, offsetY: 5 }] },
  pathDeform: { path: [[0, 0.2], [0.3, 0.7], [0.6, 0.15], [1, 0.6]], strength: 0.85 },
  turbulentDisplace: { amount: 22, scale: 0.02, octaves: 4, evolution: 1.3, seed: 4242, mode: 'turbulent' },
  'turbulentDisplace-fbm': { type: 'turbulentDisplace', amountX: 10, amountY: 30, scale: 0.013, octaves: 2, gain: 0.65, lacunarity: 2.3, evolution: -0.7, seed: 7, mode: 'fbm', offsetX: 13, offsetY: -21 },
  'turbulentDisplace-ridged': { type: 'turbulentDisplace', amount: 15, scale: 0.03, octaves: 3, seed: 99, mode: 'ridged' },
  melt: { progress: 0.62, amount: 240, columns: 17, randomness: 0.55, angle: 100, seed: 12 },
  handDrawn: { amount: 6.5, scale: 0.04, interval: 2, roughness: 0.8, seed: 31 },
  'handDrawn-smooth': { type: 'handDrawn', amount: 3, scale: 0.09, interval: 5, roughness: 0.3, seed: 5 },
  curveDeform: { axis: 'x', topCurve: 0.3, bottomCurve: -0.15, twist: 0.22 },
  'curveDeform-y': { type: 'curveDeform', axis: 'y', topCurve: -0.4, bottomCurve: 0.25, twist: -0.1 },
};

function freshMesh() {
  return Mesh.grid(320, 180, 8, 640, 360);
}

for (const [label, params] of Object.entries(CASES)) {
  const mesh = freshMesh();
  const type = params.type ?? label;
  deformers[type](mesh, params, { time: 1.35, fps: 24 });
  out.deformers[label] = [...mesh.x].concat([...mesh.y]).map((v) => Number(v.toFixed(9)));
}

// マスクを通した部分適用（apply_deform の重み分岐まで見る）
{
  const mesh = freshMesh();
  const field = buildMaskField({ type: 'ellipse', x: 0.45, y: 0.5, width: 0.7, height: 0.9, rotation: 20, feather: 0.15 }, 64, 64, {});
  deformers.twist(mesh, { angle: 90, radius: 1 }, { maskField: field });
  out.deformers['twist-masked'] = [...mesh.x].concat([...mesh.y]).map((v) => Number(v.toFixed(9)));
}

const MASKS = {
  rectangle: { type: 'rectangle', x: 0.4, y: 0.55, width: 0.5, height: 0.6, rotation: 25 },
  ellipse: { type: 'ellipse', x: 0.5, y: 0.5, width: 0.8, height: 0.4, rotation: -15 },
  sector: { type: 'sector', x: 0.5, y: 0.5, startAngle: -120, endAngle: 60, innerRadius: 0.1, outerRadius: 0.45 },
  'sector-full': { type: 'sector', startAngle: 0, endAngle: 360 },
  diagonal: { type: 'diagonal', angle: -35, center: 0.45, width: 0.4 },
  polygon: { type: 'polygon', points: [[0.1, 0.1], [0.9, 0.2], [0.7, 0.8], [0.2, 0.6]] },
  path: { type: 'path', path: [[0.1, 0.2], [0.5, 0.8], [0.9, 0.3]], thickness: 0.18 },
  'path-closed': { type: 'path', path: [[0.2, 0.2], [0.8, 0.25], [0.5, 0.75]], closed: true, thickness: 0.08 },
  'ellipse-feather': { type: 'ellipse', width: 0.5, height: 0.5, feather: 0.3 },
  'ellipse-expand': { type: 'ellipse', width: 0.3, height: 0.3, expand: 0.08 },
  'ellipse-shrink': { type: 'ellipse', width: 0.6, height: 0.6, expand: -0.05 },
  'rect-invert-opacity': { type: 'rectangle', width: 0.4, height: 0.4, invert: true, opacity: 0.65 },
  'rect-all': { type: 'rectangle', width: 0.5, height: 0.35, rotation: 10, expand: 0.04, feather: 0.2, invert: true, opacity: 0.8 },
};

for (const [label, mask] of Object.entries(MASKS)) {
  const field = buildMaskField(mask, 64, 64, {});
  out.masks[label] = [...field].map((v) => Number(v.toFixed(7)));
}

// sampleField の補間そのもの
{
  const field = buildMaskField({ type: 'ellipse', width: 0.6, height: 0.4, feather: 0.2 }, 64, 64, {});
  for (let i = 0; i <= 40; i++) {
    const u = i / 40;
    const v = 1 - u * 0.7;
    out.sampleField.push(Number(sampleField(field, 64, 64, u, v).toFixed(9)));
  }
}

// ノイズ単体（変形の «元» が合っているか）
{
  const fbm = [];
  for (let i = 0; i < 60; i++) {
    const x = (i % 10) * 1.37 - 6;
    const y = Math.floor(i / 10) * 2.11 - 4;
    for (const type of ['fbm', 'turbulent', 'ridged']) {
      fbm.push(Number(fbm2D(x, y, { seed: 1234, z: 0.75, octaves: 4, lacunarity: 2.1, gain: 0.55, type }).toFixed(12)));
    }
  }
  out.noise.fbm2D = fbm;
  const v1 = [];
  for (let i = -20; i < 20; i++) v1.push(Number(valueNoise1D(i * 0.37, 77).toFixed(12)));
  out.noise.valueNoise1D = v1;
}

process.stdout.write(JSON.stringify(out));
