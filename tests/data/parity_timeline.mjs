// JS 版のタイムライン解決結果を吐く。Python 版の基準。
//   node tests/data/parity_timeline.mjs > tests/data/parity_timeline.json
// JS 版の置き場は人によって違うので、環境変数で渡せるようにしてある。
//   MOVO_JS_ROOT=/path/to/Movo node tests/data/<this file>
const ROOT = `file://${process.env.MOVO_JS_ROOT ?? '/path/to/Movo'}/packages`;
const { buildTimeline, scenesAt, isLayerActive, timeToFrame, frameToTime, resolveRange, allLayers } =
  await import(`${ROOT}/timeline/src/index.js`);

const PROJECTS = {
  sequential: {
    video: { width: 640, height: 360, fps: 24, duration: 12 },
    scenes: [
      { id: 'a', duration: 3, layers: [{ type: 'text', text: 'x' }, { type: 'shape', zIndex: -1 }] },
      { id: 'b', duration: 4, layers: [{ type: 'image', start: 1, duration: 2 }] },
      { id: 'c', start: 9, duration: 2, layers: [] },
    ],
  },
  intrinsic: {
    video: { width: 320, height: 180, fps: 30 },
    scenes: [
      { id: 'anim', layers: [{ type: 'text', animations: [{ delay: 0.5, keyframes: [{ time: 0 }, { time: 2.5 }] }] }] },
      { id: 'ends', layers: [{ type: 'shape', end: 4 }] },
    ],
  },
  stretch: {
    video: { width: 320, height: 180, fps: 12, duration: 10 },
    scenes: [{ id: 'only', layers: [{ type: 'shape' }] }],
  },
  nested: {
    video: { width: 320, height: 180, fps: 30, duration: 6 },
    scenes: [
      {
        id: 'root', duration: 6,
        layers: [
          { id: 'group', type: 'group', start: 1, duration: 4, layers: [
            { type: 'text', zIndex: 5 }, { type: 'shape', start: 0.5, duration: 1 },
          ] },
          { type: 'image', zIndex: 5 },
          { type: 'image', zIndex: 5 },
          { type: 'shape', enabled: false },
        ],
      },
    ],
  },
  disabled: {
    video: { width: 320, height: 180, fps: 25, duration: 8 },
    scenes: [
      { id: 'skip', enabled: false, duration: 3, layers: [] },
      { id: 'keep', duration: 3, layers: [] },
    ],
  },
  empty: { video: { width: 320, height: 180, fps: 30 }, scenes: [] },
};

function summarise(timeline) {
  return {
    fps: timeline.fps,
    duration: timeline.duration,
    frameCount: timeline.frameCount,
    width: timeline.width,
    height: timeline.height,
    scenes: timeline.scenes.map((s) => ({
      id: s.id, index: s.index, start: s.start, end: s.end, duration: s.duration,
      layers: s.layers.map((l) => ({
        id: l.id, type: l.type, order: l.order, zIndex: l.zIndex,
        localStart: l.localStart, localEnd: l.localEnd,
        children: (l.children ?? []).map((c) => ({ id: c.id, localStart: c.localStart, localEnd: c.localEnd, zIndex: c.zIndex })),
      })),
    })),
    allLayerIds: allLayers(timeline).map((l) => l.id),
  };
}

const out = {};
for (const [label, project] of Object.entries(PROJECTS)) {
  const timeline = buildTimeline(project);
  const summary = summarise(timeline);
  summary.scenesAt = [];
  for (let t = 0; t <= timeline.duration + 0.001; t += 0.5) {
    summary.scenesAt.push([Number(t.toFixed(3)), scenesAt(timeline, t).map((s) => s.id)]);
  }
  summary.frames = [0, 0.5, 1.0001, 3.49, 5].map((t) => [t, timeToFrame(timeline, t), frameToTime(timeline, timeToFrame(timeline, t))]);
  summary.ranges = {
    all: resolveRange(timeline, {}),
    partial: resolveRange(timeline, { from: 1, to: 3 }),
    clamped: resolveRange(timeline, { from: -5, to: 1e6 }),
  };
  const first = timeline.scenes[0];
  if (first) summary.ranges.scene = resolveRange(timeline, { scene: first.id });
  if (first && first.layers.length) {
    summary.layerActive = [0, 0.5, 1, 2, 3, 4].map((t) => [t, first.layers.map((l) => isLayerActive(l, t))]);
  }
  out[label] = summary;
}

process.stdout.write(JSON.stringify(out));
