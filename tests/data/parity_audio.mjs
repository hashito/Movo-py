// JS 版の «解析・ラウドネス・ダッキング» の結果を吐く。Python 版の基準。
//   node tests/data/parity_audio.mjs > tests/data/parity_audio.json
//
// 自作音源 10 本（examples/assets/audio/mv-*bpm-*.wav）は JS 版のリポジトリに
// あるので、そこから読みます。ファイル名に «正解の BPM» が入っています。
import fs from 'node:fs';
import path from 'node:path';

// JS 版の置き場は人によって違うので、環境変数で渡せるようにしてある。
//   MOVO_JS_ROOT=/path/to/Movo node tests/data/<this file>
const MOVO = `${process.env.MOVO_JS_ROOT ?? '/path/to/Movo'}`;
const ROOT = `file:///${MOVO}/packages`;
const { analyzeAudio, estimateTempo, onsetEnvelope, detectSections, decodeAudioFile } = await import(`${ROOT}/audio/src/analyze.js`);
const { measureLoudness, measureTruePeak, normalizeLoudness, limitTruePeak, kWeightingStages } = await import(`${ROOT}/audio/src/loudness.js`);
const { resolveDuckSpec, detectorEnvelope, duckGainCurve } = await import(`${ROOT}/audio/src/duck.js`);
const { analyzeEnvelope, mixProjectAudio } = await import(`${ROOT}/audio/src/index.js`);
const { createSilence, encodeWav, decodeWav, resample } = await import(`${ROOT}/core/src/wav.js`);
const { createRandom } = await import(`${ROOT}/core/src/rng.js`);
const { normalizeProject } = await import(`${ROOT}/schema/src/index.js`);

const RATE = 48000;
const out = {};

/* ---- 検査用の音 --------------------------------------------------- */

function sine(seconds, { amplitude = 0.1, frequency = 1000, phase = 0, rate = RATE } = {}) {
  const audio = createSilence(seconds, rate, 2);
  for (let i = 0; i < audio.length; i++) {
    const value = amplitude * Math.sin((2 * Math.PI * frequency * i) / rate + phase);
    audio.channels[0][i] = value;
    audio.channels[1][i] = value;
  }
  return audio;
}

function noise(seconds, { amplitude = 0.1, seed = 7, rate = RATE } = {}) {
  const audio = createSilence(seconds, rate, 2);
  const random = createRandom(seed);
  for (let i = 0; i < audio.length; i++) {
    audio.channels[0][i] = (random() * 2 - 1) * amplitude;
    audio.channels[1][i] = (random() * 2 - 1) * amplitude;
  }
  return audio;
}

function clickTrack({ bpm, seconds, sampleRate = 32000, offset = 0.25 }) {
  const audio = createSilence(seconds, sampleRate, 1);
  const random = createRandom(1234);
  const beat = 60 / bpm;
  const channel = audio.channels[0];
  for (let time = offset; time < seconds; time += beat) {
    const start = Math.round(time * sampleRate);
    const length = Math.round(0.03 * sampleRate);
    for (let i = 0; i < length && start + i < channel.length; i++) {
      channel[start + i] = (random() * 2 - 1) * Math.exp(-(i / length) * 6) * 0.8;
    }
  }
  return audio;
}

function threePartTrack({ sampleRate = 32000 } = {}) {
  const seconds = 36;
  const audio = createSilence(seconds, sampleRate, 1);
  const random = createRandom(99);
  const channel = audio.channels[0];
  for (let i = 0; i < channel.length; i++) {
    const time = i / sampleRate;
    const level = time < 12 ? 0.06 : time < 24 ? 0.8 : 0.06;
    channel[i] = (random() * 2 - 1) * level;
  }
  return audio;
}

function concat(...parts) {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const audio = createSilence(0, parts[0].sampleRate, 2);
  audio.length = length;
  audio.channels = [new Float32Array(length), new Float32Array(length)];
  let cursor = 0;
  for (const part of parts) {
    for (let c = 0; c < 2; c++) audio.channels[c].set(part.channels[c], cursor);
    cursor += part.length;
  }
  return audio;
}

const r = (v, d = 9) => (Number.isFinite(v) ? Number(v.toFixed(d)) : (v === Infinity ? 'inf' : v === -Infinity ? '-inf' : null));

/* ---- WAV の往復 --------------------------------------------------- */
{
  const audio = noise(0.05, { amplitude: 0.6, seed: 3 });
  out.wav = {};
  for (const bits of [16, 24, 32]) {
    const encoded = encodeWav(audio, { bitsPerSample: bits });
    out.wav[`bits${bits}`] = [...encoded.subarray(0, 64)].concat([...encoded.subarray(encoded.length - 32)]);
    const decoded = decodeWav(encoded);
    out.wav[`roundtrip${bits}`] = [...decoded.channels[0].subarray(0, 40)].map((v) => r(v, 10));
  }
  const resampled = resample(audio, 32000);
  out.wav.resample = [...resampled.channels[1].subarray(0, 40)].map((v) => r(v, 10));
  out.wav.resampleLength = resampled.length;
}

/* ---- K 特性 ------------------------------------------------------- */
{
  out.kWeighting = {};
  for (const rate of [32000, 44100, 48000, 96000]) {
    out.kWeighting[rate] = kWeightingStages(rate).map((stage) => ({
      b: stage.b.map((v) => r(v, 15)),
      a: stage.a.map((v) => r(v, 15)),
    }));
  }
}

/* ---- ラウドネス --------------------------------------------------- */
{
  const cases = {
    'sine-23': sine(10, { amplitude: 10 ** (-23 / 20) }),
    'sine-14': sine(6, { amplitude: 10 ** (-14 / 20) }),
    'sine-100hz': sine(6, { amplitude: 0.2, frequency: 100 }),
    'noise-6': noise(6, { amplitude: 0.5 }),
    'noise-26': noise(6, { amplitude: 0.05 }),
    'noise-44100': noise(6, { amplitude: 0.1, rate: 44100 }),
    gated: concat(noise(6, { amplitude: 0.2 }), noise(20, { amplitude: 0.2 * 10 ** (-26 / 20), seed: 11 })),
    'with-silence': concat(noise(6, { amplitude: 0.1 }), createSilence(20, RATE, 2)),
  };
  out.loudness = {};
  for (const [label, audio] of Object.entries(cases)) {
    const measured = measureLoudness(audio);
    out.loudness[label] = {
      lufs: r(measured.lufs, 9),
      blocks: measured.blocks,
      gatedBlocks: measured.gatedBlocks,
      threshold: r(measured.threshold, 9),
      truePeak: r(measureTruePeak(audio), 9),
    };
  }
}

/* ---- トゥルーピークとリミッター ------------------------------------ */
{
  const audio = sine(2, { amplitude: 0.5, frequency: RATE / 4, phase: Math.PI / 4 });
  out.truePeak = { intersample: r(measureTruePeak(audio), 9), low: r(measureTruePeak(sine(2, { amplitude: 0.5, frequency: 100 })), 9) };

  const limited = noise(6, { amplitude: 0.3 });
  const spike = Math.round(3 * RATE);
  for (let i = 0; i < 64; i++) {
    limited.channels[0][spike + i] = 0.99;
    limited.channels[1][spike + i] = 0.99;
  }
  const before = measureLoudness(limited).lufs;
  const result = limitTruePeak(limited, -6);
  out.limiter = {
    reduction: r(result.reduction, 9),
    before: r(before, 9),
    after: r(measureLoudness(limited).lufs, 9),
    truePeak: r(measureTruePeak(limited), 9),
    samples: [...limited.channels[0].subarray(spike - 8, spike + 80)].map((v) => r(v, 9)),
  };
}

/* ---- ラウドネス正規化 --------------------------------------------- */
{
  out.normalize = {};
  const cases = {
    quiet: () => sine(8, { amplitude: 10 ** (-30 / 20) }),
    loud: () => sine(8, { amplitude: 10 ** (-6 / 20) }),
    noise: () => noise(8, { amplitude: 0.05 }),
    'noise-loud': () => noise(8, { amplitude: 0.5 }),
    low: () => sine(8, { amplitude: 0.2, frequency: 100 }),
    gapped: () => concat(noise(6, { amplitude: 0.1 }), createSilence(10, RATE, 2), noise(6, { amplitude: 0.1, seed: 3 })),
    '44100': () => noise(8, { amplitude: 0.1, rate: 44100 }),
  };
  for (const [label, build] of Object.entries(cases)) {
    const audio = build();
    const info = normalizeLoudness(audio, { target: -14, truePeak: -1, standard: 'ebu-r128' });
    out.normalize[label] = {
      input: r(info.input, 9), output: r(info.output, 9), gain: r(info.gain, 9),
      passes: info.passes, limited: info.limited,
      verified: r(measureLoudness(audio).lufs, 9), truePeak: r(measureTruePeak(audio), 9),
      samples: [...audio.channels[0].subarray(1000, 1040)].map((v) => r(v, 9)),
    };
  }
}

/* ---- ダッキング --------------------------------------------------- */
{
  const narration = createSilence(8, RATE, 2);
  for (let i = Math.round(2 * RATE); i < Math.round(4 * RATE); i++) {
    const value = 0.4 * Math.sin((2 * Math.PI * 300 * i) / RATE);
    narration.channels[0][i] = value;
    narration.channels[1][i] = value;
  }
  const spec = resolveDuckSpec({ target: 'track', amount: -12, attack: 0.08, release: 0.4, threshold: -30, hold: 0 });
  const envelope = detectorEnvelope(narration, RATE);
  const curve = duckGainCurve(envelope, RATE, spec);
  const probe = [];
  for (let s = 0; s <= 8; s += 0.05) probe.push(r(curve[Math.min(curve.length - 1, Math.round(s * RATE))], 9));
  out.duck = {
    spec,
    envelope: [...envelope.subarray(Math.round(2 * RATE), Math.round(2 * RATE) + 40)].map((v) => r(v, 9)),
    curve: probe,
  };
}

/* ---- ミックス（ダッキング込み） ------------------------------------ */
{
  const bgm = createSilence(8, RATE, 2);
  for (let i = 0; i < bgm.length; i++) {
    const value = 0.3 * Math.sin((2 * Math.PI * 220 * i) / RATE);
    bgm.channels[0][i] = value;
    bgm.channels[1][i] = value;
  }
  const narration = createSilence(8, RATE, 2);
  for (let i = Math.round(3 * RATE); i < Math.round(5 * RATE); i++) {
    const value = 0.4 * Math.sin((2 * Math.PI * 900 * i) / RATE);
    narration.channels[0][i] = value;
    narration.channels[1][i] = value;
  }
  const assets = { getAudio: (name) => (name === 'track' ? bgm : name === 'narration' ? narration : null) };
  const project = normalizeProject({
    video: { width: 16, height: 16, fps: 30, duration: 8 },
    assets: { track: 'assets/audio/track.wav', narration: 'assets/audio/narration.wav' },
    audio: [
      { asset: 'track', volume: 1 },
      { asset: 'narration', ducks: [{ target: 'track', amount: -12, attack: 0.08, release: 0.4, threshold: -30 }] },
    ],
  });
  const mixed = mixProjectAudio(project, assets, { duration: 8, fps: 30 });
  const probe = [];
  for (let s = 0; s < 8; s += 0.1) {
    const at = Math.round(s * RATE);
    probe.push([r(mixed.audio.channels[0][at], 9), r(mixed.audio.channels[1][at], 9)]);
  }
  out.mix = { tracks: mixed.tracks, ducked: mixed.ducked, loudness: mixed.loudness, probe };
}

/* ---- ミックス（フェード・ループ・パン・トリム） ---------------------- */
{
  const source = noise(3, { amplitude: 0.4, seed: 17, rate: RATE });
  const assets = { getAudio: (name) => (name === 'track' ? source : null) };
  const project = normalizeProject({
    video: { width: 16, height: 16, fps: 30, duration: 10 },
    assets: { track: 'assets/audio/track.wav' },
    audio: [
      { asset: 'track', start: 0.5, offset: 0.25, duration: 6, volume: 0.8, pan: -0.4, fadeIn: 0.7, fadeOut: 1.2, loop: true },
      { id: 'second', asset: 'track', start: 4, volume: 1.4, pan: 0.6 },
    ],
  });
  const mixed = mixProjectAudio(project, assets, { duration: 10, fps: 30 });
  const probe = [];
  for (let i = 0; i < 400; i++) {
    const at = i * 1201;
    probe.push([r(mixed.audio.channels[0][at], 9), r(mixed.audio.channels[1][at], 9)]);
  }
  out.mixShapes = { tracks: mixed.tracks, probe };
}

/* ---- 包絡（audio-reactive の元） ------------------------------------ */
{
  const audio = concat(noise(2, { amplitude: 0.5, seed: 5 }), sine(2, { amplitude: 0.4, frequency: 80 }), sine(2, { amplitude: 0.3, frequency: 6000 }));
  const { levels, bands } = analyzeEnvelope(audio, 30, 180);
  out.envelope = {
    levels: [...levels].map((v) => r(v, 7)),
    bands: bands.map((band) => [...band].map((v) => r(v, 7))),
  };
}

/* ---- オンセットと BPM ---------------------------------------------- */
{
  out.onset = {};
  const audio = clickTrack({ bpm: 120, seconds: 12 });
  const { hop, onset, rms, frames } = onsetEnvelope(audio);
  out.onset.click120 = {
    hop, frames,
    onset: [...onset].map((v) => r(v, 9)),
    rms: [...rms].map((v) => r(v, 9)),
    tempo: (() => { const t = estimateTempo(onset, hop); return { bpm: r(t.bpm, 9), period: r(t.period, 9), firstBeat: r(t.firstBeat, 9), confidence: r(t.confidence, 9) }; })(),
  };

  out.analyze = {};
  for (const [label, audio] of Object.entries({
    click120: clickTrack({ bpm: 120, seconds: 12 }),
    click150: clickTrack({ bpm: 150, seconds: 12 }),
    click128: clickTrack({ bpm: 128, seconds: 8 }),
    flat: clickTrack({ bpm: 120, seconds: 24 }),
    silence: createSilence(4, 32000, 1),
  })) {
    out.analyze[label] = analyzeAudio(audio);
  }

  const three = threePartTrack();
  const env = onsetEnvelope(three);
  out.sections = detectSections(env.rms, env.hop, { barSeconds: 2, duration: three.length / three.sampleRate });
}

/* ---- 自作音源 10 本（BPM の正解つき） ------------------------------- */
{
  const dir = path.join(MOVO, 'examples', 'assets', 'audio');
  out.tracks = {};
  for (const name of fs.readdirSync(dir).filter((n) => /^mv-\d+bpm-.*\.wav$/.test(n)).sort()) {
    const analysis = analyzeAudio(decodeAudioFile(path.join(dir, name)), { maxBeats: 12 });
    out.tracks[name] = {
      truth: Number(name.match(/(\d+)bpm/)[1]),
      bpm: analysis.bpm,
      confidence: analysis.confidence,
      firstBeat: analysis.firstBeat,
      beats: analysis.beats,
      bars: analysis.bars,
      sections: analysis.sections,
      duration: analysis.duration,
    };
  }
}

process.stdout.write(JSON.stringify(out));
