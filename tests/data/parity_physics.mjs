// JS 版の軌跡を吐く。Python 版と突き合わせるための基準。
// 静的 import の from にはテンプレートリテラルを書けない（構文エラーになる）。
// 他の parity_*.mjs と同じく動的 import にする。
const ROOT = `file://${process.env.MOVO_JS_ROOT ?? '/path/to/Movo'}/packages`;
const { World, Body, SoftChain, ParticleSystem, circleShape, rectangleShape, capsuleShape, polygonShape } =
  await import(`${ROOT}/physics/src/index.js`);

const out = {};

function trace(label, build, steps = 240) {
  const { world, watch } = build();
  const rows = [];
  for (let i = 0; i < steps; i++) {
    world.step(1 / 60);
    if (i % 20 === 0) rows.push(watch().map((v) => Number(v.toFixed(10))));
  }
  out[label] = rows;
}

trace('falling-ball', () => {
  const world = new World({ gravity: { x: 0, y: 1000 }, timeStep: 1 / 120, subSteps: 2, iterations: 10 });
  world.addBody(new Body({ id: 'floor', bodyType: 'static', shape: rectangleShape(1000, 40), x: 500, y: 800 }));
  const ball = new Body({ id: 'ball', bodyType: 'dynamic', shape: circleShape(50), x: 500, y: 200, mass: 1, restitution: 0.4, friction: 0.5, linearDamping: 0.4 });
  world.addBody(ball);
  return { world, watch: () => [ball.position.x, ball.position.y, ball.velocity.y, ball.angle] };
});

trace('box-stack', () => {
  const world = new World({ gravity: { x: 30, y: 900 }, timeStep: 1 / 60, subSteps: 2, iterations: 8 });
  world.addBody(new Body({ id: 'ground', bodyType: 'static', shape: rectangleShape(800, 40), x: 400, y: 600 }));
  const boxes = [];
  for (let i = 0; i < 5; i++) {
    const b = new Body({ id: `box${i}`, bodyType: 'dynamic', shape: rectangleShape(60, 60), x: 400 + i * 3, y: 500 - i * 70, mass: 2, restitution: 0.15, friction: 0.4, angle: i * 0.05 });
    world.addBody(b);
    boxes.push(b);
  }
  return { world, watch: () => boxes.flatMap((b) => [b.position.x, b.position.y, b.angle, b.angularVelocity]) };
});

trace('capsule-mix', () => {
  const world = new World({ gravity: { x: 0, y: 980 }, timeStep: 1 / 60, subSteps: 3, iterations: 6, bounds: { minX: 0, maxX: 400, minY: 0, maxY: 400, restitution: 0.5 } });
  const a = new Body({ id: 'cap', bodyType: 'dynamic', shape: capsuleShape(80, 20), x: 200, y: 100, mass: 1.5, restitution: 0.5, angle: 0.7 });
  const b = new Body({ id: 'tri', bodyType: 'dynamic', shape: polygonShape([[0, 0], [50, 0], [25, 40]]), x: 190, y: 20, mass: 1, restitution: 0.3 });
  const c = new Body({ id: 'ball', bodyType: 'dynamic', shape: circleShape(25), x: 210, y: 250, mass: 3, restitution: 0.6 });
  world.addBody(a); world.addBody(b); world.addBody(c);
  return { world, watch: () => [a.position.x, a.position.y, a.angle, b.position.x, b.position.y, b.angle, c.position.x, c.position.y] };
});

trace('constraints', () => {
  const world = new World({ gravity: { x: 0, y: 1000 }, timeStep: 1 / 120, iterations: 8 });
  const anchor = new Body({ id: 'anchor', bodyType: 'static', shape: circleShape(5), x: 0, y: 0 });
  const w1 = new Body({ id: 'w1', bodyType: 'dynamic', shape: circleShape(10), x: 30, y: 50, mass: 1 });
  const w2 = new Body({ id: 'w2', bodyType: 'dynamic', shape: circleShape(10), x: 60, y: 90, mass: 2 });
  world.addBody(anchor); world.addBody(w1); world.addBody(w2);
  world.addConstraint({ type: 'distance', bodyA: anchor, bodyB: w1, length: 120, stiffness: 0.8 });
  world.addConstraint({ type: 'spring', bodyA: w1, bodyB: w2, restLength: 60, stiffness: 200, damping: 6 });
  world.addConstraint({ type: 'rope', bodyA: anchor, bodyB: w2, length: 250 });
  return { world, watch: () => [w1.position.x, w1.position.y, w2.position.x, w2.position.y, w1.velocity.x, w2.velocity.y] };
});

trace('soft-chain', () => {
  const world = new World({ gravity: { x: 0, y: 980 }, timeStep: 1 / 120 });
  const chain = new SoftChain({ segments: 8, length: 200, origin: { x: 100, y: 100 }, stiffness: 0.75, damping: 0.12 });
  chain.setWind(120, -30);
  world.addSoftBody(chain);
  return { world, watch: () => chain.points.flatMap((p) => [p.x, p.y]) };
});

// 粒は «乱数列が同じか» を見る。位置・寿命・シードまで全部。
{
  const world = new World({ gravity: { x: 20, y: 500 } });
  const system = new ParticleSystem({ seed: 20240801, rate: 45, lifetime: 1.2, lifetimeVariance: 0.4, speed: 180, speedVariance: 0.5, spread: 60, direction: -80, size: 10, sizeVariance: 0.5, spin: 90, drag: 0.4, floorY: 300, bounce: 0.5, width: 40, height: 20, x: 5, y: 7 });
  const rows = [];
  for (let i = 0; i < 90; i++) {
    system.step(1 / 30, world);
    if (i % 10 === 0) {
      rows.push(system.particles.map((p) => [p.x, p.y, p.vx, p.vy, p.life, p.maxLife, p.size, p.rotation, p.spin, p.seed].map((v) => Number(v.toFixed(10)))));
    }
  }
  out['particles'] = rows;
}

process.stdout.write(JSON.stringify(out));
