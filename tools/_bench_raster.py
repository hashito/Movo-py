import time, numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def scanline_fill(px, py, W, H, cov):
    n = px.shape[0]
    ymin = 1e30; ymax = -1e30
    for i in range(n):
        if py[i] < ymin: ymin = py[i]
        if py[i] > ymax: ymax = py[i]
    y0 = max(0, int(ymin)); y1 = min(H - 1, int(ymax) + 1)
    xs = np.empty(n, np.float32)
    for y in range(y0, y1 + 1):
        cy = y + 0.5; m = 0
        for i in range(n):
            j = (i + 1) % n
            ay = py[i]; by = py[j]
            if (ay > cy) != (by > cy):
                xs[m] = (px[j] - px[i]) * (cy - ay) / (by - ay) + px[i]; m += 1
        for a in range(1, m):
            v = xs[a]; b = a - 1
            while b >= 0 and xs[b] > v:
                xs[b + 1] = xs[b]; b -= 1
            xs[b + 1] = v
        for a in range(0, m - 1, 2):
            l = max(0, int(xs[a] + 0.5)); r = min(W, int(xs[a + 1] + 0.5))
            for x in range(l, r): cov[y, x] = 1.0

H, W = 720, 1280
px = np.array([200, 900, 1000, 300], np.float32)
py = np.array([100, 180, 600, 650], np.float32)
cov = np.zeros((H, W), np.float32)
t0 = time.perf_counter(); scanline_fill(px, py, W, H, cov); jit = time.perf_counter() - t0
t = time.perf_counter()
for _ in range(50):
    cov[:] = 0; scanline_fill(px, py, W, H, cov)
ms = (time.perf_counter() - t) / 50 * 1000
print(f'JIT 初回コンパイル: {jit:.2f} 秒（2 回目以降はキャッシュ）')
print(f'Numba 走査線: {ms:.3f} ms/多角形  →  239 レイヤーで {ms*239:.0f} ms/フレーム')
print(f'  NumPy 一括判定 30.36 ms との比: {30.36/ms:.0f} 倍速い')
