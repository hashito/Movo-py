"""`movo preview` — 小さなローカルサーバーと、こすって確かめられる再生画面。

フレームはプレビュー品質でその都度描くので、書き出しを待つより速く試せます。

**127.0.0.1 にだけ待ち受けます。** 0.0.0.0 にすると、同じ Wi-Fi の他人が
作りかけの映像を見られてしまいます。
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import bridge
from ..console import logger, style
from ..errors import ErrorCodes, MovoError
from ..pipeline import create_session


def preview_command(positional: list[str], options: dict[str, Any]) -> dict[str, Any]:
    file = (positional[0] if positional else None) or "movo.json"
    port = int(options.get("port") or 7777)
    quality = options.get("quality") or "preview"

    state: dict[str, Any] = {
        "session": create_session(file, {"quality": quality, "generate_assets": False, "strict_plugins": False}),
        "lock": threading.Lock(),
    }

    def reload() -> None:
        try:
            state["session"] = create_session(
                file, {"quality": quality, "generate_assets": False, "strict_plugins": False}
            )
            logger.info(f'  {style.green("reloaded")} {Path(file).name}')
        except Exception as error:  # noqa: BLE001 - 直している最中なので落とさない
            logger.error(str(error))

    handler = partial(_PreviewHandler, state, Path(file).name, reload)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as error:
        raise MovoError(
            ErrorCodes.MOVO_CLI_USAGE, f"ポート {port} は既に使われています", hint="--port <番号> で別の番号を指定してください"
        ) from error

    address = f"http://127.0.0.1:{port}/"
    logger.success(f"プレビューサーバーを起動しました: {address}")
    logger.info(f'  品質 {quality} / {state["session"]["timeline"]["frameCount"]} フレーム / Ctrl+C で終了')
    if options.get("open"):
        _open_browser(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"address": address}


class _PreviewHandler(BaseHTTPRequestHandler):
    def __init__(self, state, file_name, reload, *args, **kwargs):
        self._state = state
        self._file_name = file_name
        self._reload = reload
        super().__init__(*args, **kwargs)

    def log_message(self, *_args) -> None:  # noqa: D102 - 既定のアクセスログは要らない
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server の決まり
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self._send(200, "text/html; charset=utf-8", _player_html(self._state["session"], self._file_name))
                return
            if url.path == "/api/info":
                session = self._state["session"]
                timeline = session["timeline"]
                info = {
                    "name": (session["project"].get("project") or {}).get("name"),
                    "width": timeline["width"],
                    "height": timeline["height"],
                    "fps": timeline["fps"],
                    "duration": timeline["duration"],
                    "frameCount": timeline["frameCount"],
                    "quality": session["project"]["render"]["quality"],
                    "scenes": [
                        {
                            "id": scene["id"],
                            "start": scene["start"],
                            "duration": scene["duration"],
                            "layers": len(scene.get("layers") or []),
                        }
                        for scene in timeline["scenes"]
                    ],
                }
                self._send(200, "application/json; charset=utf-8", json.dumps(info, ensure_ascii=False))
                return
            if url.path == "/api/reload":
                self._reload()
                self._send(200, "application/json", '{"ok":true}')
                return
            if url.path == "/frame":
                session = self._state["session"]
                requested = int(float(parse_qs(url.query).get("f", ["0"])[0]))
                frame = max(0, min(session["timeline"]["frameCount"] - 1, requested))
                # レンダラーは 1 つしかないので、同時に叩かれないよう守ります。
                with self._state["lock"]:
                    bitmap = session["renderer"].render_frame(frame)
                    png = bytes(bridge.encode_png(bitmap, level=3))
                self._send_bytes(200, "image/png", png)
                return
            self._send(404, "text/plain", "not found")
        except Exception as error:  # noqa: BLE001
            logger.error(str(error))
            self._send(500, "text/plain; charset=utf-8", str(error))

    def _send(self, status: int, content_type: str, body: str) -> None:
        self._send_bytes(status, content_type, body.encode("utf-8"))

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _open_browser(url: str) -> None:
    try:
        if sys.platform == "win32":
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
    except OSError:
        logger.verbose("ブラウザを自動で開けませんでした")


def _player_html(session, file_name: str) -> str:
    timeline = session["timeline"]
    scenes = "".join(
        f'<span>{html.escape(str(scene["id"]))} {scene["start"]:.1f}–{scene["start"] + scene["duration"]:.1f}s</span>'
        for scene in timeline["scenes"]
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Movo preview — {html.escape(file_name)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; font-family: ui-sans-serif, system-ui, "Segoe UI", "Noto Sans JP", sans-serif;
         background: #14161f; color: #e7e9f3; }}
  header {{ padding: 12px 20px; border-bottom: 1px solid #2a2e3f; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  header h1 {{ font-size: 15px; margin: 0; font-weight: 600; }}
  header .meta {{ font-size: 12px; color: #98a0bd; }}
  main {{ padding: 20px; display: flex; flex-direction: column; gap: 14px; align-items: center; }}
  .stage {{ background: #0b0d14; border: 1px solid #2a2e3f; border-radius: 8px;
           width: min(100%, 1280px); aspect-ratio: {timeline["width"]} / {timeline["height"]}; }}
  .stage img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
  .controls {{ display: flex; gap: 12px; align-items: center; width: min(100%, 1280px); flex-wrap: wrap; }}
  input[type=range] {{ flex: 1; min-width: 240px; accent-color: #6c8cff; }}
  button {{ background: #262b3d; color: #e7e9f3; border: 1px solid #3a4159; border-radius: 6px;
           padding: 6px 14px; font-size: 13px; cursor: pointer; }}
  .time {{ font-variant-numeric: tabular-nums; font-size: 13px; color: #98a0bd; min-width: 132px; }}
  .scenes {{ display: flex; gap: 8px; flex-wrap: wrap; font-size: 12px; color: #98a0bd; }}
  .scenes span {{ border: 1px solid #2a2e3f; border-radius: 999px; padding: 2px 10px; }}
  .status {{ font-size: 12px; color: #7d86a6; min-height: 16px; }}
</style>
</head>
<body>
<header>
  <h1>Movo preview</h1>
  <span class="meta">{html.escape(file_name)} — {timeline["width"]}×{timeline["height"]} @ {timeline["fps"]}fps — {timeline["duration"]:.2f}s</span>
</header>
<main>
  <div class="stage"><img id="frame" alt="frame"></div>
  <div class="controls">
    <button id="play">▶ 再生</button>
    <input id="scrub" type="range" min="0" max="{timeline["frameCount"] - 1}" value="0" step="1">
    <span class="time" id="time">0.000s / frame 0</span>
    <button id="reload">再読み込み</button>
  </div>
  <div class="scenes">{scenes}</div>
  <div class="status" id="status"></div>
</main>
<script>
const fps = {timeline["fps"]};
const frameCount = {timeline["frameCount"]};
const img = document.getElementById('frame');
const scrub = document.getElementById('scrub');
const timeLabel = document.getElementById('time');
const playButton = document.getElementById('play');
const status = document.getElementById('status');
let playing = false;
let current = 0;
let busy = false;

function label(frame) {{
  timeLabel.textContent = (frame / fps).toFixed(3) + 's / frame ' + frame;
}}

async function show(frame) {{
  if (busy) return;
  busy = true;
  const started = performance.now();
  try {{
    const response = await fetch('/frame?f=' + frame, {{ cache: 'no-store' }});
    if (!response.ok) throw new Error(await response.text());
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const previous = img.src;
    img.src = url;
    if (previous.startsWith('blob:')) URL.revokeObjectURL(previous);
    status.textContent = 'render ' + Math.round(performance.now() - started) + 'ms';
  }} catch (err) {{
    status.textContent = String(err.message || err);
  }} finally {{
    busy = false;
  }}
}}

scrub.addEventListener('input', () => {{
  playing = false;
  playButton.textContent = '▶ 再生';
  current = Number(scrub.value);
  label(current);
  show(current);
}});

playButton.addEventListener('click', () => {{
  playing = !playing;
  playButton.textContent = playing ? '⏸ 停止' : '▶ 再生';
  if (playing) tick();
}});

document.getElementById('reload').addEventListener('click', async () => {{
  await fetch('/api/reload');
  show(current);
}});

async function tick() {{
  while (playing) {{
    current = (current + 1) % frameCount;
    scrub.value = String(current);
    label(current);
    await show(current);
    await new Promise((r) => setTimeout(r, Math.max(0, 1000 / fps - 8)));
  }}
}}

label(0);
show(0);
</script>
</body>
</html>"""
