"""タイミング付きの歌詞を読む。

歌詞 MV を作る人は、たいてい **すでに時刻付きの歌詞を持っています**
（カラオケアプリの ``.lrc``、字幕の ``.srt`` / ``.vtt``）。それを受け取れないと
行を等分するしかなく、実際の歌は行ごとに長さが違うので必ずずれます。

ここは **読むだけ**です。音声認識は入れません。外部 API に出すと «依存ゼロ» と
«同じ JSON からは同じ動画が出る» の両方が壊れるためです。文字起こしは外で
やって、その結果をこの口から受けます。

**辞書のキーは JS 版のまま** ``text`` / ``at`` / ``for`` / ``syllables`` です。
``for`` は Python の予約語ですが、JSON にそう書かれるので変えられません
（属性ではなく辞書の鍵なので、実害はありません）。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .errors import ErrorCodes, MovoError

LyricLine = dict[str, Any]

_LRC_STAMP_RE = re.compile(r"\[(\d{1,2}:\d{1,2}(?:[.:]\d{1,3})?)\]")
_LRC_SYLLABLE_RE = re.compile(r"<(\d{1,2}:\d{1,2}(?:[.:]\d{1,3})?)>([^<]*)")


def _to_seconds(stamp: str) -> float | None:
    """``mm:ss.xx`` / ``hh:mm:ss,mmm`` / ``ss.xx`` を秒にする。"""
    parts = str(stamp).strip().replace(",", ".").split(":")
    values = []
    for piece in parts:
        try:
            value = float(piece)
        except ValueError:
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        values.append(value)
    # 後ろから 秒・分・時 と読みます（`ss` だけ / `mm:ss` / `hh:mm:ss` のどれでも）
    seconds = 0.0
    scale = 1.0
    for value in reversed(values):
        seconds += value * scale
        scale *= 60
    return seconds


def parse_lrc(text: str) -> list[LyricLine]:
    """LRC を読む。

    拡張 LRC（``[00:12.40]<00:12.40>ね<00:12.62>え``）の音節タイミングも拾います。
    拾えれば «語単位で塗る» カラオケができます。
    """
    lines: list[LyricLine] = []
    for raw in str(text).splitlines():
        # 行頭の [mm:ss.xx] は複数付くことがあります（同じ歌詞を複数箇所で使う書き方）
        stamps = list(_LRC_STAMP_RE.finditer(raw))
        if not stamps:
            continue  # [ar:] などのメタ行は読み飛ばす
        body = raw[stamps[-1].end() :]

        syllables = []
        pieces = list(_LRC_SYLLABLE_RE.finditer(body))
        for piece in pieces:
            at = _to_seconds(piece.group(1))
            if at is not None and piece.group(2):
                syllables.append({"text": piece.group(2), "at": at})
        plain = ("".join(s["text"] for s in syllables) if pieces else body).strip()
        if not plain:
            continue

        for stamp in stamps:
            at = _to_seconds(stamp.group(1))
            if at is None:
                continue
            line: LyricLine = {"text": plain, "at": at}
            if syllables:
                line["syllables"] = syllables
            lines.append(line)
    return _finish(lines)


def parse_subtitles(text: str) -> list[LyricLine]:
    """SRT / WebVTT を読む。

    どちらも «時刻の行 + 本文» の繰り返しなので、まとめて扱います。
    区切りが ``-->`` である点だけ見ています。
    """
    lines: list[LyricLine] = []
    body_text = re.sub(r"^WEBVTT.*$", "", str(text), count=1, flags=re.MULTILINE)
    for block in re.split(r"\r?\n\s*\r?\n", body_text):
        rows = [row for row in block.splitlines() if row.strip()]
        time_index = next((i for i, row in enumerate(rows) if "-->" in row), -1)
        if time_index == -1:
            continue
        halves = rows[time_index].split("-->")
        if len(halves) < 2:
            continue
        start = halves[0].strip().split()[0] if halves[0].strip() else ""
        end_text = halves[1].strip().split()[0] if halves[1].strip() else ""
        at = _to_seconds(start)
        end = _to_seconds(end_text)
        # 字幕番号や位置指定の行を落とし、本文だけを繋ぐ
        body = "\n".join(rows[time_index + 1 :]).strip()
        if at is None or not body:
            continue
        if end is not None and end > at:
            lines.append({"text": body, "at": at, "for": end - at})
        else:
            lines.append({"text": body, "at": at})
    return _finish(lines)


def _finish(lines: list[LyricLine]) -> list[LyricLine]:
    """時刻順に並べ、``for`` が無い行は «次の行の頭まで» にする。

    **最後の 1 行だけは ``for`` を付けません。** «次» が無いので、呼ぶ側が
    「シーンの終わりまで」と解釈できるようにしておきます。
    """
    ordered = sorted(lines, key=lambda line: line["at"])  # sorted は安定なので同時刻の順は保たれる
    out: list[LyricLine] = []
    for index, line in enumerate(ordered):
        if "for" in line:
            out.append(line)
            continue
        nxt = ordered[index + 1] if index + 1 < len(ordered) else None
        copy = dict(line)
        if nxt is not None:
            copy["for"] = max(0.0, nxt["at"] - line["at"])
        out.append(copy)
    return out


def detect_lyrics_format(text: str) -> str:
    """中身を見て形式を当てる。``'lrc'`` / ``'subtitle'`` / ``'unknown'``。

    **拡張子は当てになりません**（``.txt`` に LRC が入っていることがよくあります）。
    """
    if "-->" in text:
        return "subtitle"
    if re.search(r"\[\d{1,2}:\d{1,2}", text):
        return "lrc"
    return "unknown"


def parse_lyrics(text: str, *, file: str | None = None) -> list[LyricLine]:
    """歌詞を読む。形式は中身から当てます。

    JSON（``[{"text": "...", "at": 1.5, "for": 0.8}]``）もそのまま受けます。
    **時刻の無いただの行の羅列は受けません** — 等分したいだけなら文字列の
    配列を渡す既存の道があり、そちらと紛れるほうが困るためです。
    """
    trimmed = str(text).strip()
    if trimmed.startswith("[") and re.match(r"^\[\s*\{", trimmed):
        data = json.loads(trimmed)
        rows = []
        for row in data:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            try:
                at = float(row["at"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isnan(at) or math.isinf(at):
                continue
            entry: LyricLine = {"text": row["text"], "at": at}
            try:
                span = float(row["for"])
                if not (math.isnan(span) or math.isinf(span)):
                    entry["for"] = span
            except (KeyError, TypeError, ValueError):
                pass
            if isinstance(row.get("syllables"), list):
                entry["syllables"] = row["syllables"]
            rows.append(entry)
        return _finish(rows)

    fmt = detect_lyrics_format(trimmed)
    if fmt == "subtitle":
        return parse_subtitles(trimmed)
    if fmt == "lrc":
        return parse_lrc(trimmed)
    raise MovoError(
        ErrorCodes.MOVO_ASSET_DECODE_FAILED,
        "歌詞の形式が分かりません（LRC / SRT / WebVTT / JSON）",
        file=file,
        hint="時刻の無い «ただの行» を等分したいときは、文字列の配列をそのまま渡してください",
    )


def slice_lyrics(
    lines: list[LyricLine],
    from_seconds: float = 0.0,
    to_seconds: float = float("inf"),
    *,
    overlap: bool = False,
    min_span: float = 0.25,
) -> list[LyricLine]:
    """指定した範囲の行だけ取り出し、範囲の頭を 0 秒とした時刻に直す。

    シーンに流し込むときに使います。シーンは «そのシーンの中の時刻» で動くので、
    曲頭からの時刻のままでは合いません。

    既定は «その範囲で «始まる» 行» だけ。素直な切り出しです。

    ``overlap=True`` にすると **範囲に «掛かっている» 行も拾います。**
    シーンを細かく割ったとき、前のシーンで歌い始めて次に続く行が丸ごと
    消えてしまうためです（大サビを 6.5 秒ごとに割ったら、4.86 秒間隔の歌詞が
    境目をまたいで何行も消えました）。歌は絵のカットとは無関係に続くので、
    **またぐのが自然**です。

    ``min_span`` は «またいで入ってきた行が一瞬だけ残る» のを防ぎます。
    0.05 秒しか見えない歌詞は読めないうえ、フェードの尺が取れずにスキル側が
    落ちます。既定で 0.25 秒未満の断片は捨てます（``min_span=0`` で従来どおり）。
    """
    span_limit = min_span if isinstance(min_span, (int, float)) and not math.isnan(min_span) else 0.25

    def keeps(line: LyricLine) -> bool:
        if not overlap:
            return line["at"] >= from_seconds - 1e-6 and line["at"] < to_seconds - 1e-6
        if "for" not in line:
            # 尺の無い最後の行は開始だけで判定
            return line["at"] >= from_seconds - 1e-6 and line["at"] < to_seconds - 1e-6
        end = line["at"] + line["for"]
        # 範囲と少しでも重なっていれば拾う
        return end > from_seconds + 1e-6 and line["at"] < to_seconds - 1e-6

    shifted: list[LyricLine] = []
    for line in lines:
        if not keeps(line):
            continue
        at = line["at"] - from_seconds
        # またいで入ってきた行は «途中から» 始まります。負の開始は扱いに困るので
        # 0 に丸め、そのぶん尺を削ります（見えている時間は変わりません）。
        clamped = max(0.0, at)
        copy = dict(line)
        copy["at"] = clamped
        if "for" in line:
            copy["for"] = max(0.0, line["for"] - (clamped - at))
        if "syllables" in line:
            copy["syllables"] = [{**s, "at": s["at"] - from_seconds} for s in line["syllables"]]
        shifted.append(copy)

    # 範囲の «端» で切れて短くなりすぎた断片を落とす
    out: list[LyricLine] = []
    for line in shifted:
        if not overlap or "for" not in line:
            out.append(line)
            continue
        visible = min(line["for"], to_seconds - from_seconds - line["at"])
        if visible >= span_limit:
            out.append(line)
    return out
