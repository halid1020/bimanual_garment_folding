#!/usr/bin/env python3
"""Build the narrated LaGarNet presentation video from the deck and the script.

Pipeline (deterministic; not a screen recording, so audio and slides cannot drift):

  script.md  --edge-tts-->  narration/NN.mp3  --ffprobe-->  true per-slide durations
  lagarnet-5min.html  --headless Chrome-->  1920x1080 stills (authors hidden)
  still + narration  --ffmpeg-->  one segment per slide, exactly as long as its narration
  slide 9 additionally composites the four rollout clips into their real tile rectangles,
  measured from a ?rects screenshot rather than guessed
  segments  --concat-->  lagarnet-talk.mp4   (+ .srt and .ass sidecars)
  deck slides past the last script section are held silently at the end

Run:  <venv-with-edge-tts>/bin/python build_video.py [--skip-tts]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "lagarnet-5min.html"
SCRIPT = HERE / "script.md"
NARR_DIR = HERE / "narration"
OUT_MP4 = HERE / "lagarnet-talk.mp4"
OUT_SRT = HERE / "lagarnet-talk.srt"
OUT_ASS = HERE / "lagarnet-talk.ass"

WORK = Path(
    os.environ.get(
        "LAGARNET_WORK",
        "/tmp/claude-1000/-home-halid-Projects-bimanual-garment-folding/"
        "10e06b17-20b3-4bb6-89ac-3390bce54560/scratchpad/video",
    )
)

VOICE = "en-GB-RyanNeural"
RATE = os.environ.get("LAGARNET_RATE", "+0%")   # e.g. "-5%" to slow the narrator down
W, H, FPS = 1920, 1080, 30
GAP = 0.35            # silence appended after each slide's narration
VIDEO_SLIDE = 9       # 1-based index of the rollout wall

# The rollout clips are 832x464 and the tiles are wider than that aspect, so each clip
# is scaled to the tile height and centred, leaving black to either side -- exactly what
# `object-fit: contain` does in the deck, so the video reproduces the slide as seen.
CLIP_W, CLIP_H = 832, 464

SUB_MAX_CHARS = 84             # per cue, wrapped onto at most two ~42-char lines
SUB_MAX_SECS = 6.0

SILENT_HOLD = 8.0              # seconds to hold each deck slide that has no narration

CHROME = next(
    (c for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
     if shutil.which(c)), None)


def run(cmd, **kw):
    """Run a command, raising with the tail of stderr if it fails."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout).splitlines()[-15:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}\n{tail}")
    return p.stdout


# ---------------------------------------------------------------- script -----

def parse_script() -> list[dict]:
    """Read script.md into [{num, title, text}] using the '## NN — title' headings."""
    sections, cur = [], None
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(\d+)\s*[—-]\s*(.+?)\s*$", line)
        if m:
            cur = {"num": int(m.group(1)), "title": m.group(2), "lines": []}
            sections.append(cur)
            continue
        if cur is None:
            continue
        s = line.strip()
        # narration is plain prose only: skip headings, lists, tables, rules, comments
        if not s or s.startswith(("#", "-", "*", "|", ">", "<!--", "---", "`")):
            continue
        cur["lines"].append(s)
    for s in sections:
        s["text"] = re.sub(r"\s+", " ", " ".join(s.pop("lines"))).strip()
    missing = [s["num"] for s in sections if not s["text"]]
    if missing:
        raise SystemExit(f"script.md sections with no narration text: {missing}")
    return sections


# ------------------------------------------------------------------- tts -----

def synthesise(sections, skip=False):
    NARR_DIR.mkdir(exist_ok=True)
    import asyncio
    import edge_tts

    async def one(sec):
        out = NARR_DIR / f"{sec['num']:02d}.mp3"
        if skip and out.exists():
            return
        await edge_tts.Communicate(sec["text"], VOICE, rate=RATE).save(str(out))

    async def all_():
        for sec in sections:          # serial: the endpoint dislikes bursts
            await one(sec)
            print(f"  narrated {sec['num']:02d} — {sec['title']}")

    asyncio.run(all_())


def duration(path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(path)]).strip())


# ---------------------------------------------------------------- stills -----

def shoot(n, out, query=""):
    url = f"file://{DECK}?hide-authors{query}#{n}"
    run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
         "--allow-file-access-from-files", "--virtual-time-budget=4000",
         f"--window-size={W},{H}", f"--screenshot={out}", url])


def tile_rects(rects_png) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the four magenta tiles painted by the ?rects debug view."""
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(rects_png).convert("RGB")).astype(int)
    mask = (a[:, :, 0] > 200) & (a[:, :, 1] < 60) & (a[:, :, 2] > 200)
    if not mask.any():
        raise SystemExit("?rects produced no magenta tiles — is the debug view still wired up?")
    ys, xs = np.where(mask)
    ymid, xmid = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    out = []
    for y0, y1 in ((ys.min(), ymid), (ymid, ys.max() + 1)):
        for x0, x1 in ((xs.min(), xmid), (xmid, xs.max() + 1)):
            sub = mask[y0:y1, x0:x1]
            sy, sx = np.where(sub)
            # even dimensions keep libx264 happy about chroma subsampling
            w = (sx.max() - sx.min() + 1) // 2 * 2
            h = (sy.max() - sy.min() + 1) // 2 * 2
            out.append((int(x0 + sx.min()), int(y0 + sy.min()), int(w), int(h)))
    return out                      # row-major: TL, TR, BL, BR — the deck's DOM order


def tag_overlay(rects_png, rects, out_png):
    """Lift the garment name plates onto a transparent layer, pixel-exact.

    The clips are overlaid across the whole tile and would otherwise bury the
    labels the deck draws on top of each video. The source is the ?rects capture,
    where the videos are hidden and every tile is flat magenta: anything inside a
    tile that is not magenta *is* the plate, which gives an exact alpha mask.

    Do not lift these from the ordinary still. Headless Chrome catches the videos
    already playing, so a colour-threshold crop there grabs a rectangle of that
    one frame and freezes it over the rollout for the whole slide.
    """
    from PIL import Image
    import numpy as np
    im = Image.open(rects_png).convert("RGBA")
    a = np.asarray(im)[:, :, :3].astype(int)
    # a generous magenta test, so antialiased white-to-magenta edge pixels count as
    # background too and the plates do not carry a pink halo. The plate is white with
    # dark text, and both have R ~= G ~= B.
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    magenta = (r - g > 40) & (b - g > 40)

    layer = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
    found = 0
    for (x, y, w, h) in rects:
        keep = ~magenta[y:y + h, x:x + w]
        if keep.sum() < 200:            # no plate drawn in this tile
            continue
        layer[y:y + h, x:x + w, :3] = np.where(keep[..., None], a[y:y + h, x:x + w], 0)
        layer[y:y + h, x:x + w, 3] = np.where(keep, 255, 0)
        found += 1
    Image.fromarray(layer, "RGBA").save(out_png)
    return found


# -------------------------------------------------------------- segments -----

V_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart"]


def deck_slide_count() -> int:
    return len(re.findall(r'<section class="slide', DECK.read_text(encoding="utf-8")))


def silent_segment(png, dur, out):
    """A slide held with no narration — used for deck slides past the script."""
    run(["ffmpeg", "-y", "-loop", "1", "-i", str(png),
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-filter_complex", f"[0:v]scale={W}:{H},fps={FPS},format=yuv420p[v]",
         "-map", "[v]", "-map", "1:a", "-t", f"{dur:.3f}", *V_ENC, str(out)])


def static_segment(png, mp3, dur, out):
    run(["ffmpeg", "-y", "-loop", "1", "-i", str(png), "-i", str(mp3),
         "-filter_complex",
         f"[0:v]scale={W}:{H},fps={FPS},format=yuv420p[v];[1:a]apad=pad_dur={GAP}[a]",
         "-map", "[v]", "-map", "[a]", "-t", f"{dur + GAP:.3f}", *V_ENC, str(out)])


def tile_video_width(tile_h: int) -> int:
    """Width a clip occupies when scaled to the tile height, keeping its aspect."""
    return int(round(tile_h * CLIP_W / CLIP_H)) // 2 * 2


def wall_segment(png, tags_png, clips, rects, mp3, dur, out):
    inputs = ["-loop", "1", "-i", str(png)]
    for c in clips:
        inputs += ["-stream_loop", "-1", "-i", str(c)]
    inputs += ["-loop", "1", "-i", str(tags_png)]
    tag_idx = len(clips) + 1

    fc = [f"[0:v]scale={W}:{H},fps={FPS},format=yuv420p[bg]"]
    for i, (x, y, w, h) in enumerate(rects, start=1):
        # full frame at tile height, no cropping. mpdecimate drops the dwell frames
        # the 10x speed-up left behind (24-29% of every clip), so the rollout keeps
        # moving instead of stalling between actions.
        sw = tile_video_width(h)
        fc.append(f"[{i}:v]mpdecimate,setpts=N/FRAME_RATE/TB,"
                  f"scale={sw}:{h},setsar=1,fps={FPS}[c{i}]")
    prev = "bg"
    for i, (x, y, w, h) in enumerate(rects, start=1):
        nxt = f"o{i}"
        # centre it in the tile, as `object-fit: contain` does in the browser
        ox = x + (w - tile_video_width(h)) // 2
        fc.append(f"[{prev}][c{i}]overlay={ox}:{y}:shortest=0[{nxt}]")
        prev = nxt
    fc.append(f"[{prev}][{tag_idx}:v]overlay=0:0,format=yuv420p[v]")
    fc.append(f"[{tag_idx + 1}:a]apad=pad_dur={GAP}[a]")   # the mp3 is the last input

    run(["ffmpeg", "-y", *inputs, "-i", str(mp3), "-filter_complex", ";".join(fc),
         "-map", "[v]", "-map", "[a]", "-t", f"{dur + GAP:.3f}", *V_ENC, str(out)])


# ------------------------------------------------------------------- srt -----

def chunk(sentence: str) -> list[str]:
    """Split one sentence into cue-sized pieces of at most SUB_MAX_CHARS.

    Prefers clause boundaries (comma, semicolon, colon, dash) so a cue rarely
    breaks mid-phrase, and falls back to word boundaries when a clause is itself
    too long. One sentence per cue is what made the old subtitles unreadable:
    they ran to 256 characters and 13 seconds.
    """
    if len(sentence) <= SUB_MAX_CHARS:
        return [sentence]
    # split after clause punctuation, keeping the punctuation with the left piece
    clauses = [c for c in re.split(r"(?<=[,;:—–-])\s+", sentence) if c]
    out, cur = [], ""
    for c in clauses:
        cand = f"{cur} {c}".strip()
        if cur and len(cand) > SUB_MAX_CHARS:
            out.append(cur)
            cur = c
        else:
            cur = cand
    if cur:
        out.append(cur)
    # any piece still oversized is broken on words
    final = []
    for piece in out:
        if len(piece) <= SUB_MAX_CHARS:
            final.append(piece)
            continue
        cur = ""
        for word in piece.split():
            cand = f"{cur} {word}".strip()
            if cur and len(cand) > SUB_MAX_CHARS:
                final.append(cur)
                cur = word
            else:
                cur = cand
        if cur:
            final.append(cur)
    return final


def wrap2(text: str) -> list[str]:
    """Balance a cue onto at most two lines, breaking near the middle."""
    if len(text) <= SUB_MAX_CHARS // 2:
        return [text]
    words, best, cur = text.split(), None, ""
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(a) - len(b))
        if best is None or score < best[0]:
            best, cur = (score, a, b), None
    return [best[1], best[2]]


def cue_times(sections, durs):
    """[(start, end, text)] — cues sized by chunk(), timed by character share."""
    cues, t = [], 0.0
    for sec, d in zip(sections, durs):
        pieces = []
        for s in re.split(r"(?<=[.?!])\s+", sec["text"]):
            s = s.strip()
            if s:
                pieces.extend(chunk(s))
        total = sum(len(p) for p in pieces) or 1
        start = t
        for p in pieces:
            span = min(d * len(p) / total, SUB_MAX_SECS)
            cues.append((start, start + span, p))
            start += d * len(p) / total
        t += d + GAP
    return cues


def write_subs(sections, durs):
    def srt_ts(t):
        h, r = divmod(t, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")

    def ass_ts(t):
        h, r = divmod(t, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):d}:{int(m):02d}:{s:05.2f}"

    cues = cue_times(sections, durs)

    OUT_SRT.write_text("\n".join(
        f"{k}\n{srt_ts(a)} --> {srt_ts(b)}\n" + "\n".join(wrap2(p)) + "\n"
        for k, (a, b, p) in enumerate(cues, start=1)), encoding="utf-8")

    # .ass carries its own font size, for players that ignore .srt styling
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\nPlayResY: {H}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Talk,Arial,38,&H00FFFFFF,&H00000000,&H80000000,0,0,1,2,1,2,180,180,54,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    OUT_ASS.write_text(head + "".join(
        f"Dialogue: 0,{ass_ts(a)},{ass_ts(b)},Talk,,0,0,0,," + r"\N".join(wrap2(p)) + "\n"
        for a, b, p in cues), encoding="utf-8")
    return len(cues)


# ------------------------------------------------------------------ main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tts", action="store_true",
                    help="reuse narration/*.mp3 instead of re-synthesising")
    args = ap.parse_args()
    if CHROME is None:
        raise SystemExit("no Chrome/Chromium on PATH")
    WORK.mkdir(parents=True, exist_ok=True)

    sections = parse_script()
    print(f"script.md: {len(sections)} sections, "
          f"{sum(len(s['text'].split()) for s in sections)} words")

    print("synthesising narration…")
    synthesise(sections, skip=args.skip_tts)
    durs = [duration(NARR_DIR / f"{s['num']:02d}.mp3") for s in sections]
    for s, d in zip(sections, durs):
        print(f"  {s['num']:02d} {d:6.2f}s  {s['title']}")
    total = sum(durs) + GAP * len(durs)
    print(f"total narration {total:.1f}s = {int(total // 60)}:{total % 60:04.1f}")

    # deck slides past the last script section (the references slide) get no
    # narration and are simply held at the end
    silent = list(range(max(s["num"] for s in sections) + 1, deck_slide_count() + 1))
    if silent:
        print(f"silent slides (no script section): {silent} — {SILENT_HOLD:.0f}s each")

    print("capturing slides…")
    for n in [s["num"] for s in sections] + silent:
        shoot(n, WORK / f"s{n:02d}.png")
    shoot(VIDEO_SLIDE, WORK / "rects.png", query="&rects")
    rects = tile_rects(WORK / "rects.png")
    print(f"  tile rects: {rects}")
    n_tags = tag_overlay(WORK / "rects.png", rects, WORK / "tags.png")
    print(f"  lifted {n_tags}/4 name plates onto the overlay layer")

    clips = [HERE / "assets" / f for f in ("longsleeved-Tshirt_fast.mp4", "trousers_fast.mp4",
                                          "dress_fast.mp4", "skirt_fast.mp4")]
    for c in clips:
        if not c.exists():
            raise SystemExit(f"missing clip {c}")

    print("encoding segments…")
    segs = []
    for s, d in zip(sections, durs):
        out = WORK / f"seg{s['num']:02d}.mp4"
        png = WORK / f"s{s['num']:02d}.png"
        mp3 = NARR_DIR / f"{s['num']:02d}.mp3"
        if s["num"] == VIDEO_SLIDE:
            wall_segment(png, WORK / "tags.png", clips, rects, mp3, d, out)
        else:
            static_segment(png, mp3, d, out)
        segs.append(out)
        print(f"  seg {s['num']:02d} ok")

    for n in silent:
        out = WORK / f"seg{n:02d}.mp4"
        silent_segment(WORK / f"s{n:02d}.png", SILENT_HOLD, out)
        segs.append(out)
        print(f"  seg {n:02d} ok (silent)")

    listing = WORK / "segments.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in segs), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", "-movflags", "+faststart", str(OUT_MP4)])

    n_cues = write_subs(sections, durs)
    (WORK / "durations.json").write_text(json.dumps(
        {s["num"]: d for s, d in zip(sections, durs)}, indent=2))

    print(f"\nwrote {OUT_MP4}  ({OUT_MP4.stat().st_size / 1e6:.1f} MB, "
          f"{duration(OUT_MP4):.1f}s)")
    print(f"wrote {OUT_SRT} and {OUT_ASS}  ({n_cues} cues)")


if __name__ == "__main__":
    main()
