"""Explanatory text for the interface.

Kept out of the widget code so the wording can be tested and revised without
touching Qt, and reused by any other front end.

Every control must have an entry. `test_help_text.py` enforces that, so adding
a control without explaining it fails the suite.

House style: lead with what it does in one line, then why it matters, then
concrete values. No jargon, no lecturing.
"""

from __future__ import annotations

# Keys match the control names used in `window.py`.
HELP: dict[str, str] = {
    "source": (
        "The file to enlarge.\n\n"
        "Video: mp4, mkv, mov, avi, webm\n"
        "Images: png, jpg, webp — transparency kept\n\n"
        "Drop it here, or click Browse."
    ),
    "analysis": (
        "What was detected in your file.\n\n"
        "Scan says which correction is applied:\n"
        "progressive — none needed\n"
        "telecined — film on video; original frames recovered\n"
        "interlaced — video; deinterlaced\n\n"
        "Blockiness over 2: raise Deblock.\n"
        "Grain over 8: a grainy film print."
    ),
    "model": (
        "The AI model doing the enlarging.\n\n"
        "Affects speed more than everything else here combined — the slowest "
        "is roughly ten times the fastest.\n\n"
        "Hover any entry for what it suits. Start with 2xParimgCompact."
    ),
    "rescan": (
        "Re-read the models folder after adding a file.\n\n"
        "Saves restarting the app."
    ),
    "no_restore": (
        "Enlarge only — no cleanup, no texture work.\n\n"
        "About 35% faster. Useful for seeing what the model does alone. "
        "Leave it off for a real render, or skin tends to look plastic."
    ),
    "degrain": (
        "Removes noise and grain before enlarging.\n\n"
        "Some is needed, or the model magnifies grain into digital noise. But "
        "grain and skin texture are the same size of detail, so removing one "
        "removes the other. This is the usual cause of waxy faces.\n\n"
        "Waxy skin? Lower this first.\n\n"
        "0.10 grainy film · 0.25 default · 0.50 noisy video"
    ),
    "detail": (
        "Lays your source's own fine detail back over the result.\n\n"
        "The only control that restores real texture rather than texture the "
        "model invented. It cannot add pores that were never there.\n\n"
        "Skin too smooth? Raise this.\n\n"
        "0.25 default · 0.40 faces · 0.60 soft sources"
    ),
    "regrain": (
        "Adds fine grain after enlarging.\n\n"
        "Models strip the faint texture of skin as if it were noise. Putting "
        "grain back is the strongest fix for plastic-looking faces.\n\n"
        "Strongest in mid-tones, fading in highlights and shadows, and it "
        "moves between frames rather than sitting still.\n\n"
        "0.60 default · 0.80 old film · 0.20 modern digital"
    ),
    "deblock": (
        "Removes compression blocks and edge fuzz.\n\n"
        "Off by default: it softens slightly, and clean files do not need it. "
        "Raise it when Blockiness above reads high.\n\n"
        "0.00 Blu-ray · 0.30 YouTube · 0.50 VCD and low-bitrate rips"
    ),
    "fps_mode": (
        "Smoother motion, by adding frames between the existing ones.\n\n"
        "Your original frames are kept unchanged.\n\n"
        "Off · Target FPS names the result · Multiplier scales the source\n\n"
        "Roughly doubles render time."
    ),
    "fps_target": (
        "Frame rate out.\n\n"
        "60 usual choice · 48 keeps a cinema feel · 50 for PAL · "
        "120 for high-refresh screens or slow motion\n\n"
        "Uneven ratios work: 24 to 60 is 2.5 times."
    ),
    "fps_multiplier": (
        "Scale the source rate instead of naming a target.\n\n"
        "2 doubles it (24 becomes 48) · 3 · 4 for slow motion\n\n"
        "Use when the source rate does not matter to you."
    ),
    "scene_threshold": (
        "How readily a cut between shots is detected.\n\n"
        "Matters for fast-cut footage. Adding a frame across a cut smears two "
        "unrelated shots together; at a cut, a real frame is repeated instead.\n\n"
        "0.30 default · 0.20 fast cutting, if you see smearing · 0.45 long takes"
    ),
    "output": (
        "Where the result is written.\n\n"
        "mkv carries 10-bit video, audio and subtitles cleanly. For images, "
        "the extension sets the format."
    ),
    "segment_frames": (
        "How much work an interruption costs.\n\n"
        "Output is written in chunks. Finished chunks survive closing the app, "
        "sleeping, or pressing Stop, and the render continues from there.\n\n"
        "500 default — about three minutes at worst."
    ),
    "vram_budget": (
        "How much graphics memory the render may use.\n\n"
        "Lower it to keep the machine usable for other things; the render "
        "slows but never fails. Raise it for full speed on an idle machine.\n\n"
        "Running short is handled automatically either way."
    ),
    "cpu": (
        "Use the processor instead of the graphics card.\n\n"
        "Tens of times slower. Only for machines without a usable GPU.\n\n"
        "Not needed to avoid running out of graphics memory — that is already "
        "handled."
    ),
    "preview_button": (
        "Render about ten seconds to judge the settings.\n\n"
        "Worth doing every time. A full film takes hours; this takes under a "
        "minute and answers the same question.\n\n"
        "Compare it with the original on a face you know."
    ),
    "render_button": (
        "Add these settings to the queue and start.\n\n"
        "Safe to leave overnight. If interrupted, start it again with the same "
        "settings and it continues from where it stopped."
    ),
    "cancel_button": (
        "Stop after the current frame.\n\n"
        "Finished work is kept. Start again later to continue."
    ),
    "progress": (
        "Frames done, current speed, time remaining.\n\n"
        "The estimate settles after the first minute. Much slower than "
        "expected usually means the model is too heavy — try a 2x one."
    ),
    "queue": (
        "Jobs waiting, running and finished.\n\n"
        "One runs at a time. Two would compete for the same graphics memory "
        "and both would be slower.\n\n"
        "Select a row to stop or remove it."
    ),
    "queue_start": (
        "Start the next waiting job, or restart the selected one.\n\n"
        "A job that was stopped continues from where it left off."
    ),
    "queue_stop": (
        "Stop the running job after the current frame.\n\n"
        "Progress is kept — starting it again continues from there."
    ),
    "queue_remove": (
        "Remove the selected job from the list.\n\n"
        "Stop a running job first. Files already written are left alone."
    ),
    "queue_clear": (
        "Clear finished, stopped and failed jobs from the list.\n\n"
        "Waiting and running jobs stay."
    ),
    "forecast": (
        "What these settings will produce, worked out before anything runs.\n\n"
        "Size, frame rate, the steps that will be applied, and rough time and "
        "file size. Updates as you change any control.\n\n"
        "Times come from throughput measured on this class of card. Treat them "
        "as a guide; once a render starts, the live speed readout is real."
    ),
    "guide_button": "Open a page explaining every setting.",
}


# Matched against the model file name.
MODEL_NOTES: dict[str, str] = {
    "2xParimgCompact": (
        "Fastest. Doubles size.\n\n"
        "Best for: full-length films, anything you want finished today. "
        "General purpose, good on live action.\n\n"
        "About 19 frames a second at DVD size."
    ),
    "2xModernSpanimationV1": (
        "Fast. Doubles size.\n\n"
        "Best for: animation, and clean live action. A little slower than "
        "Compact, often cleaner on flat colour.\n\n"
        "About 14 frames a second at DVD size."
    ),
    "4xNomos2_realplksr_dysample": (
        "Best texture, but slow. Quadruples size.\n\n"
        "Best for: close-ups and faces, short clips where skin detail is the "
        "point. Not practical for a full film.\n\n"
        "Cannot use the card's fast mode, so slower than its size suggests."
    ),
    "4xPurePhoto-span": (
        "Quadruples size. Photographic look.\n\n"
        "Best for: photographs and stills; short video clips.\n\n"
        "A 4x model does four times the work of a 2x one. For 1080p, 2x "
        "already reaches 4K."
    ),
    "realesr-general-x4v3": (
        "Quadruples size. Tolerant of poor sources.\n\n"
        "Best for: heavily compressed video — YouTube rips, old web video, "
        "VCD. Handles artifacts better than most.\n\n"
        "Fast for a 4x model, but still four times a 2x one."
    ),
    "RealESRGAN_x2plus": (
        "Slowest, highest fidelity. Doubles size.\n\n"
        "Best for: single photographs and short hero shots.\n\n"
        "Too slow for a film: about 2 frames a second at DVD size, over a day "
        "for a feature."
    ),
}

ARCH_NOTES: dict[str, str] = {
    "compact": "Compact architecture — the fastest family. Good general choice.",
    "span": "SPAN architecture — fast, clean results.",
    "plksr": "PLKSR architecture — excellent texture, noticeably slower.",
    "esrgan": "ESRGAN architecture — high quality, very slow on video.",
    "dat": "DAT architecture — very high quality, very slow.",
    "swinir": "SwinIR architecture — high quality, very slow.",
}


# Order and grouping used by the Guide window.
GUIDE_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Getting started", [
        ("Source", "source"),
        ("What was detected", "analysis"),
        ("Model", "model"),
        ("Rescan models", "rescan"),
    ]),
    ("Skin texture", [
        ("Degrain", "degrain"),
        ("Detail retention", "detail"),
        ("Re-grain", "regrain"),
        ("Deblock", "deblock"),
        ("Skip all restoration", "no_restore"),
    ]),
    ("Smoother motion", [
        ("Mode", "fps_mode"),
        ("Target frame rate", "fps_target"),
        ("Multiplier", "fps_multiplier"),
        ("Cut sensitivity", "scene_threshold"),
    ]),
    ("Output and performance", [
        ("Output file", "output"),
        ("Segment frames", "segment_frames"),
        ("Graphics memory", "vram_budget"),
        ("Force CPU", "cpu"),
    ]),
    ("Running jobs", [
        ("Preview", "preview_button"),
        ("Render", "render_button"),
        ("Cancel", "cancel_button"),
        ("Progress", "progress"),
        ("Queue", "queue"),
        ("Start", "queue_start"),
        ("Stop", "queue_stop"),
        ("Remove", "queue_remove"),
        ("Clear finished", "queue_clear"),
        ("What you will get", "forecast"),
        ("Guide", "guide_button"),
    ]),
]

RECIPES = [
    (
        "Skin looks plastic or airbrushed",
        "Degrain to 0.10, Detail retention to 0.40, Re-grain to 0.80. "
        "Degrain is almost always the cause.",
    ),
    (
        "Picture looks blocky, or edges look fuzzy",
        "Deblock to 0.30, or 0.50 for a very poor source. Common with "
        "YouTube downloads and old rips.",
    ),
    (
        "Far too slow",
        "Use 2xParimgCompact, and prefer a 2x model over a 4x one. For 1080p, "
        "2x already reaches 4K. Model choice changes speed roughly tenfold.",
    ),
    (
        "Motion smears at cuts",
        "Cut sensitivity to 0.20 so shot changes are spotted sooner. Common "
        "in fast-cut song sequences.",
    ),
    (
        "Old DVD looks combed or striped",
        "Nothing to do — detected and corrected automatically. Check the Scan "
        "line reads telecined or interlaced rather than progressive.",
    ),
    (
        "Too grainy",
        "Re-grain to 0.30, Degrain up slightly to 0.35.",
    ),
    (
        "Machine unusable while rendering",
        "Lower Graphics memory. The render slows but stays reliable.",
    ),
]


# A readable label for each known model: what it is for, and how quick it is.
# Ordered by how often it is the right answer, which is the order they are
# listed in.
MODEL_LABELS: dict[str, tuple[str, str, int]] = {
    # file stem: (what it is for, speed word, sort rank)
    "2xParimgCompact": ("Video — general", "fastest", 0),
    "2xModernSpanimationV1": ("Animation & clean video", "fast", 1),
    "realesr-general-x4v3": ("Video — compressed sources", "fast", 2),
    "4xPurePhoto-span": ("Photos", "medium", 3),
    "4xNomos2_realplksr_dysample": ("Faces & close-ups", "slow", 4),
    "RealESRGAN_x2plus": ("Photos — highest quality", "very slow", 5),
}

# Guessed from the file name when the model is not one of the above.
CATEGORY_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("anime", "animation", "cartoon", "toon"), "Animation"),
    (("photo", "foto"), "Photos"),
    (("face", "portrait", "gfpgan", "codeformer"), "Faces"),
    (("text", "manga", "comic"), "Text & line art"),
    (("video", "film", "movie"), "Video"),
)


def model_scale(name: str) -> int:
    """Scale factor read from the file name, by community convention.

    Both orders are in use and both must be read, or the output size is
    reported wrongly: "4xNomos2" writes the number first, "realesr-general-x4v3"
    writes it second. Matching only one of them silently called a 4x model 2x.
    """
    import re

    leading = re.search(r"(?:^|[^0-9])([248])\s*[xX]", name)
    if leading:
        return int(leading.group(1))
    trailing = re.search(r"[xX]\s*([248])(?![0-9])", name)
    return int(trailing.group(1)) if trailing else 2


def model_category(name: str) -> str:
    """What kind of material a model suits, from its name."""
    stem = name.rsplit(".", 1)[0]
    for key, (purpose, _speed, _rank) in MODEL_LABELS.items():
        if key.lower() == stem.lower():
            return purpose
    lowered = stem.lower()
    for needles, category in CATEGORY_HINTS:
        if any(needle in lowered for needle in needles):
            return category
    return "General"


def display_name(name: str, source_width: int | None = None) -> str:
    """A label saying what a model is for, rather than what its file is called.

    "2xParimgCompact.pth" says nothing useful when choosing. This gives the
    purpose, the enlargement, and where relevant what that turns a common
    source into.
    """
    stem = name.rsplit(".", 1)[0]
    scale = model_scale(stem)

    speed = ""
    purpose = model_category(name)
    for key, (label, speed_word, _rank) in MODEL_LABELS.items():
        if key.lower() == stem.lower():
            purpose, speed = label, speed_word
            break

    parts = [purpose, f"{scale}x"]
    if source_width:
        parts.append(f"→ {_size_name(source_width * scale)}")
    if speed:
        parts.append(speed)
    return "  ·  ".join(parts)


def _size_name(width: int) -> str:
    """A familiar name for an output width, or the number itself."""
    for low, high, name in (
        (7000, 99999, "8K"),
        (3400, 4400, "4K"),
        (2400, 2800, "2K"),
        (1800, 2100, "1080p"),
        (1200, 1400, "720p"),
    ):
        if low <= width < high:
            return name
    return f"{width}px wide"


def model_rank(name: str) -> int:
    """Sort order: the usual answers first, the specialist ones last."""
    stem = name.rsplit(".", 1)[0]
    for key, (_p, _s, rank) in MODEL_LABELS.items():
        if key.lower() == stem.lower():
            return rank
    return 50


def describe_model(name: str) -> str:
    """Plain-language note for a model file name.

    Falls back to an architecture hint, then to a generic message, so a
    drop-in model the catalogue has never heard of still says something useful.
    """
    stem = name.rsplit(".", 1)[0]

    for key, note in MODEL_NOTES.items():
        if key.lower() == stem.lower():
            return note

    lowered = stem.lower()
    for key, note in ARCH_NOTES.items():
        if key in lowered:
            scale = "Quadruples size." if "4x" in lowered else (
                "Doubles size." if "2x" in lowered else ""
            )
            return f"{note}\n\n{scale}".strip()

    return (
        "A model you added yourself.\n\n"
        "Its architecture and scale are detected when it loads. Run a "
        "ten-second preview to see what it does and how fast it is."
    )
