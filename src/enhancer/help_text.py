"""Explanatory text for the interface.

Kept out of the widget code so it can be tested, reviewed and corrected without
touching Qt, and so the same wording can be reused by any other front end.

Every control the user can touch must have an entry here. `test_help_text.py`
enforces that, so adding a control without explaining it fails the suite.
"""

from __future__ import annotations

# Keys match the control names used in `window.py`.
HELP: dict[str, str] = {
    "source": (
        "The video or image you want to enlarge.\n\n"
        "Video: .mp4, .mkv, .mov, .avi, .webm\n"
        "Images: .png, .jpg, .webp (transparency is preserved)\n\n"
        "Drop a file here, or click Browse."
    ),
    "analysis": (
        "What was found in your file, checked automatically.\n\n"
        "The Scan line matters most:\n\n"
        "• progressive — a normal modern file, nothing to correct.\n\n"
        "• telecined — film transferred to video, common on DVDs of older "
        "movies. The original film frames are recovered exactly. This is not "
        "the same as interlaced, and treating it as such would soften every "
        "frame permanently.\n\n"
        "• interlaced — genuine video, typically from TV or camcorders. Gets a "
        "different correction.\n\n"
        "Blockiness above about 2 means the file was compressed hard; raise "
        "Deblock. Grain above about 8 means a grainy film print."
    ),
    "model": (
        "The AI model that does the enlarging.\n\n"
        "This choice affects speed far more than anything else here — the "
        "slowest model is roughly ten times slower than the fastest.\n\n"
        "Hover any entry in the list for what it suits. If unsure, start with "
        "2xParimgCompact."
    ),
    "rescan": (
        "Re-read the models\\custom folder.\n\n"
        "Use this after dropping in a new .pth file, so you do not have to "
        "restart the app."
    ),
    "no_restore": (
        "Turn off all cleanup and texture work — just enlarge.\n\n"
        "About 35% faster, and useful for a quick look at what the model alone "
        "does. Leave it off for a real render: without re-grain, skin tends to "
        "look plastic."
    ),
    "degrain": (
        "Removes noise and film grain BEFORE enlarging.\n\n"
        "Some is necessary, because the model would otherwise magnify grain "
        "into ugly digital noise.\n\n"
        "But this is the control most likely to ruin skin. Grain and skin "
        "micro-texture sit at the same level of fine detail, so removing one "
        "removes the other. Too much of this is what makes faces look "
        "airbrushed.\n\n"
        "If skin looks waxy, LOWER this first.\n\n"
        "0.10 — grainy film you want to keep looking like film\n"
        "0.25 — sensible default\n"
        "0.50 — noisy video sources, VHS captures"
    ),
    "detail": (
        "Puts the source's own fine detail back over the enlarged frame.\n\n"
        "This is the only control that restores real photographed texture "
        "rather than texture the model invented. It takes the fine detail out "
        "of your original file and lays it back on top, so it physically "
        "cannot make up pores or hairs that were never there.\n\n"
        "If skin looks smooth or plastic, RAISE this.\n\n"
        "0.25 — sensible default\n"
        "0.40 — faces and close-ups\n"
        "0.60 — very soft or blurry sources"
    ),
    "regrain": (
        "Adds fine film grain back AFTER enlarging.\n\n"
        "AI models strip the faint texture of skin because it looks like noise "
        "to them. Putting grain back is the single strongest thing you can do "
        "to stop faces looking like plastic.\n\n"
        "Grain is strongest in mid-tones and fades out in bright and dark "
        "areas, the way real film behaves, and it changes every frame so it "
        "moves rather than sitting still like dirt on the lens.\n\n"
        "0.60 — sensible default\n"
        "0.80 — old film prints, or if skin still looks too clean\n"
        "0.20 — modern digital footage that was never grainy"
    ),
    "deblock": (
        "Removes the blocky squares and mosquito-like fuzz left by heavy "
        "compression.\n\n"
        "Off by default, because it softens the picture slightly and clean "
        "sources do not need it. Turn it up when the Blockiness reading above "
        "is high.\n\n"
        "0.00 — Blu-ray, clean digital files\n"
        "0.30 — YouTube downloads, streaming rips\n"
        "0.50 — VCD, low-bitrate rips, old web video"
    ),
    "fps_mode": (
        "Makes motion smoother by inventing extra in-between frames.\n\n"
        "Nothing is lost — your original frames are kept exactly as they are, "
        "and new ones are added between them.\n\n"
        "Off — keep the original frame rate\n"
        "Target FPS — choose the exact result, e.g. 60\n"
        "Multiplier — double or triple whatever the source happens to be\n\n"
        "This roughly doubles render time, and needs RIFE weights installed."
    ),
    "fps_target": (
        "The frame rate you want out.\n\n"
        "60 — smooth playback on a normal screen, the usual choice\n"
        "48 — a gentler lift from 24, keeps some of the cinema feel\n"
        "50 — for PAL sources and European TVs\n"
        "120 — high refresh-rate screens, or for slow motion later\n\n"
        "Awkward ratios are fine: 24 to 60 is 2.5 times, which works properly "
        "here rather than being rounded off."
    ),
    "fps_multiplier": (
        "Multiply the source rate instead of naming a target.\n\n"
        "2 — twice as smooth (24 becomes 48, 25 becomes 50)\n"
        "3 — three times\n"
        "4 — four times, for slow motion\n\n"
        "Useful when you do not know or care what the source rate is."
    ),
    "scene_threshold": (
        "How eagerly the app spots a cut between shots.\n\n"
        "This matters for fast-cut footage like song and dance sequences. "
        "Inventing a frame ACROSS a cut would smear two unrelated shots into "
        "each other, which looks awful. At a cut, a real frame is repeated "
        "instead — invisible to the eye.\n\n"
        "0.30 — sensible default\n"
        "0.20 — very fast cutting, if you see smearing at cuts\n"
        "0.45 — long unbroken takes, if real motion is being mistaken for cuts"
    ),
    "output": (
        "Where the finished file goes.\n\n"
        ".mkv is recommended: it holds 10-bit video, audio and subtitles "
        "without fuss. For images, the extension you type decides the format."
    ),
    "segment_frames": (
        "How much work you lose if the render is interrupted.\n\n"
        "The output is written in chunks. If the app closes, the machine "
        "sleeps, or you press Cancel, finished chunks are kept and the render "
        "picks up from there.\n\n"
        "500 — sensible default, about three minutes of lost work at worst\n"
        "Lower — lose less on interruption, marginally slower overall"
    ),
    "cpu": (
        "Run on the processor instead of the graphics card.\n\n"
        "Dramatically slower — think tens of times, not a bit. Only useful if "
        "the graphics card is unavailable or you need it for something else.\n\n"
        "You do not need this to avoid running out of graphics memory: that is "
        "handled automatically."
    ),
    "preview_button": (
        "Render about ten seconds, so you can judge the settings.\n\n"
        "Do this every time. A full film takes many hours; ten seconds takes "
        "under a minute and answers the same question.\n\n"
        "Open the result next to your original and look at a face you know "
        "well. If the skin looks airbrushed: lower Degrain, raise Detail "
        "retention, raise Re-grain."
    ),
    "render_button": (
        "Render the whole file.\n\n"
        "Safe to leave running overnight. If it is interrupted for any reason, "
        "reopen the app with the same settings and press Render again — it "
        "continues from where it stopped."
    ),
    "cancel_button": (
        "Stop after the current frame.\n\n"
        "Nothing already finished is thrown away. Press Render again later "
        "with the same settings and it carries on."
    ),
    "progress": (
        "Frames done, current speed, and estimated time left.\n\n"
        "The estimate settles down after the first minute or so. If the speed "
        "is far lower than expected, a faster model is usually the answer."
    ),
}


# Matched against the model file name, longest match first.
MODEL_NOTES: dict[str, str] = {
    "2xParimgCompact": (
        "FASTEST. Doubles size.\n\n"
        "Best for: full-length films, anything where you need it finished "
        "today. General-purpose, handles live action well.\n\n"
        "Roughly 19 frames per second at DVD resolution."
    ),
    "2xModernSpanimationV1": (
        "FAST. Doubles size.\n\n"
        "Best for: animation and cartoons, and clean live action. Slightly "
        "slower than Compact, often a little cleaner on flat colour.\n\n"
        "Roughly 14 frames per second at DVD resolution."
    ),
    "4xNomos2_realplksr_dysample": (
        "BEST TEXTURE, but slow. Quadruples size.\n\n"
        "Best for: close-ups, faces, short clips where skin detail is the "
        "whole point. Not practical for a full film.\n\n"
        "Note: this one cannot use the graphics card's fast mode, so it runs "
        "several times slower than its size suggests."
    ),
    "4xPurePhoto-span": (
        "Quadruples size. Photographic look.\n\n"
        "Best for: photographs and stills. Usable on short video clips.\n\n"
        "Remember a 4x model does four times the work of a 2x one — for a "
        "1080p source, 2x already reaches 4K."
    ),
    "realesr-general-x4v3": (
        "Quadruples size. Tolerant of poor sources.\n\n"
        "Best for: heavily compressed video — YouTube rips, old web video, "
        "VCD. Copes with artifacts better than most.\n\n"
        "Fast for a 4x model, but still four times the work of a 2x one."
    ),
    "RealESRGAN_x2plus": (
        "SLOWEST, highest fidelity. Doubles size.\n\n"
        "Best for: single photographs and short hero shots.\n\n"
        "Not usable for a full film: roughly 2 frames per second at DVD "
        "resolution, which is over a day for a feature."
    ),
}

ARCH_NOTES: dict[str, str] = {
    "compact": "Compact architecture — the fastest family. Good general choice.",
    "span": "SPAN architecture — fast, clean results.",
    "plksr": "PLKSR architecture — excellent texture, noticeably slower.",
    "esrgan": "ESRGAN architecture — high quality but very slow on video.",
    "dat": "DAT architecture — very high quality, very slow.",
    "swinir": "SwinIR architecture — high quality, very slow.",
}


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
        "Its architecture and scale are detected automatically when it loads. "
        "Run a ten-second preview to see what it does and how fast it is."
    )
