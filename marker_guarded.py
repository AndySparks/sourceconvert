"""Run marker_single with one guard: a 0-pixel image is dropped, not saved.

    <marker venv python> marker_guarded.py <pdf> --output_dir <dir> [marker flags]

Same arguments and behaviour as `marker_single`, which convert.py ran directly
until 2026-11-10. marker's save_output writes the markdown, then saves every
extracted image with PIL, and PIL refuses to write an image with no pixels
("cannot write empty image as JPEG"). One such image aborted the whole run
after the text was already written, so *Obviously Awesome* failed the same way
on two nights and sat behind a durable retry hold.

An image with zero width or height carries nothing. Dropping it, and every
markdown reference to it, keeps the book convertible without inventing a
figure (a blank placeholder) or leaving a dangling link that mc-wiki's figure
gate would quarantine. Each drop is printed to stderr, so the conversion log
says what happened.

This file is run by the marker venv's interpreter, never imported by
convert.py: the sourceconvert venv does not have marker installed.
"""
from __future__ import annotations

import re
import sys


def drop_empty_images(text: str, images: dict) -> tuple[str, dict, list[str]]:
    """Remove 0-pixel images and their markdown references.

    Returns (text, images, dropped names). Pure: the inputs are not mutated.
    An image object without a usable size is left alone; only a measured zero
    is evidence of emptiness."""
    dropped = []
    for name, img in images.items():
        size = getattr(img, "size", None)
        if isinstance(size, tuple) and len(size) == 2 and (size[0] == 0 or size[1] == 0):
            dropped.append(name)
    if not dropped:
        return text, images, []
    kept = {k: v for k, v in images.items() if k not in dropped}
    for name in dropped:
        # `![alt](name)` and `![alt](name "title")`, wherever marker put them.
        text = re.sub(r'!\[[^\]]*\]\(' + re.escape(name) + r'(?:\s+"[^"]*")?\)', "", text)
    return text, kept, dropped


def main() -> int:
    # Importing convert_single first sets marker's own environment (MPS
    # fallback, log levels) before anything else loads, exactly as
    # marker_single does.
    from marker.scripts.convert_single import convert_single_cli
    import marker.output as output

    original = output.text_from_rendered

    def guarded(rendered):
        text, ext, images = original(rendered)
        text, images, dropped = drop_empty_images(text, images or {})
        for name in dropped:
            print(f"sourceconvert: dropped empty (0-pixel) image {name} and its references", file=sys.stderr)
        return text, ext, images

    # save_output looks text_from_rendered up in marker.output at call time.
    output.text_from_rendered = guarded
    sys.argv[0] = "marker_single"
    return convert_single_cli()


if __name__ == "__main__":
    sys.exit(main())
