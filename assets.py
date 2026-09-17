"""Image region detection, rendering, and reference bookkeeping.

Two halves live here.

**Detection and rendering** (pymupdf backend): extracts figures, diagrams,
and raster images from PDF pages and renders them as PNGs via clipped
`page.get_pixmap`. The rendered assets get a markdown image reference
stitched into the page text by the pymupdf backend.

Three sources feed the region list:
  1. Raster images embedded in the PDF (page.get_image_info()).
  2. Vector drawings clustered by bounding box (page.get_drawings()).
  3. Figure captions near the above regions (CAPTION_RE).

**Reference bookkeeping** (every backend): the invariant that the emitted
markdown never references a file that does not exist, plus the asset
manifest that lets a consumer relocate assets without knowing how any
backend names its files. See `enforce_reference_invariant` and
`build_asset_manifest`. This half deliberately does not import fitz-level
concepts — it works on markdown text and the filesystem, so every backend
(marker, pandoc, pymupdf4llm, docling, pymupdf) can end with the same call.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Tuple
from urllib.parse import unquote, urlsplit

import fitz


# Captions: "Figure 1", "Figure 6.1", "FIGURE 3", "Exhibit 2.4", "Diagram 5".
# Must appear at the start of a line-ish string. Requires a number after
# the label word so body-text sentences that happen to start with "Figure"
# don't match.
CAPTION_RE = re.compile(
    r"^\s*(?:figure|fig\.?|exhibit|diagram|chart)\s+\d+(?:[.\-]\d+)?\b",
    re.IGNORECASE,
)


# Minimum pixel area for a raster region to count. Tiny icons / logos /
# bullet decorations under this threshold are ignored.
MIN_IMAGE_AREA = 2000  # ~45x45 px at PDF native resolution


class RasterProbeFailed(Exception):
    """`get_image_info` could not be read for a page.

    Distinct from "this page has no images". The detector must not count a
    page it could not inspect as evidence that a book is not a scan -- that
    is the direction that loses text.
    """


def find_raster_regions(page: fitz.Page) -> List[fitz.Rect]:
    """Return bounding boxes for embedded raster images on this page.

    Filters out tiny images (below MIN_IMAGE_AREA) that are almost always
    decorative: bullet glyphs, page decorations, imprint logos.

    Raises RasterProbeFailed if the page's image table cannot be read at all.
    Callers that only want figures treat that as "no images"; the page-scan
    detector treats it as "no evidence" instead.
    """
    regions: List[fitz.Rect] = []
    try:
        # get_image_info returns dicts with 'bbox' key in PDF points.
        info = page.get_image_info(xrefs=True)
    except Exception as exc:
        raise RasterProbeFailed(str(exc)) from exc
    for item in info:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        rect = fitz.Rect(*bbox)
        if rect.is_empty or (rect.width * rect.height) < MIN_IMAGE_AREA:
            continue
        regions.append(rect)
    return regions


# Minimum vector-drawing cluster area. Single stroke-width lines (rules
# above/below headings, underlines, footnote separators) are tiny and
# never qualify.
MIN_VECTOR_AREA = 2500
# Two boxes cluster together if they overlap OR are within this many PDF
# points of each other.
VECTOR_CLUSTER_GAP = 30


def find_vector_regions(page: fitz.Page) -> List[fitz.Rect]:
    """Return bounding boxes for clustered vector drawings on this page.

    PyMuPDF's get_drawings returns one entry per drawing primitive
    (stroke, fill, rect). A figure is a cluster of primitives with
    overlapping or nearby bounding boxes. We merge until no more merges
    are possible, then drop clusters below MIN_VECTOR_AREA.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    boxes: List[fitz.Rect] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None or rect.is_empty:
            continue
        boxes.append(fitz.Rect(rect))

    # Merge until stable.
    merged = _merge_rects(boxes, gap=VECTOR_CLUSTER_GAP)

    # Filter by area (use width * height — fitz.Rect.get_area not available).
    return [r for r in merged if r.width * r.height >= MIN_VECTOR_AREA]


def _merge_rects(rects: List[fitz.Rect], gap: float) -> List[fitz.Rect]:
    """Iteratively merge rects that overlap or touch within `gap` points."""
    out = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            j = i + 1
            while j < len(out):
                if _rects_close(out[i], out[j], gap):
                    out[i] = out[i] | out[j]  # union
                    del out[j]
                    changed = True
                else:
                    j += 1
            i += 1
    return out


def _rects_close(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
    """True if two rects overlap or are within `gap` PDF points."""
    if a.intersects(b):
        return True
    # Expand `a` by `gap` on all sides and test intersection.
    expanded = fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap)
    return expanded.intersects(b)


# Pad each region by this many PDF points before rendering so captions
# and borders don't get cropped.
REGION_PADDING = 4
# Caption must be within this many points below the region.
CAPTION_SEARCH_DISTANCE = 40
# DPI for rendered assets.
ASSET_DPI = 220

# --- page scans are not figures --------------------------------------------
#
# A home-scanned or archive.org book is one full-page bitmap per page with an
# OCR text layer sitting on top of it. `find_raster_regions` sees that bitmap
# as a figure region spanning the entire page, and the region splicer in
# convert.py emits region markdown IN PLACE OF the text underneath it -- so a
# full-page region consumes every character on the page.
#
# On William James's *Principles of Psychology* Vol 1 (704 pages) that turned
# the whole book into 704 `![Figure on page N]` lines and 9,350 words, about
# 1% of the text. Nothing caught it: quality_score read 1.0, the
# degenerate-text gate was clean, and the locator gate was clean.
#
# THE DECISION IS PER DOCUMENT, NOT PER PAGE, and that is the whole subtlety.
#
# Per page you cannot tell these two apart without guessing:
#
#   - a scanned book's chapter-opener, which carries a full-page bitmap and
#     only three lines of text; and
#   - a prose book's full-page plate, which carries a full-page image and only
#     its caption.
#
# Guessing by "how much text is on this page" gets one of them wrong every
# time, and the first draft of this fix did: it used a 200-character floor,
# which would have silently eaten the text of every sparse page in a scanned
# book -- a smaller version of the very bug it was written for.
#
# A book, though, is scanned throughout or it is not. So we ask the DOCUMENT:
# do most of its pages carry a full-page bitmap with text on top? If yes, every
# full-page bitmap in it is a page scan and none of them is a figure. If no, a
# full-page image is a plate and is kept.
#
# The text condition in the detector is what protects an un-OCR'd scan: a book
# of page images with NO text layer is never classified as a page-scan
# document, so its images -- the only content it has -- are still emitted.
# A region must cover this fraction of the page in BOTH dimensions. Comparing
# areas alone is not enough: a banner twice the page width and 45% of its
# height has the area of 90% of the page without covering it, and an image
# hanging off the crop box is bigger than the page it sits on. Both dimensions
# plus the intersection with the page rect is what "covers the page" means.
PAGE_SCAN_COVER_RATIO = 0.9
# A page needs this much text to count as carrying a text layer.
PAGE_SCAN_MIN_TEXT_CHARS = 200
# Of the sampled pages THAT CARRY TEXT, this fraction must be full-page
# bitmaps before the document is treated as a scan. The denominator is
# text-bearing pages, not all sampled pages, because a scanned book's blank
# versos, plate pages and sparse chapter-openers are not evidence either way
# and must not drag a real scan below the bar -- that was the defect in the
# first draft, which classified a 3-page scan with one text-heavy page as
# not-a-scan and ate all three text layers.
PAGE_SCAN_DOC_FRACTION = 0.6
# Sampling bound -- 704-page books should not pay a full extra pass.
PAGE_SCAN_SAMPLE_PAGES = 40


def _page_text_length(page: fitz.Page) -> int:
    try:
        return len(page.get_text().strip())
    except Exception:
        return 0


def _covers_page(rect: fitz.Rect, page: fitz.Page) -> bool:
    """True if `rect` covers essentially the whole page in both dimensions."""
    page_rect = page.rect
    if page_rect.width <= 0 or page_rect.height <= 0:
        return False
    # Only the part of the region actually on the page counts.
    visible = fitz.Rect(rect) & page_rect
    if visible.is_empty:
        return False
    return (
        visible.width / page_rect.width >= PAGE_SCAN_COVER_RATIO
        and visible.height / page_rect.height >= PAGE_SCAN_COVER_RATIO
    )


def _sample_indices(total: int, want: int):
    """Up to `want` indices spread evenly across [0, total).

    `range(0, total, total // want)` truncated to `want` entries samples only
    the FRONT of the document whenever the step rounds down to 1 -- a 79-page
    book was sampled over pages 0-39 and its whole second half went unseen.
    """
    if total <= 0 or want <= 0:
        return []
    if total <= want:
        return list(range(total))
    return sorted({round(i * (total - 1) / (want - 1)) for i in range(want)})


def detect_page_scan_document(doc) -> bool:
    """True if `doc` is a scanned book: page bitmaps with a text layer on top.

    Samples evenly across the document rather than reading every page, so a
    700-page book does not pay a second full pass.

    The ratio is taken over pages that CARRY TEXT. A document with no
    text-bearing pages at all -- an un-OCR'd scan, a book of plates -- is
    never a page-scan document, so its images survive: they are the only
    content it has.

    KNOWN, DELIBERATE ASYMMETRY on very short documents. A two-page PDF whose
    single text-bearing page is dominated by a legitimate full-page infographic
    scores 1/1 and is called a scan, and that infographic is dropped. There is
    no minimum-evidence floor to prevent it, because a floor would push short
    genuine scans the other way -- and the two errors are not equal. Dropping
    an image loses an image. Failing to detect a scan loses the TEXT of every
    page, silently, with every downstream gate green. In a corpus whose product
    is quotable text, the error that keeps the text is the one to prefer.
    """
    try:
        total = doc.page_count
    except Exception:
        return False

    indices = _sample_indices(total, PAGE_SCAN_SAMPLE_PAGES)
    if not indices:
        return False

    text_pages = 0
    scanned = 0
    for i in indices:
        try:
            page = doc[i]
            if _page_text_length(page) < PAGE_SCAN_MIN_TEXT_CHARS:
                continue
            # Merge before testing coverage. `extract_page_assets` merges
            # raster regions before it decides what to splice, so a detector
            # that tests the UNMERGED regions is answering a different
            # question than the extractor asks. Acrobat Capture stores a page
            # scan as a stack of full-width bands; no band covers the page,
            # the merge of them is the page. See
            # test_a_scan_sliced_into_strips_is_still_a_scan.
            rasters = _merge_rects(find_raster_regions(page), gap=REGION_PADDING)
            text_pages += 1
            if any(_covers_page(r, page) for r in rasters):
                scanned += 1
        except Exception:
            # A page that cannot be read is not evidence either way, and must
            # not sit in the denominator biasing the answer toward "not a
            # scan" -- the answer that loses text.
            continue
    if text_pages == 0:
        return False
    return (scanned / text_pages) >= PAGE_SCAN_DOC_FRACTION


def extract_page_assets(
    page: fitz.Page,
    stem: str,
    asset_dir: Path,
    page_num: int,
    page_scan_document: bool = False,
) -> List[Tuple[fitz.Rect, str]]:
    """Render all figure regions on a page and return (rect, markdown) pairs.

    `stem` is the markdown output filename without extension — used to
    build an asset subdirectory name. `page_num` is the 1-indexed page
    number used in the asset filename.

    `page_scan_document` comes from `detect_page_scan_document` and says the
    book is a scan; when it is set, a region covering the whole page is the
    page's own bitmap and is dropped so the text layer survives. See
    "page scans are not figures" above.
    """
    try:
        raster = find_raster_regions(page)
    except RasterProbeFailed:
        # For figure extraction, an unreadable image table means no figures.
        raster = []
    vector = find_vector_regions(page)

    # Combine and merge overlapping regions.
    combined = _merge_rects(raster + vector, gap=REGION_PADDING)
    if page_scan_document:
        # Drop the page's own scan bitmap (see "page scans are not figures").
        combined = [r for r in combined if not _covers_page(r, page)]
    if not combined:
        return []

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Harvest caption candidates from page text.
    captions = _find_caption_candidates(page)

    results: List[Tuple[fitz.Rect, str]] = []
    for idx, rect in enumerate(combined):
        caption_text = _match_caption(rect, captions)
        padded = fitz.Rect(
            max(0, rect.x0 - REGION_PADDING),
            max(0, rect.y0 - REGION_PADDING),
            min(page.rect.x1, rect.x1 + REGION_PADDING),
            min(page.rect.y1, rect.y1 + REGION_PADDING),
        )
        asset_name = f"page-{page_num:04d}-figure-{idx + 1:02d}.png"
        asset_path = asset_dir / asset_name
        try:
            pix = page.get_pixmap(dpi=ASSET_DPI, clip=padded, alpha=False)
            pix.save(str(asset_path))
        except Exception:
            continue

        rel = f"{asset_dir.name}/{asset_name}"
        alt = caption_text if caption_text else f"Figure on page {page_num}"
        md = f"![{alt}]({rel})"
        if caption_text:
            md = f"{md}\n\n*{caption_text}*"
        results.append((padded, md))
    return results


def _find_caption_candidates(page: fitz.Page) -> List[Tuple[fitz.Rect, str]]:
    """Return (bbox, text) pairs for lines on the page that match CAPTION_RE."""
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    out: List[Tuple[fitz.Rect, str]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            text = " ".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if not text:
                continue
            if CAPTION_RE.match(text):
                bbox = line.get("bbox")
                if not bbox:
                    continue
                out.append((fitz.Rect(*bbox), text))
    return out


def _match_caption(
    region: fitz.Rect,
    captions: List[Tuple[fitz.Rect, str]],
):
    """Return the closest caption within CAPTION_SEARCH_DISTANCE below region."""
    best = None
    best_dist = float("inf")
    for bbox, text in captions:
        # Caption must be below (or overlapping) the region, roughly
        # within its horizontal span.
        if bbox.y0 < region.y0:
            continue
        dy = bbox.y0 - region.y1
        if dy > CAPTION_SEARCH_DISTANCE:
            continue
        # Horizontal overlap check: caption intersects region's x-range.
        if bbox.x1 < region.x0 or bbox.x0 > region.x1:
            continue
        if dy < best_dist:
            best_dist = dy
            best = text
    return best


# ---------------------------------------------------------------------------
# Reference bookkeeping — the never-dangle invariant and the asset manifest
# ---------------------------------------------------------------------------
#
# The invariant: the markdown a conversion emits never contains a reference to
# a file that does not exist. It was broken for a year by the marker backend,
# which writes `_page_64_Figure_7.jpeg` files into its scratch directory and
# emits bare `![](_page_64_Figure_7.jpeg)` references to them; sourceconvert
# harvested `*.png`/`*.jpg` (never `*.jpeg`) out of that scratch directory and
# then deleted it, leaving every reference pointing at nothing. 1,140 dead
# references across 49 sources downstream. See issue #34.
#
# Enforcement is two-layered on purpose:
#   1. Extract by default, so the asset the reference names is actually there.
#   2. Sweep afterwards regardless, so a reference that still has no file —
#      because extraction was disabled, or because a backend emitted a
#      reference to something it never wrote — is stripped, not left dangling.
# Layer 2 is the invariant; layer 1 is what makes satisfying it not a loss.

# Every raster suffix any backend of ours emits, plus the ones they might.
IMAGE_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".svg",
})

# A markdown image reference. The inner group is everything between the
# parentheses — target plus an optional "title" — split apart by
# `_split_link_target` rather than by a hairier regex.
IMAGE_REF_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<inner>[^()]*)\)')

# What a stripped reference leaves behind: nothing a reader sees, and enough
# for a human debugging a thin conversion to find out what happened.
STRIPPED_REF_COMMENT = "<!-- sourceconvert: image omitted, asset not extracted: {target} -->"

# A trailing `"title"` / 'title' on a link target.
_LINK_TITLE_RE = re.compile(r'''\s+["'][^"']*["']\s*$''')


class ImageRef(NamedTuple):
    """One markdown image reference located in a document."""

    alt: str
    target: str          # exactly as written between the parens
    start: int           # character offset of the `!` in the source text
    end: int             # character offset one past the closing paren
    line: int            # 1-indexed line number


def _split_link_target(inner: str) -> str:
    """Return the target from a markdown link's parenthesised inner text."""
    inner = inner.strip()
    angled = re.match(r'^<(?P<t>[^>]*)>', inner)
    if angled:
        return angled.group("t").strip()
    return _LINK_TITLE_RE.sub("", inner).strip()


def is_local_target(target: str) -> bool:
    """True if `target` names a file on disk rather than a remote resource.

    Absolute paths count as local — they are still a file reference, and one
    that points outside the output directory is exactly the kind of thing the
    invariant should catch.
    """
    if not target:
        return False
    if target.startswith("//") or target.startswith("#"):
        return False
    scheme = urlsplit(target).scheme
    # A bare Windows drive letter ("c:/x.png") parses as a scheme; a real
    # scheme is at least two characters.
    return len(scheme) < 2


def iter_image_refs(text: str) -> List[ImageRef]:
    """Every markdown image reference in `text`, in document order."""
    refs: List[ImageRef] = []
    for m in IMAGE_REF_RE.finditer(text):
        target = _split_link_target(m.group("inner"))
        refs.append(ImageRef(
            alt=m.group("alt"),
            target=target,
            start=m.start(),
            end=m.end(),
            line=text.count("\n", 0, m.start()) + 1,
        ))
    return refs


def resolve_target(md_path: Path, target: str) -> Path:
    """Filesystem path a local reference in `md_path` points at."""
    decoded = unquote(target)
    path = Path(decoded)
    if path.is_absolute():
        return path
    return Path(md_path).parent / path


def collect_asset_files(root: Path) -> List[Path]:
    """Every image file under `root`, recursively, sorted for determinism.

    Suffix matching is case-insensitive and covers `.jpeg` as well as `.jpg`
    — the one-character gap that produced issue #34.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def harvest_assets(files: Iterable[Path], dest_dir: Path) -> Dict[str, str]:
    """Move `files` into `dest_dir`, flattened, and map old name -> new name.

    Backends drop assets in nested scratch directories under names that are
    unique only within their own subdirectory, so flattening can collide. A
    collision gets the parent directory name prefixed rather than silently
    overwriting the earlier file — no asset is ever lost to a name clash.

    The returned mapping is keyed by original basename, which is also how
    references are matched, so two genuinely different files sharing a
    basename cannot be told apart by a reference either. First one wins; both
    files are kept. This is a real ambiguity in the input, not one this
    function introduces, and the invariant holds regardless: the reference
    points at a file that exists.
    """
    files = [Path(f) for f in files]
    if not files:
        return {}
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    taken: set = {p.name for p in dest_dir.iterdir()} if dest_dir.exists() else set()
    for src in files:
        name = src.name
        if name in taken:
            name = f"{src.parent.name}-{src.name}"
            counter = 2
            while name in taken:
                name = f"{src.parent.name}-{counter}-{src.name}"
                counter += 1
        taken.add(name)
        shutil.move(str(src), str(dest_dir / name))
        mapping.setdefault(src.name, name)
    return mapping


def rewrite_asset_refs(text: str, mapping: Dict[str, str], dest_rel: str) -> Tuple[str, int]:
    """Point every local reference whose basename is in `mapping` at `dest_rel`.

    Matching is on the reference's *basename*, so it works whether the backend
    emitted a bare filename (`![](_page_64_Figure_7.jpeg)`, marker) or a path
    with directories in it (`![](images/x.png)`). Returns (text, rewrites).
    """
    if not mapping:
        return text, 0

    rewrites = 0

    def replace(m: re.Match) -> str:
        nonlocal rewrites
        target = _split_link_target(m.group("inner"))
        if not is_local_target(target):
            return m.group(0)
        basename = Path(unquote(target)).name
        new_name = mapping.get(basename)
        if new_name is None:
            return m.group(0)
        rewrites += 1
        return f"![{m.group('alt')}]({dest_rel}/{new_name})"

    return IMAGE_REF_RE.sub(replace, text), rewrites


def strip_dangling_refs(text: str, md_path: Path) -> Tuple[str, List[str]]:
    """Remove every local image reference whose file is not on disk.

    Returns (text, stripped_targets). Remote references are left alone: this
    module makes no claim about the internet. The stripped reference is
    replaced by an HTML comment, which renders as nothing and keeps the fact
    that something was dropped recoverable by a human.
    """
    stripped: List[str] = []

    def replace(m: re.Match) -> str:
        target = _split_link_target(m.group("inner"))
        if not is_local_target(target):
            return m.group(0)
        if resolve_target(md_path, target).exists():
            return m.group(0)
        stripped.append(target)
        return STRIPPED_REF_COMMENT.format(target=target)

    return IMAGE_REF_RE.sub(replace, text), stripped


# KNOWN LIMITATION: this sweeps the asset directory as it stands, so it counts
# files left behind by a PREVIOUS conversion into the same directory as assets
# the current run wrote. It became easier to hit once the page-scan filter
# started emitting no assets at all for a scanned book: reconverting such a
# book after upgrading leaves the old per-page PNGs on disk and reports them as
# current. No dangling markdown reference results -- the markdown simply does
# not name them -- so the never-dangle invariant holds; what is wrong is the
# asset COUNT in the report. Convert into a clean output directory after an
# upgrade. (Raised by codex review, 2026-08-31; not fixed here because clearing
# the directory is a destructive change that deserves its own change.)
def build_asset_manifest(md_path: Path, asset_dir=None) -> List[dict]:
    """Describe every asset this conversion wrote, and what points at it.

    `asset_dir` is a directory (or an iterable of directories) to sweep for
    assets that exist but are referenced by nothing.

    One entry per file, whether or not anything references it (an extracted
    but unreferenced asset is information too), plus one entry per referenced
    file that lives outside `asset_dir`. Paths are relative to the markdown
    file, so a consumer can relocate assets and rewrite references without
    knowing anything about how a backend names its files — which is the whole
    point: `_page_N_Figure_M.jpeg` is marker's business, not a contract.

    Entry shape:
        {"path": "Book_images/_page_64_Figure_7.jpeg",
         "bytes": 48213,
         "references": [{"target": "...", "alt": "...", "line": 812}]}
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    md_dir = md_path.parent
    text = md_path.read_text(encoding="utf-8", errors="replace")

    # Referenced files first, keyed by resolved path so two spellings of the
    # same file (`x.png` and `./x.png`) land on one entry.
    refs_by_file: Dict[Path, List[dict]] = {}
    for ref in iter_image_refs(text):
        if not is_local_target(ref.target):
            continue
        resolved = resolve_target(md_path, ref.target)
        if not resolved.exists():
            continue          # invariant already swept these; belt and braces
        refs_by_file.setdefault(resolved.resolve(), []).append(
            {"target": ref.target, "alt": ref.alt, "line": ref.line}
        )

    files: List[Path] = list(refs_by_file)
    candidates = ([asset_dir] if isinstance(asset_dir, (str, Path))
                  else list(asset_dir or []))
    for candidate in candidates:
        if not Path(candidate).is_dir():
            continue
        for path in collect_asset_files(candidate):
            if path.resolve() not in refs_by_file:
                files.append(path.resolve())

    manifest: List[dict] = []
    for path in sorted(set(files)):
        try:
            rel = path.relative_to(md_dir.resolve())
            rel_str = rel.as_posix()
        except ValueError:
            rel_str = path.as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        manifest.append({
            "path": rel_str,
            "bytes": size,
            "references": refs_by_file.get(path, []),
        })
    return manifest


def count_dangling_refs(md_path: Path) -> int:
    """How many local image references in `md_path` point at nothing.

    The invariant says this is 0 for every conversion sourceconvert emits. It
    exists so a test — and a suspicious operator — can assert that directly.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return 0
    text = md_path.read_text(encoding="utf-8", errors="replace")
    return sum(
        1 for ref in iter_image_refs(text)
        if is_local_target(ref.target)
        and not resolve_target(md_path, ref.target).exists()
    )


def enforce_reference_invariant(md_path: Path) -> List[str]:
    """Strip every dangling local image reference from `md_path`, in place.

    Returns the targets that were stripped. Writing only happens when
    something changed, so this is a cheap no-op on the common case (a
    text-only conversion, or one where every asset landed).
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8", errors="replace")
    swept, stripped = strip_dangling_refs(text, md_path)
    if stripped:
        md_path.write_text(swept, encoding="utf-8", errors="replace")
    return stripped
