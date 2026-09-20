#!/usr/bin/env python3
"""
sourceconvert - Convert PDF and EPUB books to clean Markdown.

PDF conversion methods:
  - pymupdf (default): Fast, reliable text extraction using PyMuPDF/fitz
  - marker: High-quality conversion using marker-pdf (requires Python 3.10+)
  - ocr: Tesseract OCR for scanned/image-based PDFs (slowest but handles images)

EPUB conversion: routed through pandoc, which preserves the epub's chapter
structure as markdown headings. The --method flag only applies to PDFs; epubs
always use pandoc. EPUBs that carry no semantic h1-h6 get their headings
derived from the book's own navigation first (see epub_structure.py); the
sidecar always declares `heading_source` and `headings_emitted`.

Usage:
    python convert.py input/MyBook.pdf
    python convert.py input/MyBook.epub                   # EPUB -> markdown via pandoc
    python convert.py input/MyBook.pdf --output output/
    python convert.py input/MyBook.pdf --method ocr      # Force OCR for scanned PDFs
    python convert.py input/MyBook.pdf --method marker    # Use marker-pdf
    python convert.py input/                              # Convert all PDFs/EPUBs in a directory
    python convert.py input/ --skip-existing              # Skip already-converted files
"""

import argparse
import collections
import io
import json
import logging
import os
import shlex
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

log = logging.getLogger("sourceconvert")

from report import ConversionReport, write_report
import assets
import cleanup
import epub_structure


class DependencyError(Exception):
    """Raised when a required dependency is missing."""
    pass


class ConversionError(Exception):
    """Raised when a conversion fails."""
    pass


def check_dependencies(method):
    """Check that required tools are installed for the chosen method.

    Raises DependencyError if anything is missing.
    """
    if method == "pymupdf":
        try:
            import fitz
        except ImportError:
            raise DependencyError("Missing dependency: PyMuPDF (pip install pymupdf)")

    elif method == "marker":
        # marker-pdf uses PEP 604 syntax (X | None) in its type hints, which
        # requires Python 3.10+. Fail early with a clear message rather than
        # surfacing a confusing TypeError from deep inside the import chain.
        if sys.version_info < (3, 10):
            raise DependencyError(
                "marker-pdf requires Python 3.10 or newer (current venv is "
                f"{sys.version_info.major}.{sys.version_info.minor}).\n"
                "  Either:\n"
                "    - Use the default pymupdf method (drop --papers / --method marker), or\n"
                "    - Create a Python 3.12 venv and reinstall: "
                "python3.12 -m venv .venv-marker && "
                ".venv-marker/bin/pip install marker-pdf"
            )
        # Use a static existence check rather than invoking `marker_single
        # --help`. The marker CLI loads its ML models at import time, so
        # `--help` routinely takes 30+ seconds on a cold start and trips a
        # 10s subprocess timeout even when marker is working fine.
        #
        # Look for `marker_single` in this order:
        #   1. The sys.executable's sibling bin directory (so running
        #      `.venv-marker/bin/python convert.py` finds
        #      `.venv-marker/bin/marker_single` without needing the venv
        #      to be activated on PATH).
        #   2. PATH itself (for users who've activated the venv).
        # The venv-sibling lookup matters because running sourceconvert
        # from a non-activated venv is the common case in tools and
        # scripts, and requiring activation introduces a class of silent
        # failures when the caller forgets `source`.
        import shutil
        import os as _os
        venv_bin = Path(sys.executable).parent
        marker_bin = venv_bin / "marker_single"
        marker_on_path = shutil.which("marker_single")
        if not marker_bin.exists() and marker_on_path is None:
            raise DependencyError(
                "Missing dependency: marker-pdf (pip install marker-pdf).\n"
                f"  Looked for `marker_single` in {venv_bin} and on PATH."
            )
        # The binary exists -- but a venv console script hardcodes its
        # interpreter, so "exists" is not "runnable" after a directory rename.
        # Check before a long conversion rather than after it fails to start.
        _found = marker_bin if marker_bin.exists() else Path(marker_on_path)
        _dead = _stale_shebang(_found)
        if _dead is not None:
            _bin_dir = _found.parent
            raise DependencyError(
                f"`{_found}` points at an interpreter that does not exist:\n"
                f"    {_dead}\n"
                "  This is what a moved or renamed project directory looks like -- the\n"
                "  script is present but every spawn fails with `bad interpreter`.\n"
                "  Repair the shebangs in place (line 1 only):\n"
                f"    grep -rl '{_dead}' {_bin_dir} | while read f; do \\\n"
                f"      sed -i '' \"1s|{_dead}|$(command -v python3)|\" \"$f\"; done\n"
                "  Or rebuild the venv from scratch:\n"
                f"    python3.12 -m venv --clear {_bin_dir.parent} && "
                f"{_bin_dir}/pip install marker-pdf"
            )
        # If the venv-sibling binary exists but isn't on PATH, add it so
        # convert_with_marker() can spawn it via plain subprocess.run.
        if marker_bin.exists() and marker_on_path is None:
            _os.environ["PATH"] = str(venv_bin) + _os.pathsep + _os.environ.get("PATH", "")
        try:
            import marker  # noqa: F401
        except ImportError:
            raise DependencyError(
                "Missing dependency: marker-pdf is not importable in this "
                "Python. Activate the marker venv (source .venv-marker/bin/activate) "
                "or reinstall: pip install marker-pdf"
            )

    elif method == "ocr":
        missing = []
        try:
            import pdf2image
        except ImportError:
            missing.append("pdf2image (pip install pdf2image)")
        try:
            import pytesseract
        except ImportError:
            missing.append("pytesseract (pip install pytesseract)")
        try:
            subprocess.run(["tesseract", "--version"], capture_output=True, timeout=10)
        except FileNotFoundError:
            missing.append("tesseract (brew install tesseract)")
        # Check for Poppler (required by pdf2image for PDF rendering)
        try:
            subprocess.run(["pdftoppm", "-v"], capture_output=True, timeout=10)
        except FileNotFoundError:
            missing.append("poppler (brew install poppler)")

        if missing:
            lines = ["Missing dependencies:"]
            for dep in missing:
                lines.append(f"  - {dep}")
            raise DependencyError("\n".join(lines))

    elif method == "pymupdf4llm":
        if sys.version_info < (3, 10):
            raise DependencyError(
                "pymupdf4llm requires Python 3.10 or newer (current venv is "
                f"{sys.version_info.major}.{sys.version_info.minor}).\n"
                "  Use the .venv-marker (Python 3.12) venv:\n"
                "    .venv-marker/bin/pip install -r requirements-pymupdf4llm.txt\n"
                "    .venv-marker/bin/python convert.py --method pymupdf4llm <pdf>"
            )
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            raise DependencyError(
                "Missing dependency: pymupdf4llm.\n"
                "  .venv-marker/bin/pip install -r requirements-pymupdf4llm.txt"
            )

    elif method == "docling":
        if sys.version_info < (3, 10):
            raise DependencyError(
                "docling requires Python 3.10 or newer (current venv is "
                f"{sys.version_info.major}.{sys.version_info.minor}).\n"
                "  Use the .venv-marker (Python 3.12) venv:\n"
                "    .venv-marker/bin/pip install -r requirements-docling.txt\n"
                "    .venv-marker/bin/python convert.py --method docling <pdf>"
            )
        try:
            import docling  # noqa: F401
        except ImportError:
            raise DependencyError(
                "Missing dependency: docling.\n"
                "  .venv-marker/bin/pip install -r requirements-docling.txt"
            )

    elif method == "pandoc":
        # EPUB conversion shells out to pandoc. We check for the binary on
        # PATH rather than importing a Python wrapper because pandoc is a
        # standalone tool with no official Python bindings; pypandoc would
        # just wrap the same subprocess call with an extra dependency.
        try:
            subprocess.run(
                ["pandoc", "--version"], capture_output=True, timeout=10
            )
        except FileNotFoundError:
            raise DependencyError(
                "Missing dependency: pandoc (brew install pandoc)\n"
                "  EPUB conversion uses pandoc to preserve chapter structure."
            )


def _stale_shebang(script_path):
    """Return the dead interpreter path if `script_path`'s shebang is broken.

    A venv console script hardcodes an absolute interpreter path on line 1.
    Move or rename the project directory and every one of those scripts keeps
    pointing at the old location: the file still exists, `shutil.which` still
    finds it, and it still fails the moment it is spawned, with a `bad
    interpreter` error from the kernel that never mentions the rename.

    That is not hypothetical. Renaming BookConvert to sourceconvert on
    2026-07-28 left 73 stale scripts in `.venv-marker/bin` and 42 in
    `.venv/bin`, and the next ingest died instantly on `marker_single`. The
    existence check above cannot catch it, so it is checked here.

    Returns None when the shebang is fine, unreadable, or not a shebang at
    all -- this is a diagnostic, and it must never be the reason a working
    conversion refuses to start.
    """
    try:
        with open(script_path, "rb") as fh:
            first = fh.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    line = first[2:].decode("utf-8", "replace").strip()
    if not line:
        return None
    # `#!/usr/bin/env python3` delegates the lookup to env -- nothing absolute
    # to verify, and env will resolve it against PATH at spawn time.
    interp = line.split()[0]
    if not interp.startswith("/") or Path(interp).name == "env":
        return None
    return None if Path(interp).exists() else interp


def _marker_available():
    """Cheap check: is marker-pdf installed in the current interpreter?

    Uses the same logic as `check_dependencies("marker")` but returns a
    bool instead of raising. Called by pick_ocr_backend to decide where
    to route a scanned PDF.
    """
    if sys.version_info < (3, 10):
        return False
    try:
        venv_bin = Path(sys.executable).parent
        marker_bin = venv_bin / "marker_single"
        if marker_bin.exists():
            return True
        return shutil.which("marker_single") is not None
    except Exception:
        return False


def pick_ocr_backend():
    """Choose between 'marker' and 'ocr' for scanned PDFs.

    Returns 'marker' when marker-pdf is installed (it handles scanned
    PDFs via its own layout-aware OCR pipeline and produces cleaner
    output than raw tesseract). Otherwise falls back to 'ocr'.
    """
    if _marker_available():
        return "marker"
    return "ocr"


_WORD_ORDINALS = (
    r'first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth'
)
_EDITION_QUALIFIERS = (
    r'annotated|revised|reissue|updated|expanded|enlarged|new|'
    + _WORD_ORDINALS
)


def clean_title(stem):
    """Derive a clean book title from the PDF filename stem.

    Strips common trailing version markers (e.g. "4th Edition", "Annotated
    Edition", "V3") and any leftover punctuation from the separator that
    preceded them.
    """
    title = stem
    # Apply version-marker strippers repeatedly so compound patterns like
    # "3rd Annotated Edition" get fully removed.
    prev = None
    # Optional adverbs that decorate compound edition phrases. Standalone
    # they are NOT trailing junk (the regex below only matches them when
    # followed by an actual qualifier word).
    adverb_prefix = r'(?:fully|completely|newly|partially|extensively)\s+'
    qualifier_chain = (
        r'(?:' + _EDITION_QUALIFIERS + r')'
        r'(?:\s*(?:,|and|&)\s*'
        r'(?:' + adverb_prefix + r')?'
        r'(?:' + _EDITION_QUALIFIERS + r'))*'
    )
    while prev != title:
        prev = title
        title = re.sub(r'\s*[Vv]\d+(\.\d+)?\s*$', '', title)
        title = re.sub(r'\s*\d+(st|nd|rd|th)\s+[Ee]dition\s*$', '', title)
        # Numbered Anniversary Edition: "10th Anniversary Edition"
        title = re.sub(
            r'\s*\d+(?:st|nd|rd|th)\s+Anniversary\s+[Ee]dition\s*$',
            '',
            title,
            flags=re.IGNORECASE,
        )
        # Compound qualifier strip. Handles "Annotated Edition", "Updated
        # and Expanded", "Fully Revised & Updated Edition", "Revised,
        # Updated, and Expanded Edition", etc. The leading adverb is only
        # matched when it precedes a real qualifier word.
        title = re.sub(
            r'\s*(?:' + adverb_prefix + r')?'
            + qualifier_chain
            + r'(?:\s+[Ee]dition)?'
            r'\s*$',
            '',
            title,
            flags=re.IGNORECASE,
        )
        title = re.sub(r'\s*\([^)]*[Ee]dition[^)]*\)\s*$', '', title)
        # Strip trailing separator punctuation left behind by the strippers
        # (e.g. "Title, 4th Edition" -> "Title," -> "Title")
        title = re.sub(r'[,;:\-\s]+$', '', title)
    return title.strip()


MIN_TEXT_RATIO = 0.1  # At least 10% of pages must have extractable text

# Minimum quality score for a pymupdf extraction to be accepted without
# falling back to OCR. Scored by _text_quality_score (artifact density).
# Calibrated against:
#   The Human Side of Enterprise (good pymupdf):   1.000
#   The Functions of the Executive (good OCR):     0.943
#   The Pyramid Principle (bad pymupdf, mangled):  0.000
# 0.5 leaves a 0.44+ margin on the good side and 0.5 on the bad side.
# See docs/quality-fallback-design.md for the rationale.
QUALITY_THRESHOLD = 0.5


# --- Text quality scoring ---
# Regex for stripping markdown syntax before measuring artifact density.
# Strips: HTML comments (<!-- ... -->), heading/bullet/emphasis markers,
# table pipes, backticks, and horizontal rules.
_MARKDOWN_STRIP_RE = re.compile(
    r'<!--.*?-->|[#*_`|]+|^-{3,}$',
    re.MULTILINE | re.DOTALL
)

# Font-encoding artifact patterns. Each match counts as one artifact.
# These are the specific extraction bugs we've seen in the wild from
# PyMuPDF on PDFs with non-standard Type 3 fonts or custom encodings.
# Density (matches per 10k chars) is the signal we gate on.
# Digit-acronyms: real vocabulary shaped exactly like the artifact below,
# where the digit is the word "to". These are REMOVED before counting rather
# than excluded by narrowing the pattern, because narrowing it to lowercase
# would stop flagging genuine damage in headings and all-caps pages.
#
# Origin 2026-09-20: Gornall & Strebulaev's VC field experiment failed
# conversion three nights running. The extraction was perfect -- 156,679 clean
# characters off a born-digital pdfTeX file -- but the paper says "B2B" or
# "B2C" 37 times, every one a letter-digit-letter match under IGNORECASE.
# Score 0.00, hard fail, and the failure told the operator to re-run it
# through OCR, which would have degraded a clean text layer for no reason.
_DIGIT_ACRONYM_RE = re.compile(
    r'\b(?:b2b|b2c|b2g|c2c|d2c|m2m|o2o|p2p)\b', re.IGNORECASE
)

_ARTIFACT_PATTERNS = (
    # Letter-digit-letter: e.g. "managen1ent", "con1panies", "n1ethod".
    # Legitimate English WORDS never have this pattern (acronyms do -- see
    # _DIGIT_ACRONYM_RE, stripped before counting).
    re.compile(r'[a-z]\d[a-z]', re.IGNORECASE),
    # "vv" between letters: e.g. "hovvever", "vvriting", "vve".
    # Extraction substitutes "vv" for "w" when the PDF font lacks a ToUnicode
    # map for U+0077. Legit "vv" in English is vanishingly rare.
    re.compile(r'[a-z]vv[a-z]', re.IGNORECASE),
)


def _text_quality_score(text, min_chars=500):
    """Return a 0.0-1.0 quality score for extracted text.

    Higher is better. Computed from the density of known PyMuPDF font-encoding
    artifacts (letter-digit-letter, "vv" between letters) per 10k characters.
    Zero artifacts -> 1.0. 2+ artifacts per 10k chars -> 0.0. Linear between.

    Returns 1.0 when there is too little text to judge (< `min_chars`),
    to avoid flagging short or image-heavy docs.

    The scorer is deterministic: same input -> same score.
    """
    # Strip markdown syntax so page markers and bullets don't dilute the
    # character-count denominator with non-content chars.
    cleaned = _MARKDOWN_STRIP_RE.sub(' ', text)
    n_chars = len(cleaned)

    # Remove digit-acronyms before counting artifacts, but AFTER measuring the
    # denominator: they are real characters of real text, so they belong in
    # n_chars. Only their contribution to the artifact NUMERATOR is wrong.
    cleaned = _DIGIT_ACRONYM_RE.sub(' ', cleaned)

    # Too little text to judge -- refuse to flag.
    if n_chars < min_chars:
        return 1.0

    artifacts = sum(
        len(pattern.findall(cleaned)) for pattern in _ARTIFACT_PATTERNS
    )
    density_per_10k = artifacts * 10000 / n_chars

    # Score: 1.0 at zero density, 0.0 at 2.0+ per 10k chars, linear between.
    # The 2.0 ceiling is calibrated against real data:
    #   Human Side (good pymupdf): 0.0 per 10k
    #   Functions of Executive (OCR): 0.0 per 10k
    #   Pyramid Principle (bad pymupdf): ~19.5 per 10k (n1 + vv combined)
    # Any density above ~1.0 means the extraction is broken.
    return max(0.0, 1.0 - density_per_10k / 2.0)


# OCR-specific error shapes. These almost never appear in clean English.
_OCR_ERROR_PATTERNS = (
    # Consonant-only run of 4+ letters: "SWIX", "BARBAIA" (note: 4+ to avoid USA/NFL).
    re.compile(r'\b[B-DF-HJ-NP-TV-Z]{4,}\b'),
    # "Ine." — OCR mistakes "Inc." for "Ine.". Very common.
    re.compile(r'\bIne\.\b'),
    # Letter-digit-letter (same as _ARTIFACT_PATTERNS but counted as warning).
    re.compile(r'[a-z]\d[a-z]', re.IGNORECASE),
)


def _ocr_quality_warnings(text):
    """Return a list of human-readable warnings about OCR output quality.

    Unlike _text_quality_score (which produces a 0..1 score for hard-gating),
    these warnings are informational. They get surfaced in the sidecar
    report so the user knows a given OCR run was noisy.
    """
    cleaned = _MARKDOWN_STRIP_RE.sub(' ', text)
    n_chars = len(cleaned)
    if n_chars < 500:
        return []

    counts = {}
    for p in _OCR_ERROR_PATTERNS:
        counts[p.pattern] = len(p.findall(cleaned))

    warnings = []
    for name, count in counts.items():
        density = count * 10000 / n_chars
        if density >= 2.0:
            warnings.append(
                f"OCR quality: high density of artifacts matching {name!r} "
                f"({count} matches, {density:.1f} per 10k chars)"
            )
    return warnings


def _is_structural_line(stripped):
    """Check if a line is a structural markdown element that should not be joined."""
    if not stripped:
        return True
    if stripped.startswith('<!-- Page'):
        return True
    if stripped.startswith('#'):
        return True
    if stripped.startswith('>'):
        return True
    if stripped.startswith('- ') or stripped.startswith('* '):
        return True
    if stripped.startswith('|'):
        return True
    if re.match(r'^\d+[\.\)]\s', stripped):
        return True
    if stripped.startswith('```') or stripped.startswith('---') or stripped.startswith('***'):
        return True
    # ALL-CAPS lines are likely headings
    if re.match(r'^[A-Z][A-Z\s]{5,}$', stripped):
        return True
    return False


def _fix_ligatures(text):
    """Replace PDF ligature characters and fix split-word artifacts.

    PDF extraction often produces ligature characters (fi, ff, fl, ffi, ffl)
    and sometimes splits the word around them with a space (e.g. "eﬀ ective"
    becomes "eff ective" after ligature replacement). This function handles
    both problems.
    """
    # Replace ligature characters (order matters: longer first)
    text = text.replace('\ufb03', 'ffi')
    text = text.replace('\ufb04', 'ffl')
    text = text.replace('\ufb01', 'fi')
    text = text.replace('\ufb00', 'ff')
    text = text.replace('\ufb02', 'fl')

    # Fix split-word artifacts: "fi rst" -> "first", "eff ective" -> "effective"
    text = re.sub(r'(\w*(?:fi|ff|fl))\s+([a-z]{1,8})\b', r'\1\2', text)

    # Fix "Th e" -> "The" (common ligature-adjacent artifact)
    text = re.sub(r'\bTh\s+e\b', 'The', text)
    text = re.sub(r'\bth\s+e\b', 'the', text)

    # Fix soft hyphens with following whitespace/newline
    text = re.sub(r'\u00ad\s*\n\s*', '', text)
    text = re.sub(r'\u00ad\s+', '', text)

    return text


# --- Missing-space fix for words jammed together by PDF extraction ---

_OFF_REAL = re.compile(
    r'^off(er|ers|ered|ering|erings|ice|ices|icer|icers|icial|ials|ially|'
    r'set|sets|line|end|ends|ended|ender|enders|ending|ense|enses|ensive|'
    r'spring|beat|hand|load|shore|side|stage|season|shoot|shoots|'
    r'ish|putting|ramp|screen|site|track|year)$', re.I
)
_STUFF_REAL = re.compile(r'^stuff(ed|ing|ings|s|y|ier|iest)$', re.I)
_SELF_REAL = re.compile(
    r'^self(ish|ishly|ishness|less|lessness|lessly|same|dom|hood)$', re.I
)


def _fix_missing_spaces(text):
    """Fix words jammed together by PDF extraction.

    PDF text extraction sometimes drops the space between words, especially
    after "off", "stuff", and "self" + following word. This function splits
    them: "offthe" -> "off the", "stuffin" -> "stuff in",
    "selfprotection" -> "self-protection".
    """
    def _fix_off(m):
        full = m.group(0)
        if _OFF_REAL.match(full):
            return full
        return 'off ' + full[3:]

    def _fix_stuff(m):
        full = m.group(0)
        if _STUFF_REAL.match(full):
            return full
        return 'stuff ' + full[5:]

    def _fix_self(m):
        full = m.group(0)
        if _SELF_REAL.match(full):
            return full
        return 'self-' + full[4:]

    text = re.sub(r'\boff[a-z]+', _fix_off, text)
    text = re.sub(r'\bstuff[a-z]+', _fix_stuff, text)
    text = re.sub(r'\bself[a-z]+', _fix_self, text)
    return text


# Curated wordlist for splitting joined all-caps headings. Focuses on
# words common in book titles: articles, prepositions, conjunctions, and
# vocabulary found on title pages / chapter headings.
_JOINED_CAPS_WORDS = frozenset(w.lower() for w in (
    # articles, prepositions, conjunctions, pronouns
    "a an the and or but of in on at by for to from with without into onto "
    "over under up down out off as is are was were be been being am i you "
    "we us our your their his her its this that these those if then than "
    "so not no yes all any some each every other another such which who "
    "what when where why how also"
).split() + [
    # title-page vocabulary
    "annotated", "revised", "updated", "expanded", "enlarged", "edition",
    "introduction", "foreword", "preface", "acknowledgments", "contents",
    "chapter", "part", "section", "appendix", "index", "bibliography",
    "copyright", "published", "volume", "series", "reissue", "first",
    "second", "third", "fourth", "fifth", "complete", "abridged",
    "unabridged", "illustrated", "author", "authors", "editor", "editors",
    "translated", "translator", "publisher",
    # common book / management title words
    "human", "side", "enterprise", "management", "leadership", "organization",
    "organizations", "organizational", "behavior", "science", "business",
    "history", "culture", "economy", "economics", "theory", "practice",
    "principles", "method", "methods", "system", "systems", "model",
    "models", "analysis", "study", "research", "guide", "handbook",
    "overview", "fundamentals", "essentials", "foundations", "thinking",
    "decision", "making", "strategy", "strategic", "tactical", "innovation",
    "design", "development", "growth", "change", "transition", "work",
    "workplace", "team", "teams", "group", "groups", "people", "person",
    "leader", "leaders", "manager", "managers", "founder", "founders",
    "company", "companies", "corporation", "project", "process", "product",
    "quality", "productivity", "performance", "operations", "crisis",
    "modern", "contemporary", "global", "local", "drive", "motivate",
    "motivation", "inspire", "inspiration", "purpose", "new", "old",
    # proper nouns commonly appearing in title pages
    "mcgregor", "deming", "pink", "daniel", "douglas", "edwards",
])


def _segment_joined_caps(word):
    """Split a run of joined letters into dictionary words via DP.

    Returns a list of segments if a valid segmentation is found, else None.
    A valid segmentation uses only words from the curated list, prefers
    fewer segments, and requires at least two segments.
    """
    n = len(word)
    if n < 6:
        return None

    lower = word.lower()
    # dp[i] = (segment_count, prev_index, matched_segment) for best split of word[:i]
    dp = [None] * (n + 1)
    dp[0] = (0, -1, None)

    for i in range(1, n + 1):
        # Try every possible previous boundary j < i where word[j:i] is a word.
        for j in range(max(0, i - 15), i):
            if dp[j] is None:
                continue
            candidate = lower[j:i]
            if len(candidate) < 2:
                # Only allow 'a' and 'i' as standalone single-char words
                if candidate not in ("a", "i"):
                    continue
            if candidate not in _JOINED_CAPS_WORDS:
                continue
            count = dp[j][0] + 1
            if dp[i] is None or count < dp[i][0]:
                dp[i] = (count, j, word[j:i])

    if dp[n] is None:
        return None

    # Reconstruct
    segments = []
    idx = n
    while idx > 0:
        _, prev, seg = dp[idx]
        segments.append(seg)
        idx = prev
    segments.reverse()

    # Require at least two segments and at most one single-char segment
    if len(segments) < 2:
        return None
    if sum(1 for s in segments if len(s) == 1) > 1:
        return None
    return segments


def _split_joined_caps(text):
    """Split joined ALL-CAPS words on heading-like lines.

    PyMuPDF sometimes extracts letter-spaced (tracked-out) headings with
    spacing collapsed, producing "THEHUMANSIDE" from "T H E  H U M A N  S I D E".
    Finds short, mostly-uppercase lines containing a run of joined letters
    and splits them using a curated English wordlist.

    Only applies to lines that look like headings (short, standalone,
    mostly uppercase) to avoid corrupting body prose.
    """
    join_pattern = re.compile(r'\b[A-Z]{6,}\b')

    def _rewrite_line(line):
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            return line
        alpha_chars = [c for c in stripped if c.isalpha()]
        if not alpha_chars:
            return line
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio < 0.8:
            return line

        def _sub(match):
            word = match.group(0)
            segments = _segment_joined_caps(word)
            if segments is None:
                return word
            return ' '.join(segments)

        return join_pattern.sub(_sub, line)

    return '\n'.join(_rewrite_line(l) for l in text.split('\n'))


def _collapse_spaced_letters(text):
    """Collapse spaced-out letter artifacts back into words.

    PDF extraction sometimes produces headings like "H e a d i n g" from
    custom-spaced font rendering. Detects sequences of single letters
    separated by spaces and collapses them.
    """
    def _collapse_match(m):
        collapsed = m.group(0).replace(' ', '')
        # Only collapse if result is at least 3 chars (avoid false positives)
        if len(collapsed) >= 3:
            return collapsed
        return m.group(0)

    # Match sequences of single letters separated by 1-3 spaces (or thin/en spaces)
    # At least 4 single-letter groups to avoid false positives like "a b"
    text = re.sub(r'\b[A-Za-z](?:[\s\u2002\u2003\u2009]{1,3}[A-Za-z]){3,}\b', _collapse_match, text)
    return text


def _strip_running_headers(pages_text):
    """Detect and remove running headers and footers from page texts.

    Analyzes the first and last few lines of each page to find repeated
    patterns (book title, chapter name, page numbers) that appear across
    multiple pages. Strips them from the body text.

    Args:
        pages_text: list of (page_num, raw_text) tuples

    Returns:
        list of (page_num, cleaned_text, page_printed) tuples, where
        page_printed is the printed page number captured from the running
        header/footer as a string ("47", "xii"), or None when the page
        carries none.

    Stripping and capture are deliberately separate concerns. What gets
    removed from the body text is unchanged from the pre-page_printed
    behavior; what we are willing to *believe* is a printed page number is
    a strictly narrower set. A page_printed value we emit is presented to
    a reader as the number printed on that page, so a wrong one is worse
    than none at all.

    Page-printed candidates are collected as (rank, value) and the lowest
    rank wins:

        rank 0 — a standalone number on its own line near the page bottom
        rank 1 — a standalone number on its own line near the page top

    There is deliberately no rank for "a number trailing the last line of
    body text". Such a number is still stripped (it is usually a footer
    that extraction glued onto the preceding line), but it is never
    captured: a page whose last sentence legitimately ends in a year would
    otherwise publish "1999" as its printed page number.
    """
    if len(pages_text) < 5:
        return [(page_num, text, None) for page_num, text in pages_text]

    # Collect the first few and last few lines from each page
    # to detect running headers/footers regardless of position
    from collections import Counter

    top_lines = []   # (normalized_line, raw_line) from top of each page
    bottom_lines = []

    def _normalize_header(line):
        """Strip page numbers and whitespace to find the repeating core."""
        s = re.sub(r'^\d+\s*', '', line)       # leading page number
        s = re.sub(r'\s*\d+$', '', s)           # trailing page number
        s = re.sub(r'\s*\|\s*\w*$', '', s)      # trailing "| page" (O'Reilly style)
        s = re.sub(r'^\w+\s*\|\s*', '', s)      # leading "page |" (O'Reilly style)
        return s.strip()

    for _, text in pages_text:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Check up to first 3 lines for header patterns
        for l in lines[:3]:
            top_lines.append(_normalize_header(l))
        # Check last 3 lines for footer patterns
        for l in lines[-3:]:
            bottom_lines.append(_normalize_header(l))

    top_counts = Counter(n for n in top_lines if n and len(n) > 3)
    bottom_counts = Counter(n for n in bottom_lines if n and len(n) > 3)

    # A header/footer pattern must appear on at least 15% of pages
    threshold = max(3, len(pages_text) * 0.15)

    header_patterns = {pat for pat, count in top_counts.items() if count >= threshold}
    footer_patterns = {pat for pat, count in bottom_counts.items() if count >= threshold}

    log.debug("Detected %d header pattern(s), %d footer pattern(s)",
              len(header_patterns), len(footer_patterns))
    for p in header_patterns:
        log.debug("  Header: %r", p)
    for p in footer_patterns:
        log.debug("  Footer: %r", p)

    # Also detect inline header patterns: "123 Book Title Text continues..."
    # where the page number + title is prepended to the first paragraph
    inline_header_patterns = set()
    for pat in header_patterns:
        if pat:
            inline_header_patterns.add(pat)
    # Also detect the book title from the majority of first-line content
    # even if it wasn't standalone (embedded in first paragraph)
    first_line_content = []
    for _, text in pages_text:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            first_line_content.append(lines[0])

    # Look for a common title prefix: "NNN Title Text rest of paragraph"
    title_prefix_counts = Counter()
    for line in first_line_content:
        # Try to extract "NNN Title Words" from start of line
        m = re.match(r'^(\d{1,4})\s+(.+?)(?:\s{2,}|\s+[A-Z][a-z])', line)
        if m:
            candidate = m.group(2).strip()
            if len(candidate) > 5:
                title_prefix_counts[candidate] += 1

    inline_title_prefixes = {
        pat for pat, count in title_prefix_counts.items()
        if count >= threshold
    }
    for p in inline_title_prefixes:
        log.debug("  Inline title prefix: %r (appears %d times)", p, title_prefix_counts[p])

    # Also detect standalone page-number lines (just a number, maybe with spaces)
    # Always strip these, even if no header/footer patterns were found
    standalone_pagenum = re.compile(r'^\s*\d{1,4}\s*$')
    # Also match standalone roman numeral page numbers (front matter).
    # NOTE: this is the STRIPPING test only, and it is intentionally loose
    # (a bag of roman letters). Loosening or tightening it changes what
    # disappears from the body text across every already-converted book.
    # Whether a stripped line is believed to BE a printed page number is
    # decided separately, by _is_roman_page_number + the two-sample
    # confirmation below.
    roman_pagenum = re.compile(
        r'^\s*[ivxlc]{1,7}\s*$', re.I
    )

    result = []
    for page_num, text in pages_text:
        lines = text.split('\n')
        cleaned = []
        page_printed_candidates = []   # (rank, value)
        for j, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue

            norm = _normalize_header(stripped)

            # Check position relative to page boundaries
            non_empty_indices = [k for k, l in enumerate(lines) if l.strip()]
            if not non_empty_indices:
                cleaned.append(line)
                continue
            pos_in_nonempty = non_empty_indices.index(j) if j in non_empty_indices else -1
            is_near_top = pos_in_nonempty >= 0 and pos_in_nonempty < 3
            is_near_bottom = pos_in_nonempty >= 0 and pos_in_nonempty >= len(non_empty_indices) - 3
            is_first = pos_in_nonempty == 0
            is_last = pos_in_nonempty == len(non_empty_indices) - 1

            # Strip if it matches a detected header pattern near top of page
            if is_near_top and norm in header_patterns:
                log.debug("  Stripping header on page %d: %r", page_num, stripped)
                continue
            # Strip if it matches a detected footer pattern near bottom of page
            if is_near_bottom and norm in footer_patterns:
                log.debug("  Stripping footer on page %d: %r", page_num, stripped)
                continue

            # Strip inline title prefix from first content line: "123 Book Title rest..."
            if is_near_top and inline_title_prefixes:
                for prefix in inline_title_prefixes:
                    pat = re.compile(r'^\d{1,4}\s+' + re.escape(prefix) + r'\s+')
                    m = pat.match(stripped)
                    if m:
                        stripped = stripped[m.end():]
                        line = stripped
                        log.debug("  Stripped inline header on page %d: prefix=%r", page_num, prefix)
                        break

            # Capture-then-strip standalone page numbers (arabic or roman).
            # This is the printed page number — the only address that is
            # valid for citation. It used to be discarded here.
            #
            # The strip condition is unchanged. The capture condition is
            # narrower: an arabic run is taken at face value, but a roman
            # line must parse as a real roman numeral (so "ill" and "civil"
            # are stripped as before and captured never).
            if (is_near_top or is_near_bottom) and (
                standalone_pagenum.match(stripped) or roman_pagenum.match(stripped)
            ):
                if standalone_pagenum.match(stripped):
                    page_printed_candidates.append((0 if is_near_bottom else 1, stripped))
                elif _is_roman_page_number(stripped.strip()):
                    page_printed_candidates.append((0 if is_near_bottom else 1, stripped))
                continue

            # Strip "CHAPTER TITLE | page" or "page | BOOK TITLE" running headers
            # (common in O'Reilly and similar publishers)
            if (is_near_top or is_near_bottom) and re.match(
                r'^(?:\d{1,4}\s*\|\s*[A-Z].*|[A-Z][A-Z\s\',]+\|\s*\d{1,4})$', stripped
            ):
                log.debug("  Stripping pipe header/footer on page %d: %r", page_num, stripped)
                continue

            # Strip a trailing page number appended to the last line. This
            # is a stripping rule ONLY — see the docstring. The number is
            # NOT a page_printed candidate: the last line of a page
            # legitimately ends in a number often enough ("...published in
            # 1999") that capturing here publishes numbers that were never
            # printed, poisons the offset derivation, and inflates
            # page_printed_count.
            if is_last:
                m_trail = re.search(r'\s+(\d{1,4})\s*$', line)
                if m_trail:
                    line = line[:m_trail.start()]

            cleaned.append(line)
        result.append((page_num, '\n'.join(cleaned), page_printed_candidates))

    # Roman page numbers need corroboration. Real front matter runs several
    # numbered pages, so >= 2 distinct roman captures across the document
    # is cheap evidence that we are looking at a numbering sequence. A lone
    # "I" or "Li" is far more likely to be English prose that survived as
    # its own line, and emitting it would flip page_numbering to "printed"
    # on a book that has no printed page numbers at all.
    roman_values = {
        value for _, _, cands in result
        for _, value in cands
        if not standalone_pagenum.match(value)
    }
    romans_confirmed = len(roman_values) >= 2
    if roman_values and not romans_confirmed:
        log.debug("Discarding %d unconfirmed roman page-number capture(s): %r",
                  len(roman_values), sorted(roman_values))

    finalized = []
    for page_num, cleaned_text, cands in result:
        if not romans_confirmed:
            cands = [c for c in cands if standalone_pagenum.match(c[1])]
        # Key on rank only: min() is stable, so equal ranks resolve to the
        # first candidate encountered. Comparing whole tuples would break
        # ties lexicographically by value ("12" < "47"), which is wrong.
        page_printed = min(cands, key=lambda c: c[0])[1] if cands else None
        finalized.append((page_num, cleaned_text, page_printed))

    return finalized


# Real roman-numeral grammar, not a bag of roman letters. The lookahead
# rejects the empty match that every group-optional alternative allows.
# This is what separates "xii" (a printed page number) from "ill" and
# "civil" (English words that the loose stripping regex also matches).
_ROMAN_PAGE_NUMBER_RE = re.compile(
    r'^(?=[ivxlcdm])m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$',
    re.I,
)


def _is_roman_page_number(page_printed):
    """True only for a well-formed roman numeral like "xii".

    The stripping regex is a letter-bag (`[ivxlc]{1,7}`) that also matches
    English words — "I", "ill", "civil", "Li", "lix". Stripping those was
    harmless; emitting them as printed page numbers is not, so capture
    validates against the actual grammar.

    Grammar alone still admits "I" (a legitimate roman 1, and also the
    English pronoun). That residual ambiguity is resolved separately, by
    requiring at least two distinct roman captures in the document before
    any of them counts.
    """
    return bool(page_printed) and bool(_ROMAN_PAGE_NUMBER_RE.match(page_printed))


def _is_arabic_page_number(page_printed):
    """True only for a plain ASCII decimal page number like "47".

    `str.isdigit()` is too permissive: it accepts superscripts ("²", a
    footnote marker that can reach page_printed_candidates) where `int()`
    then raises ValueError and aborts the whole conversion, and it accepts
    Arabic-Indic digits, which are not a numbering sequence we can safely
    interpolate against. `.isascii() and .isdecimal()` admits exactly the
    characters `int()` will accept as a base-10 page number.
    """
    return bool(page_printed) and page_printed.isascii() and page_printed.isdecimal()


def _arabic_page_pdf_indices(page_printed_by_pdf_index):
    """PDF page indices carrying an arabic captured page number — the
    interpolation window.

    Interpolation is permitted only BETWEEN captured samples, so the caller
    clamps to the closed interval [min, max] of these PDF page indices.
    Roman samples are excluded here for the same reason they are excluded
    from offset derivation: they belong to a separate numbering sequence
    and must not widen the arabic window.
    """
    return [
        page_pdf for page_pdf, page_printed in page_printed_by_pdf_index.items()
        if _is_arabic_page_number(page_printed)
    ]


def _page_offset_samples(page_printed_by_pdf_index):
    """dict of PDF page index -> implied offset, arabic samples only."""
    return {
        page_pdf: int(page_printed) - page_pdf
        for page_pdf, page_printed in page_printed_by_pdf_index.items()
        if _is_arabic_page_number(page_printed)
    }


def _derive_page_offset(page_printed_by_pdf_index):
    """Derive a constant page_pdf->page_printed offset from captured samples.

    Returns (offset, is_consistent). `offset` is page_printed - page_pdf.
    Consistency requires at least 3 arabic samples that overwhelmingly
    agree; a book that renumbers partway through (part-openers restarting
    at 1, roman-to-arabic front matter) will disagree in a way this refuses,
    rather than inventing page numbers. Roman page numbers never
    participate — they belong to a separate numbering sequence.

    Unanimity is not special-cased: a unanimous set is simply one with 100%
    agreement and no dissenters, so `_dominant_page_offset` returns it by
    the same path. Callers that need to know WHICH samples dissented use
    `_page_offset_outliers`; callers that need to know WHY a set was
    refused use `_page_offset_refusal`.

    Args:
        page_printed_by_pdf_index: dict of PDF page index -> printed page
            number string

    Returns:
        (int | None, bool)
    """
    by_index = _page_offset_samples(page_printed_by_pdf_index)
    if len(by_index) < _OFFSET_MIN_SAMPLES:
        return (None, False)
    dominant, _ = _dominant_page_offset(by_index)
    return (dominant, True) if dominant is not None else (None, False)


def _page_offset_refusal(page_printed_by_pdf_index):
    """Why interpolation was refused, in one operator-readable clause.

    Returns None when an offset WAS derived. The single worst failure mode
    this module has shipped is a book emerging with 14% page coverage and an
    empty `warnings` list: the operator sees a bad number and no account of
    it, and cannot tell a sparse capture from a refused consensus. Every
    refusal path here has a sentence attached for that reason.
    """
    by_index = _page_offset_samples(page_printed_by_pdf_index)
    if len(by_index) < _OFFSET_MIN_SAMPLES:
        return (
            f"only {len(by_index)} arabic printed page number(s) were "
            f"captured; {_OFFSET_MIN_SAMPLES} are needed to establish an offset"
        )
    _, reason = _dominant_page_offset(by_index)
    return reason


# A lone sample disagreeing with hundreds of others is an OCR misread of one
# digit, not evidence that the book renumbers. Boyatzis printed 9 and 39 on
# two pages that marker read as "0" and "30"; under strict unanimity those
# two misreads both blocked interpolation for 42 other pages AND were
# published verbatim as page numbers — the exact harm the locator work
# exists to prevent.
#
# Rejecting an outlier is only safe when the consensus is overwhelming and
# the dissent is scattered. A book that genuinely renumbers (endnotes
# restarting at 1, a second sequence after an insert) produces a CONTIGUOUS
# RUN of samples at the new offset, never isolated singletons. That
# distinction is what keeps this from silently flattening a real second
# numbering sequence into a wrong one.
#
# The agreement bar was 0.95, which made any dissent at all require 20+
# samples. A home scan captures folios sparsely — the folio usually sits
# beside a running head and is read as part of it — so 20 samples is a bar
# a whole class of real books cannot clear. Landsberg's Tao of Coaching
# captured 18 folios; 17 agreed on -9 across the entire body and the
# eighteenth was a numbered-list item lifted off an appendix opener. At
# 0.95 that book (94.4% agreement) was refused and shipped with 14%
# coverage and no printed page numbers at all.
#
# 0.85 is chosen over the 0.80 first cut proposed in issue #28 because 0.80
# admits a case this module already refuses on purpose: 7 scattered
# dissenters in 40 samples (82.5%) that all land on the SAME alternative
# offset. That is a competing numbering hypothesis, not seven independent
# misreads, and the run guard cannot catch it because the dissenters are
# spread out. 0.85 is the loosest bar that admits Landsberg while keeping
# every disagreement pattern the existing tests pin down.
_OFFSET_CONSENSUS_MIN_AGREEMENT = 0.85
# Below this many samples, unanimity is required. At 0.95 a minimum-samples
# guard was unreachable (the ratio always bit first) and was deliberately
# left out; at 0.85 it bites, and it should: a 6-to-1 split is not a
# consensus with an outlier, it is a small sample where a genuine second
# sequence and an OCR misread are indistinguishable.
_OFFSET_DISSENT_MIN_SAMPLES = 8
# Fewer than this many arabic samples and no offset is derived at all.
_OFFSET_MIN_SAMPLES = 3
# Two dissenters within this many sheets of each other count as a run.
_OFFSET_DISSENT_RUN_GAP = 2


def _dominant_page_offset(offset_by_pdf_index):
    """Return (consensus offset, refusal reason).

    Exactly one of the two is None: an adopted offset carries no reason, a
    refusal carries no offset. The reason exists so callers can tell the
    operator WHY a book lost its page numbering instead of leaving an empty
    `warnings` list next to bad coverage.

    Args:
        offset_by_pdf_index: dict of PDF page index -> (printed - pdf)

    Returns:
        (int | None, str | None)
    """
    counts = collections.Counter(offset_by_pdf_index.values())
    dominant, agreeing = counts.most_common(1)[0]
    total = len(offset_by_pdf_index)
    dissenting = total - agreeing
    if agreeing / total < _OFFSET_CONSENSUS_MIN_AGREEMENT:
        return (None, (
            f"the best-supported offset {dominant:+d} is carried by only "
            f"{agreeing} of {total} captured folios "
            f"({agreeing / total:.0%}, below the "
            f"{_OFFSET_CONSENSUS_MIN_AGREEMENT:.0%} consensus bar), so the "
            f"disagreement is too broad to be OCR noise"
        ))
    if dissenting and total < _OFFSET_DISSENT_MIN_SAMPLES:
        return (None, (
            f"{agreeing} of {total} captured folios agree on offset "
            f"{dominant:+d}, but under {_OFFSET_DISSENT_MIN_SAMPLES} samples "
            f"a dissenting folio cannot be told apart from a second "
            f"numbering sequence, so unanimity is required"
        ))
    dissenters = sorted(i for i, o in offset_by_pdf_index.items() if o != dominant)
    # Any two dissenters close together are a numbering change, not noise.
    for earlier, later in zip(dissenters, dissenters[1:]):
        if later - earlier <= _OFFSET_DISSENT_RUN_GAP:
            return (None, (
                f"pdf pages {earlier} and {later} both disagree with offset "
                f"{dominant:+d} and are within {_OFFSET_DISSENT_RUN_GAP} "
                f"sheets of each other; adjacent dissent is the signature of "
                f"a renumbering or a page-order defect, not an OCR misread"
            ))
    return (dominant, None)


def _page_offset_outliers(page_printed_by_pdf_index, offset):
    """PDF page indices whose captured folio contradicts the derived offset.

    These are misreads. They must be suppressed rather than emitted: a
    captured value is normally the most trustworthy kind of address, so a
    corrupted one is published with full confidence and a reader cannot tell.
    Suppressing lets interpolation supply the right number instead.
    """
    if offset is None:
        return set()
    return {
        page_pdf
        for page_pdf, page_printed in page_printed_by_pdf_index.items()
        if _is_arabic_page_number(page_printed)
        and int(page_printed) - page_pdf != offset
    }


# What kind of page address each backend can produce. `pymupdf` and `marker`
# are absent because they decide at runtime (printed when printed page
# numbers were captured, pdf_only otherwise) — see convert_with_pymupdf and
# convert_with_marker.
BACKEND_PAGE_NUMBERING = {
    "ocr": "pdf_only",
    "pymupdf4llm": "pdf_only",
    "pandoc": "none",
    "docling": "none",
}


# How much support the best offset needs before a book that FAILED the
# consistency check may still be called page-citable.
#
# Chosen from the corpus, not tuned. Of the 22 conversions whose sidecars said
# `page_numbering: printed` with `offset_consistent: false` on 2026-09-01, the
# support for the dominant offset was: 3%, then 54, 57, 67, 75, 77, 79, 84,
# 84%, and the rest refused for adjacent dissent having already cleared the 85%
# bar. There is a 51-point gap between the first and the second. 0.5 sits in
# it, and the rule it expresses is "the captured folios do not cohere AT ALL",
# not "this book renumbers somewhere".
_PAGE_NUMBERING_MIN_OFFSET_SUPPORT = 0.5


def _page_offset_support(page_printed_by_pdf_index):
    """Fraction of captured arabic folios carrying the best-supported offset.

    Returns None when there are too few samples to speak of support at all.
    `_dominant_page_offset` computes this number to decide consensus and then
    discards it; the verdict needs it, so it is computed here rather than
    parsed back out of the warning prose.
    """
    by_index = _page_offset_samples(page_printed_by_pdf_index)
    if len(by_index) < _OFFSET_MIN_SAMPLES:
        return None
    counts = collections.Counter(by_index.values())
    _, agreeing = counts.most_common(1)[0]
    return agreeing / len(by_index)


def _decide_page_numbering(captured, offset_consistent, offset_support=None):
    """What kind of page address can this conversion honestly produce?

    "printed" is a LICENCE: mc-wiki's `tools/cite.py` refuses to cite a source
    by page unless this reads "printed", so the value decides whether page
    numbers out of this book can reach a citation.

    It used to be granted on `captured > 0` alone -- any folio anywhere earned
    it, however incoherent the pagination. Found 2026-09-01 by the annotated
    edition of *The Human Side of Enterprise*: 30 folios across 480 sheets
    (coverage 0.062), the dominant offset carried by 1 of 30 (3%), and the
    captured values reading sheet 56 -> "1", 72 -> "2", 85 -> "4", 99 -> "3".
    Those are the annotated edition's MARGINAL ANNOTATION NUMBERS being read as
    folios. Filing on that verdict would have manufactured the same
    fabricated-page-number hazard that `thiel-zero-to-one` was stamped to
    correct the same night, where a chapter cited at p. 154 begins at p. 118.

    THE FIX IS DELIBERATELY NARROW, and the first draft of it was not. Refusing
    every `offset_consistent: false` book would have demoted 22 sources,
    including *The Achievement Motive* (340 folios, 89.5% coverage) and
    *Scaling People* (99.2%). Those are not incoherent -- they renumber
    somewhere, or carry two sequences, which is ordinary and which
    `page_correspondence_checked` already exists to adjudicate. Only a book
    whose folios agree with each other almost nowhere loses the licence.

    `page_printed_count` and `page_printed_coverage` are untouched, so
    "captured folios that do not cohere" stays readable as count > 0 with
    numbering == "pdf_only".

    THREE BOUNDARIES THIS DELIBERATELY DOES NOT CROSS. All three were raised in
    review; each is recorded here rather than silently left, because the next
    person to read this will have the same three questions.

    1. Fewer than three captured folios keeps the old verdict (support is
       None). With no derived offset nothing is interpolated -- the few
       markers present are CAPTURED values, not manufactured ones -- so a
       book with one real folio genuinely does carry one real page number.
       Demoting on sparseness is a different judgement about a different
       defect, and it would move four more corpus sources on a question this
       change did not investigate.

    2. Support is measured over the captured-folio population, which is not
       identical to `page_printed_count` (that counts EMITTED locators, after
       blank and broken-TOC pages are dropped). That is deliberate: support
       exists to describe the same evidence `_derive_page_offset` judged, and
       judging it against a different population would make the number
       disagree with the decision it explains.

    3. A global modal share is a blunt coherence test. A volume with three
       equally-sized numbering sequences scores ~0.33 and is demoted. That is
       the intended reading rather than a false positive: in such a volume
       "p. 50" does not identify a page without naming the sequence, so it is
       not citable by page number alone. `page_correspondence_checked` is the
       human override for a book where that judgement is wrong.
    """
    if not captured:
        return "pdf_only"
    if offset_consistent:
        return "printed"
    # Inconsistent, but with too few samples to judge support: leave the
    # existing verdict alone rather than widen this change into sparse-capture
    # territory, which is a different defect.
    if offset_support is None:
        return "printed"
    if offset_support < _PAGE_NUMBERING_MIN_OFFSET_SUPPORT:
        return "pdf_only"
    return "printed"


def _apply_backend_page_numbering(report):
    """Stamp a backend's fixed locator capability onto its report.

    Backends that cannot produce a printed page number must declare it, so
    the ingestion gate can route away from them instead of silently filing
    a source that can never be cited by page.
    """
    fixed = BACKEND_PAGE_NUMBERING.get(report.method)
    if fixed:
        report.page_numbering = fixed
    return report


# --- marker page locators --------------------------------------------------
#
# marker is the only backend that works on a scanned book: pymupdf needs a
# text layer the scan does not have, and tesseract has no table handling.
# So marker is precisely the backend a home-scanned print book depends on,
# and it was the one declaring `page_numbering: "none"`.
#
# marker already reads the printed page number — it classifies the running
# head as a PageHeader block and then discards it. Three renderer flags
# recover it: `--paginate_output` emits a `{N}-----` separator per page, and
# `--keep_pageheader_in_output` / `--keep_pagefooter_in_output` stop the
# furniture being dropped. We request all three, capture the folio, and strip
# the furniture back out.
#
# Both bands are required. Book designers put a running head carrying the
# folio on ordinary pages but a *drop folio* at the foot of chapter openers
# and full-page tables — exactly the pages a statistical appendix consists
# of. Reading only the head captured 0 folios across Boyatzis's entire
# appendix while appearing to work.
#
# A rejected alternative: leave marker in its default configuration and read
# folios independently, by OCRing a crop of each page's margins. That makes
# the body provably untouched, which is why it was tried — but a blind crop
# cannot tell a page number from a table cell. On Boyatzis it captured 237
# folios to marker's 288, and the appendix (full-page statistical tables)
# polluted the samples so badly that no offset could be derived at all: the
# book would have ended up with NO interpolated page numbers and 33 wrong
# captured ones. marker's layout model knows what a running head is; a
# rectangle does not.
#
# The separator carries marker's own page index, so pages that produce no
# text still advance the count — the address never drifts.

MARKER_LOCATOR_ARGS = (
    "--paginate_output",
    "--keep_pageheader_in_output",
    "--keep_pagefooter_in_output",
)

# marker's page separator: "{58}------------------------------------------------"
_MARKER_PAGE_SEPARATOR_RE = re.compile(r'^\{(\d+)\}-{2,}\s*$')

# A folio standing alone on its line. Arabic is capped at 4 digits so a year
# ("1982") that survived into the band is admitted here and then rejected by
# the offset check, rather than silently reinterpreted as a page number.
_MARKER_FOLIO_RE = re.compile(r'^(\d{1,4})$')

# A markdown heading is content, never page furniture. marker renders
# furniture as plain text, so a `#` line is by definition not a running head.
_MARKDOWN_HEADING_RE = re.compile(r'^#{1,6}\s')

# How many leading (or trailing) lines of a page may be furniture. A running
# head is one or two lines — a title and a folio; anything deeper is body.
_MARKER_HEADER_SCAN_LINES = 3

# How many pages a line must recur on before it counts as furniture. Verso
# and recto usually carry different heads, so a book alternates between two.
_MARKER_HEADER_MIN_REPEATS = 3

# --- folios sharing a line with a running head -----------------------------
#
# On many editions the folio does not stand alone: it sits on the same line
# as the running head, pinned to the OUTER edge of the spread — leading on a
# verso ("52 THE TAO OF COACHING"), trailing on a recto ("MOTIVATING 67").
# `_MARKER_FOLIO_RE` anchors both ends, so every one of those lines was
# discarded whole. On Landsberg's *The Tao of Coaching* that cost 117 of 136
# pages: the folios that survived were almost all chapter openers, the one
# place the edition sets the folio alone.
#
# The extraction rule is deliberately narrow, because a wrong folio is worse
# than no folio — it is published as the number printed on the page, on the
# one field a reader has no way to check. A numeral is lifted out of a band
# line only when ALL of:
#
#   1. it is the first or the last token on the line (a folio is pinned to
#      the page edge; a number in the middle of a head is part of the title);
#   2. what remains, with the numeral removed, is RECURRING FURNITURE — the
#      same residue appears in the same band on >= _MARKER_HEADER_MIN_REPEATS
#      pages. This is the frequency argument the module already rests on for
#      stripping: a running head repeats, a line of prose does not;
#   3. the line as a whole does NOT recur. A folio changes every page, so a
#      band line that repeats verbatim has a CONSTANT number in it — a year
#      in the title, a part or chapter number — and is furniture entire;
#   4. the residue contains a letter, so "52 1984" (two numerals, no head)
#      is refused rather than guessed at;
#   5. only ONE end of the line qualifies. If stripping the leading numeral
#      and stripping the trailing numeral both leave known furniture, the
#      line is ambiguous and yields nothing.
#
# Roman folios are NOT read out of a shared line. A bag of roman letters
# matches ordinary words ("civil", "mild", "did"), and unlike the arabic
# case there is no digit shape to lean on; roman front matter sets the folio
# alone often enough that the trade is not worth the false positives.
_MARKER_EMBEDDED_FOLIO_LEADING_RE = re.compile(r'^(\d{1,4})\s+(\S.*)$')
_MARKER_EMBEDDED_FOLIO_TRAILING_RE = re.compile(r'^(.*\S)\s+(\d{1,4})$')

# Line shapes that are never a running head: markdown headings and block
# quotes are content by construction, images carry no folio, and a table row
# ends in numbers for reasons that have nothing to do with pagination.
_MARKER_NOT_A_RUNNING_HEAD_RE = re.compile(r'^(?:#{1,6}\s|>|!\[)')

_HAS_LETTER_RE = re.compile(r'[^\W\d_]')


def _embedded_folio_candidates(line):
    """(folio, running-head residue) pairs for a band line, edge tokens only.

    Shape only — this says nothing about whether the residue is furniture.
    Callers must confirm that against the recurring-line set; see
    `_accepted_embedded_folio`.
    """
    stripped = line.strip()
    if not stripped or _MARKER_NOT_A_RUNNING_HEAD_RE.match(stripped):
        return []
    if '|' in stripped:
        return []
    if _is_marker_folio(stripped):
        return []          # a bare folio: the existing path already reads it
    out = []
    m = _MARKER_EMBEDDED_FOLIO_LEADING_RE.match(stripped)
    if m and int(m.group(1)) >= 1 and _HAS_LETTER_RE.search(m.group(2)):
        out.append((m.group(1), m.group(2).strip()))
    m = _MARKER_EMBEDDED_FOLIO_TRAILING_RE.match(stripped)
    if m and int(m.group(2)) >= 1 and _HAS_LETTER_RE.search(m.group(1)):
        out.append((m.group(2), m.group(1).strip()))
    return out


def _accepted_embedded_folio(line, known):
    """The folio in a running-head line, or None if the line is not one.

    `known` is the recurring-furniture set for the band this line came from.
    """
    stripped = line.strip()
    if stripped in known:
        return None        # recurs verbatim: any number in it is constant
    accepted = [folio for folio, residue in _embedded_folio_candidates(stripped)
                if residue in known]
    return accepted[0] if len(accepted) == 1 else None


def _split_marker_pages(text):
    """Split paginated marker output into (page_index, page_text) pairs.

    Returns None when the text carries no separators at all, which means
    marker ran without `--paginate_output` and no page addressing is
    possible. Callers must treat that as "no locators", never as page 0.
    """
    lines = text.split('\n')
    pages, current, index = [], [], None
    for line in lines:
        m = _MARKER_PAGE_SEPARATOR_RE.match(line)
        if m:
            if index is not None:
                pages.append((index, '\n'.join(current)))
            index, current = int(m.group(1)), []
            continue
        current.append(line)
    if index is None:
        return None
    pages.append((index, '\n'.join(current)))
    return pages


def _is_marker_folio(line):
    """Whether a line is shaped like a printed page number.

    Arabic zero is rejected outright: no book prints page 0, so a captured
    "0" is always a misread (Boyatzis's page 9, on a scan). That is a
    validity check, not a judgment call — unlike the consensus rule, it needs
    no evidence from the rest of the book.
    """
    m = _MARKER_FOLIO_RE.match(line)
    if m:
        return int(m.group(1)) >= 1
    return bool(_is_roman_page_number(line))


def _band_lines(page_text, from_end=False):
    """First (or last) few non-empty lines of a page, in reading order."""
    lines = [l for l in page_text.split('\n') if l.strip()]
    return lines[-_MARKER_HEADER_SCAN_LINES:] if from_end else lines[:_MARKER_HEADER_SCAN_LINES]


def _marker_recurring_lines(pages, from_end=False):
    """Return the lines that recur often enough to be page furniture.

    Frequency is the discriminator that makes stripping safe. A book's
    running head appears on many pages; a first line of body prose appears
    once. Requiring recurrence means an unusual opening sentence is never
    mistaken for furniture, at the cost of leaving a head that genuinely
    appears twice — the safe direction to err, since a stray head in the body
    is visible while deleted body text is not.

    Lines are counted twice over: verbatim, and with an edge numeral removed.
    The second form is what lets a head that ALWAYS carries its folio
    ("52 THE TAO OF COACHING", "MOTIVATING 67") be recognised at all — the
    whole line never repeats, but the residue repeats on every page of the
    book. Both forms land in one set; membership means "this string is
    furniture", and the caller decides what to do with the numeral.
    """
    counts = collections.Counter()
    for _index, page in pages:
        seen = set()
        for line in _band_lines(page, from_end):
            stripped = line.strip()
            # Folios differ every page, so they never recur; they are matched
            # by shape, not by frequency.
            if stripped and not _is_marker_folio(stripped):
                seen.add(stripped)
                seen.update(residue for _folio, residue
                            in _embedded_folio_candidates(stripped))
        counts.update(seen)
    return {line for line, n in counts.items() if n >= _MARKER_HEADER_MIN_REPEATS}


def _strip_marker_page_furniture(page_text, header_lines, footer_lines):
    """Remove running head and foot from one page, returning (folio, body).

    Each band is consumed from its own end and stops at the first line that
    is neither known furniture nor a bare folio — the moment body text
    starts, stripping stops.

    Capture and stripping are deliberately decoupled. Capture scans the whole
    band, so a folio sitting *behind* an unrecognised head line is still
    read; stripping removes only lines it positively recognises. They can
    therefore disagree — a folio may be captured and left in the body — and
    that is the safe direction: a stray number in the text is visible,
    whereas deleting a line that turned out to be prose is not.

    A folio standing alone is preferred over one shared with a running head,
    in both bands, before the shared-line rule is tried at all. That ordering
    is what makes this change incapable of altering any book whose folios
    already stand alone: on such a page the first pass has already returned.
    """
    lines = page_text.split('\n')
    head_band = _band_lines(page_text)
    foot_band = _band_lines(page_text, from_end=True)

    def first_folio(band):
        for line in band:
            if _is_marker_folio(line.strip()):
                return line.strip()
        return None

    def first_embedded_folio(band, known):
        for line in band:
            folio = _accepted_embedded_folio(line, known)
            if folio:
                return folio
        return None

    # A page that lists page numbers — a contents or index page — carries
    # several lines that are shaped exactly like a folio and are not one. The
    # standing-alone preference below is right everywhere else and exactly
    # backwards here: on What Works sheet 7 the head band holds both
    # `viii Contents` (the real running head, real folio) and a bare `44` from
    # the listing, and the bare number wins.
    #
    # Plurality is the discriminator, and it is a strong one: a page has one
    # folio, so a second standalone folio-shaped line means at least one of
    # them is not a folio, and nothing about shape says which. Measured over
    # the four books ingested 2026-07-29, no body page anywhere reached two —
    # the rule fires only on the four What Works contents pages and one
    # Landsberg index page, whose captured value interpolation reproduces
    # exactly. So this cannot alter a book that does not list page numbers.
    #
    # When it fires, fall through to the running-head folio, which is
    # corroborated by recurring furniture and is therefore still trustworthy.
    # If there is none, capture nothing and let interpolation fill the page:
    # an honest gap beats a confident wrong number.
    bare_folio_lines = sum(1 for l in page_text.split('\n')
                           if _is_marker_folio(l.strip()))
    lists_page_numbers = bare_folio_lines > 1

    folio = None
    if not lists_page_numbers:
        folio = first_folio(head_band) or first_folio(foot_band)
    folio = (folio
             or first_embedded_folio(head_band, header_lines)
             or first_embedded_folio(foot_band, footer_lines))

    def strippable(line, known):
        stripped = line.strip()
        if _MARKDOWN_HEADING_RE.match(stripped):
            return False
        if _is_marker_folio(stripped) or stripped in known:
            return True
        # A running head that carries its folio is furniture entire: the
        # residue is known, so the line is recognised and goes with it.
        return _accepted_embedded_folio(stripped, known) is not None

    start, consumed = 0, 0
    while start < len(lines) and consumed < _MARKER_HEADER_SCAN_LINES:
        if not lines[start].strip():
            start += 1
        elif strippable(lines[start], header_lines):
            start += 1
            consumed += 1
        else:
            break

    end, consumed = len(lines), 0
    while end > start and consumed < _MARKER_HEADER_SCAN_LINES:
        if not lines[end - 1].strip():
            end -= 1
        elif strippable(lines[end - 1], footer_lines):
            end -= 1
            consumed += 1
        else:
            break

    return folio, '\n'.join(lines[start:end]).strip('\n')


def _marker_page_locators(text):
    """Capture printed page numbers from paginated marker output.

    Returns (pages, page_printed_by_index) where `pages` is a list of
    (page_index, body_text) with page furniture removed, or (None, {}) when
    the output was not paginated.
    """
    pages = _split_marker_pages(text)
    if pages is None:
        return None, {}
    header_lines = _marker_recurring_lines(pages, from_end=False)
    footer_lines = _marker_recurring_lines(pages, from_end=True)
    cleaned, printed = [], {}
    for index, page in pages:
        folio, body = _strip_marker_page_furniture(page, header_lines, footer_lines)
        if folio:
            printed[index] = folio
        cleaned.append((index, body))
    return cleaned, printed


def _merge_split_caps_headings(lines):
    """Merge runs of consecutive short ALL-CAPS lines into single heading lines.

    PyMuPDF sometimes extracts a heading like "THE CASE FOR SECRETS" or
    "THE CHALLENGE OF THE FUTURE" as multiple separate lines ("THE" /
    "CASE FOR SECRETS" or "THE" / "CHALLENGE" / "OF THE" / "FUTURE"),
    one word-or-phrase per line. This happens when the PDF uses
    tracked-out letter spacing or stacked typography for headings.

    We merge any run of 2+ consecutive short (<= 40 char) ALL-CAPS lines
    into a single line. The merged line then flows through _format_headings
    and gets promoted to a markdown heading like every other caps heading.

    Not a perfect signal — two adjacent all-caps proper nouns (e.g.
    "NEW YORK" / "HONG KONG" in a list) could merge — but this shape is
    rare in book body text and common in chapter/section titles, so the
    tradeoff favors merging.
    """
    caps_re = re.compile(r'^[A-Z][A-Z\s:&,\-\']{0,40}$')
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if caps_re.match(stripped) and len(stripped) <= 40:
            run = [stripped]
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if caps_re.match(next_stripped) and len(next_stripped) <= 40:
                    run.append(next_stripped)
                    j += 1
                else:
                    break
            if len(run) >= 2 and sum(len(r) + 1 for r in run) <= 80:
                result.append(' '.join(run))
                i = j
                continue
        result.append(lines[i])
        i += 1
    return result


def _promote_bookmark_headings(text, titles):
    """Promote bookmark titles to markdown headings at the top of a page.

    The PDF bookmark tree (doc.get_toc()) is the author's own declaration
    of where chapters begin and what they're called. When a bookmark points
    at this page, we search the first 20 lines for the title text and
    replace it with a `##` heading. Two modes:

      (1) Multi-line split: PyMuPDF extracted the title across separate
          lines (e.g. "THE" / "CHALLENGE" / "OF THE" / "FUTURE"). We merge
          those lines into a single heading.

      (2) Mid-line prefix: PyMuPDF ran the title into the first paragraph
          as a single long line (e.g. "PARTY LIKE IT'S 1999 OUR CONTRARIAN
          QUESTION—What important truth…"). We split that line at the end
          of the title, promote the title to a heading, and put the rest
          back as body text.

    Matching is whitespace/punctuation-insensitive. Leading numerals
    ("1. ", "Chapter 1: ") are stripped from the bookmark title before
    searching. Match must end at a word boundary so we never bisect a
    real word. Heading text preserves the PDF's original casing.
    """
    if not titles or not text:
        return text

    lines = text.split('\n')
    for title in titles:
        clean_title = re.sub(
            r'^(?:chapter\s+\d+[.:]?\s*|part\s+\d+[.:]?\s*|\d+[.)]\s*)',
            '',
            title.strip(),
            flags=re.IGNORECASE,
        ).strip()
        if not clean_title:
            continue
        target = re.sub(r'[^a-z0-9]', '', clean_title.lower())
        if len(target) < 4:
            continue

        # Walk lines char-by-char through the first ~20 lines, accumulating
        # normalized alnum chars. When accumulation contains the target,
        # record the line and char position where the match ends.
        acc = ''
        first_content_idx = None
        last_content_idx = None
        split_point = None  # char index in last consumed line's stripped form
        matched = False

        for idx in range(min(20, len(lines))):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith('<!-- Page'):
                continue
            if stripped.startswith('#'):
                break
            if first_content_idx is None:
                first_content_idx = idx

            for char_idx, ch in enumerate(stripped):
                if ch.isalnum():
                    acc += ch.lower()
                if target in acc:
                    matched = True
                    last_content_idx = idx
                    split_point = char_idx + 1
                    break
            if matched:
                break
            last_content_idx = idx
            if len(acc) > len(target) + 40:
                break

        if not (matched and first_content_idx is not None):
            continue

        last_line_stripped = lines[last_content_idx].strip()
        # Require match to end at a word boundary — never split a real word.
        if split_point < len(last_line_stripped) and last_line_stripped[split_point].isalnum():
            continue

        # Build heading text from consumed content in the PDF's own casing.
        heading_parts = []
        for idx in range(first_content_idx, last_content_idx):
            stripped = lines[idx].strip()
            if stripped and not stripped.startswith('<!-- Page'):
                heading_parts.append(stripped)
        heading_prefix = last_line_stripped[:split_point].rstrip(' \t.,:;-')
        if heading_prefix:
            heading_parts.append(heading_prefix)
        heading_text = ' '.join(heading_parts).strip()
        if not heading_text:
            continue

        body_suffix = last_line_stripped[split_point:].lstrip(' \t.,:;-')

        new_lines = list(lines[:first_content_idx])
        new_lines.append(f"## {heading_text}")
        for idx in range(first_content_idx + 1, last_content_idx):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith('<!-- Page'):
                new_lines.append(lines[idx])
        if body_suffix:
            new_lines.append('')
            new_lines.append(body_suffix)
        new_lines.extend(lines[last_content_idx + 1:])
        lines = new_lines
        break  # one bookmark promotion per page

    return '\n'.join(lines)


def _format_headings(text):
    """Detect and format section headings as markdown.

    Identifies heading patterns:
    - ALL-CAPS lines (short, standalone)
    - "Chapter N" / "Part N" patterns
    - Short standalone lines before body text

    Runs _merge_split_caps_headings first to join split-caps heading
    fragments (e.g. "THE" / "CASE FOR SECRETS" -> "THE CASE FOR SECRETS")
    so the single-line heading detector below can match them.
    """
    lines = text.split('\n')
    lines = _merge_split_caps_headings(lines)
    result = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines and page markers
        if not stripped or stripped.startswith('<!-- Page'):
            result.append(line)
            continue

        # Already a markdown heading
        if stripped.startswith('#'):
            result.append(line)
            continue

        # "Chapter N" or "Part N" patterns -> ##
        # Max 10 words to avoid promoting full sentences like "Chapter 9 is a wildcard..."
        if (re.match(r'^(Chapter|CHAPTER|Part|PART)\s+(\d+|[IVXLC]+)', stripped, re.I)
                and len(stripped.split()) <= 10):
            result.append(f"## {stripped}")
            continue

        # ALL-CAPS lines that are likely headings (5-80 chars, mostly letters)
        # Only promote if the line is standalone: prev and next lines are
        # empty, page markers, or other structural elements (avoids false
        # positives from inline all-caps text like newspaper headlines)
        if (re.match(r'^[A-Z][A-Z\s\d:,\-&]{4,79}$', stripped)
                and len(stripped) < 80
                and sum(1 for c in stripped if c.isalpha()) > len(stripped) * 0.6):
            prev_stripped = lines[i - 1].strip() if i > 0 else ''
            next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ''
            prev_ok = (not prev_stripped or prev_stripped.startswith('<!-- Page')
                       or prev_stripped.startswith('#'))
            next_ok = (not next_stripped or next_stripped.startswith('<!-- Page')
                       or next_stripped.startswith('#'))
            if prev_ok or next_ok:
                if len(stripped) < 20:
                    result.append(f"### {stripped}")
                else:
                    result.append(f"## {stripped}")
                continue

        result.append(line)

    return '\n'.join(result)


def _format_toc(text):
    """Detect and reformat collapsed table of contents.

    TOC entries often get collapsed into single lines like:
    "Foreword ix Introduction xiii Chapter 1 1"
    This splits them back onto separate lines.
    """
    lines = text.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()

        # Detect collapsed TOC: multiple "Title pagenum" pairs on one line
        # Pattern: word(s) followed by roman numeral or number, repeated 3+ times
        toc_pattern = re.compile(
            r'((?:[A-Z][A-Za-z\s\':,\-]+?)\s+'
            r'(?:[ivxlc]+|\d{1,3}))'
        )
        matches = toc_pattern.findall(stripped)
        if len(matches) >= 3 and len(stripped) > 60:
            # This looks like a collapsed TOC - split entries onto separate lines
            # Use a more precise split: title followed by page number
            entries = re.findall(
                r'([A-Z][A-Za-z\s\':,\-]+?)\s+((?:[ivxlc]+|\d{1,3}))(?=\s+[A-Z]|\s*$)',
                stripped
            )
            if len(entries) >= 3:
                for title, page in entries:
                    result.append(f"- {title.strip()} ... {page}")
                continue

        result.append(line)

    return '\n'.join(result)


def _normalize_bullets(text):
    """Convert Unicode bullet characters to standard markdown list items.

    Handles inline bullet runs (● Item 1 ● Item 2) by splitting them
    onto separate lines, and standalone bullet lines by normalizing the marker.
    """
    bullet_chars = r'[●•◆▪▸►‣⬥]'

    # Split inline bullet runs: "● Item 1 ● Item 2" -> separate lines
    # Only if there are 2+ bullets on the same line
    def _split_inline_bullets(m):
        line = m.group(0)
        items = re.split(r'\s*' + bullet_chars + r'\s*', line)
        items = [item.strip() for item in items if item.strip()]
        if len(items) >= 2:
            return '\n'.join(f'- {item}' for item in items)
        return line

    text = re.sub(
        r'^.*' + bullet_chars + r'.*' + bullet_chars + r'.*$',
        _split_inline_bullets,
        text,
        flags=re.MULTILINE,
    )

    # Normalize standalone bullet lines: "● Item" -> "- Item"
    text = re.sub(
        r'^(\s*)' + bullet_chars + r'\s*',
        r'\1- ',
        text,
        flags=re.MULTILINE,
    )

    return text


def _looks_like_list_page(raw_text):
    """Detect pages that are dense lists of short entries (book indexes,
    glossaries, bibliographies) rather than prose paragraphs.

    On such pages, clean_text's paragraph-joining logic collapses every
    entry onto a single line. We preserve newlines on these pages so each
    entry stays on its own line.

    Heuristics (all must hold):
      - 10+ non-empty content lines
      - 80%+ of content lines are short (<80 chars) and don't end with
        sentence-ending punctuation
      - Average content-line length is under 50 chars
      - Either the top-level entries are alphabetically mostly-sorted
        (classic index) OR the average line is very short (<25 chars,
        glossary/bibliography shape). Sub-entries are skipped for the
        ordering check because they start with prepositions/articles
        ("and capitalism", "ideology of", "as war") and break sort order.
    """
    if not raw_text:
        return False
    content_lines = [
        ln.strip() for ln in raw_text.split('\n')
        if ln.strip() and not ln.strip().startswith('<!-- ')
    ]
    if len(content_lines) < 10:
        return False

    short_non_terminal = sum(
        1 for ln in content_lines
        if len(ln) < 80 and ln[-1] not in '.!?:;'
    )
    if short_non_terminal / len(content_lines) < 0.8:
        return False

    avg_len = sum(len(ln) for ln in content_lines) / len(content_lines)
    if avg_len >= 50:
        return False

    # Sub-entry filter: index sub-entries start with common stopwords
    # that break alphabetical order ("and", "as", "at", "by", "for", "in",
    # "of", "on", "to", "with", "the") or start with lowercase.
    SUBENTRY_LEADERS = {
        'and', 'as', 'at', 'by', 'for', 'from', 'in', 'into', 'of',
        'on', 'or', 'the', 'to', 'with', 'vs', 'see'
    }
    top_level = []
    for ln in content_lines:
        if not ln[0].isalpha():
            continue
        if ln[0].islower():
            continue
        # `ln[0].isalpha()` accepts Unicode letters but the ASCII regex only
        # matches [A-Za-z]. Lines starting with accented letters (École,
        # Übung) or non-Latin scripts pass the .isalpha() guard but produce
        # None from re.match, which used to crash. Fall back to the raw
        # first character when the ASCII regex misses.
        m = re.match(r'[A-Za-z]+', ln)
        first_word = m.group(0).lower() if m else ln[0].lower()
        if first_word in SUBENTRY_LEADERS:
            continue
        top_level.append(ln[0].lower())

    if len(top_level) >= 8:
        ordered = sum(1 for a, b in zip(top_level, top_level[1:]) if a <= b)
        if ordered / (len(top_level) - 1) >= 0.75:
            return True

    # Glossary/bibliography shape: every line is very short on average.
    return avg_len < 25


def clean_text(text, preserve_newlines=False):
    """Post-process extracted text to fix common PDF conversion artifacts.

    Joins all lines within a paragraph into single long lines. A paragraph
    ends at a blank line, a structural element (heading, list, page marker,
    etc.), or a line that clearly starts a new paragraph (after sentence-ending
    punctuation on the previous line AND starts with uppercase).

    Also fixes ligatures, split-word artifacts, hyphenated word breaks, and
    soft hyphens. Preserves YAML frontmatter (between --- fences) untouched.

    If preserve_newlines is True, runs the character-level fixes (ligatures,
    missing spaces, etc.) but skips paragraph-joining, so each source line
    stays on its own line. Used for index/glossary/bibliography pages where
    each entry is a standalone line.
    """
    # Fix ligatures, split words, missing spaces, spaced-out letters, and bullets
    text = _fix_ligatures(text)
    text = _fix_missing_spaces(text)
    text = _collapse_spaced_letters(text)
    text = _split_joined_caps(text)
    text = _normalize_bullets(text)

    if preserve_newlines:
        return text

    lines = text.split('\n')
    result = []
    i = 0

    # Skip YAML frontmatter if present
    if lines and lines[0].strip() == '---':
        result.append(lines[0])
        i = 1
        while i < len(lines):
            result.append(lines[i])
            if lines[i].strip() == '---':
                i += 1
                break
            i += 1

    # Lines that start a list item should absorb their continuation lines
    # into the same paragraph (the PDF wraps them across physical lines).
    list_item_pattern = re.compile(r'^(\s*)(?:[-*]\s|\d+[\.\)]\s)')

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Headings, page markers, blockquotes, fences, etc. stay as-is.
        # List items are structural but still need continuation joining.
        if _is_structural_line(stripped) and not list_item_pattern.match(line):
            result.append(line)
            i += 1
            continue

        # Start building a paragraph by joining continuation lines
        while i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.strip()

            # Stop joining at structural elements or blank lines (including
            # the next list item, which is itself structural)
            if _is_structural_line(next_stripped):
                break

            # Stop joining if the current line ends a sentence AND
            # the next line starts a new one (uppercase after period)
            current_trimmed = line.rstrip()
            if (current_trimmed
                    and current_trimmed[-1] in '.!?"\u201d'
                    and next_stripped
                    and next_stripped[0].isupper()
                    and len(current_trimmed) > 40):
                break

            # Fix hyphenated word breaks
            if current_trimmed.endswith('-') and next_stripped and next_stripped[0].islower():
                line = current_trimmed[:-1] + next_stripped
            else:
                line = current_trimmed + ' ' + next_stripped
            i += 1

        result.append(line)
        i += 1

    return '\n'.join(result)


def _detect_two_column_split(page):
    """Detect whether a page has a two-column layout.

    Returns the x-coordinate of the column gutter, or None if the page is
    a single column.

    Uses word-level bboxes (not blocks) so that the detection works even
    when PyMuPDF returns the entire body as a single merged block spanning
    both columns. Finds the x-position near the page midpoint where the
    highest fraction of rows have no word crossing through, and requires
    that fraction to be high (>= 65%). For a single-column page, body
    text spans the midpoint on most rows so no such gutter exists.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return None
    if not words or len(words) < 30:
        return None
    page_width = page.rect.width
    mid = page_width / 2

    # Group words into rows, clustering y values within a tolerance so
    # that words on the same visual line (with minor baseline variation)
    # merge into one row. Without this, a single full-width line can split
    # into two "rows" that each have only partial x coverage, producing
    # false-positive gutters in single-column layouts.
    clean = [
        (w[0], w[1], w[2]) for w in words
        if len(w) >= 5 and isinstance(w[4], str) and w[4].strip()
    ]
    if not clean:
        return None
    clean.sort(key=lambda w: w[1])
    rows = []  # list of (y, [(x0, x1), ...])
    row_tol = 3.0
    for x0, y, x1 in clean:
        if rows and abs(y - rows[-1][0]) <= row_tol:
            rows[-1][1].append((x0, x1))
        else:
            rows.append((y, [(x0, x1)]))
    if len(rows) < 8:
        return None

    # Scan a window near the midpoint for the x with the best "no crossing"
    # coverage. Step in small increments so we can locate the gutter precisely
    # (body column gaps are often only 6-12pt wide). Require high coverage
    # (>= 75% of rows), AND require both sides to have substantial content so
    # we don't mistake a single-column layout (where most rows happen to end
    # before the midpoint due to ragged-right flush) for two columns.
    window = page_width * 0.15
    step = 2.0
    best_x = None
    best_coverage = 0
    x = mid - window
    while x <= mid + window:
        count = 0
        for _, intervals in rows:
            if not any(ix0 < x < ix1 for ix0, ix1 in intervals):
                count += 1
        if count > best_coverage:
            best_coverage = count
            best_x = x
        x += step

    if best_x is None:
        return None
    if best_coverage / len(rows) < 0.75:
        return None

    # Two-column pages come in two flavors:
    #
    # (1) "Synchronized": rows contain content from BOTH columns at the
    #     same y (e.g. argyris1977 body where PyMuPDF merges the columns
    #     into one block and we get words from both sides on every row).
    #     Signal: rows have a clean horizontal gap at the gutter.
    #
    # (2) "Staggered": each visual row is entirely in one column, but the
    #     columns are separate blocks stacked at different y-ranges (e.g.
    #     argyris1955 p4 where PyMuPDF gives us distinct left-col and
    #     right-col blocks). Signal: many rows live wholly in the left
    #     half, many rows live wholly in the right half.
    #
    # Accept if either signal is strong. Reject only when neither holds,
    # which correctly rules out 1-column pages with ragged-right lines
    # (book indexes, bibliographies) where the right half is empty.
    sync_rows = sum(
        1 for _, intervals in rows
        if _row_has_gutter_gap(intervals, best_x, min_gap_width=6)
    )
    left_only_rows = sum(
        1 for _, intervals in rows
        if intervals and all(ix1 < best_x - 2 for ix0, ix1 in intervals)
    )
    right_only_rows = sum(
        1 for _, intervals in rows
        if intervals and all(ix0 > best_x + 2 for ix0, ix1 in intervals)
    )
    # "Any" counts: rows with at least one word left of / right of gutter.
    # Used for the asymmetric body+sidebar rule below.
    left_any_rows = sum(
        1 for _, intervals in rows
        if any(ix1 < best_x - 2 for ix0, ix1 in intervals)
    )
    right_any_rows = sum(
        1 for _, intervals in rows
        if any(ix0 > best_x + 2 for ix0, ix1 in intervals)
    )

    if sync_rows >= 6 and sync_rows / len(rows) >= 0.20:
        return best_x
    if (
        left_only_rows >= 5
        and right_only_rows >= 5
        and left_only_rows / len(rows) >= 0.15
        and right_only_rows / len(rows) >= 0.15
    ):
        return best_x
    # Asymmetric / body+sidebar layout (variant A): the main body is a
    # single wide column and a short author bio sits in the other column,
    # with the bio's y-range overlapping the body so some rows are
    # "synchronized". argyris1993 p.2 is the canonical case.
    if sync_rows >= 5 and (left_any_rows >= 20 or right_any_rows >= 20):
        return best_x
    # Asymmetric / body+sidebar layout (variant B): the bio is a dedicated
    # 3-8 line block in one column, the body fills the other column, and
    # the y-ranges don't align closely enough to produce many synchronized
    # rows. argyris1989 p.2 is the canonical case: Lo=4, Ro=38, sync=3.
    # Accept as long as both columns have at least 3 dedicated rows AND
    # one side has 25+ rows of content AND there's at least some sync
    # evidence (sync + smaller dedicated count >= 5). This still rejects
    # single-column ragged pages (e.g. book indexes) because those have
    # zero dedicated rows in the empty "right column".
    if (
        left_only_rows >= 3
        and right_only_rows >= 3
        and max(left_any_rows, right_any_rows) >= 25
        and sync_rows + min(left_only_rows, right_only_rows) >= 5
    ):
        return best_x
    return None


def _row_has_gutter_gap(intervals, gutter, min_gap_width=6):
    """Return True if `intervals` has a horizontal gap spanning `gutter`.

    Merges overlapping intervals, then looks for a pair of consecutive
    merged intervals where the gap between them straddles `gutter` and is
    at least `min_gap_width` points wide. This is the geometric signature
    of a column gutter: content ends on one side, whitespace crosses the
    gutter, content resumes on the other side.
    """
    if not intervals:
        return False
    sorted_ivs = sorted(intervals, key=lambda iv: iv[0])
    merged = [sorted_ivs[0]]
    for a, b in sorted_ivs[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_start < gutter < gap_end and (gap_end - gap_start) >= min_gap_width:
            return True
    return False


def _words_to_text(words):
    """Reconstruct text from PyMuPDF word tuples, grouping into lines by y.

    Words are (x0, y0, x1, y1, text, block_no, line_no, word_no). Lines are
    separated by '\\n' so downstream line-oriented cleanup (paragraph join,
    header stripping) continues to work.
    """
    if not words:
        return ""
    sorted_words = sorted(words, key=lambda w: (round(w[1]), w[0]))
    lines = []
    current_line = []
    current_y = None
    for w in sorted_words:
        wy = round(w[1])
        if current_y is None or abs(wy - current_y) > 2:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [w[4]]
            current_y = wy
        else:
            current_line.append(w[4])
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


# --- Table fidelity accounting ---------------------------------------------
#
# Independent of *how* a backend finds tables, we want a cheap post-hoc
# answer to "did the grids survive?" Counting captions and emitted grids
# separately gives that: a book with 47 "TABLE A-n" captions and 12
# emitted grids lost 35 tables into prose, and the sidecar says so
# without anyone reading 330 pages.

# Matches a caption at the head of a line, with or without markdown bold:
#   TABLE A-16 Mean Motive ...
#   **TABLE 3.2.** Some Events ...
#   EXHIBIT 5-1
_TABLE_CAPTION_LINE_RE = re.compile(
    r'^\s*\**\s*(?:TABLE|Table|TAB\.|Tab\.|EXHIBIT|Exhibit)\s+'
    r'[A-Z]?-?\d+(?:[-.]\d+)?',
    re.MULTILINE,
)

# A GFM table is identified by its separator row (|---|---|), which is the
# one line every well-formed markdown table must have exactly once.
_GFM_SEPARATOR_RE = re.compile(r'^\s*\|(?:\s*:?-+:?\s*\|)+\s*$', re.MULTILINE)

# marker with --html_tables_in_markdown emits <table> instead of GFM.
_HTML_TABLE_RE = re.compile(r'<table[\s>]', re.IGNORECASE)


def count_table_signals(markdown_text):
    """Return (tables_emitted, table_captions_seen) for a markdown body.

    `tables_emitted` counts both GFM grids and raw HTML tables, so the
    number stays comparable across renderer settings. Neither count is a
    quality measure — a mangled grid still counts as emitted. The signal
    is the *gap* between the two.
    """
    emitted = (
        len(_GFM_SEPARATOR_RE.findall(markdown_text))
        + len(_HTML_TABLE_RE.findall(markdown_text))
    )
    captions = len(_TABLE_CAPTION_LINE_RE.findall(markdown_text))
    return emitted, captions


def apply_table_signals(report, output_path):
    """Populate a report's table counters by reading the emitted markdown.

    Best-effort: a read failure leaves the counters at zero rather than
    failing a conversion that otherwise succeeded.
    """
    try:
        text = Path(output_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        report.warnings.append(f"could not count tables: {e}")
        return
    report.tables_emitted, report.table_captions_seen = count_table_signals(text)
    missing = report.table_captions_seen - report.tables_emitted
    if missing > 0:
        report.warnings.append(
            f"{missing} table caption(s) have no emitted grid — "
            f"those tables likely collapsed into prose"
        )


# --- Table detection and grid reconstruction -------------------------------
#
# Many books embed comparison tables with captions like "TABLE 9-1" followed
# by a borderless grid of cells. PyMuPDF's default text extraction reads
# such tables column-by-column (because the columns are narrow), which
# produces a vertical stream of one-cell-per-line text that loses the
# grid structure entirely.
#
# These helpers use the word-level coordinates PyMuPDF exposes to recover
# the grid: cluster words into visual rows by y, bin by x-column, merge
# continuation rows where a cell wraps across multiple visual rows, and
# emit a markdown table. The caption pattern is used as both a detection
# anchor and a region-bound estimator.

_TABLE_CAPTION_RE = re.compile(
    r'^\s*(?:TABLE|Table|TAB\.|Tab\.)\s+\d+(?:[-.]\d+)?\.?\s*$'
)


def _cluster_visual_rows(words, y_tol=3):
    """Group PyMuPDF word tuples into visual rows by y-coordinate.

    A visual row is a set of words whose y0 values fall within `y_tol`
    points of each other. Rows are returned sorted top-to-bottom; words
    within a row are sorted left-to-right.
    """
    if not words:
        return []
    sorted_ws = sorted(words, key=lambda w: (w[1], w[0]))
    rows = [[sorted_ws[0]]]
    for w in sorted_ws[1:]:
        row_y = min(x[1] for x in rows[-1])
        if abs(w[1] - row_y) <= y_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w[0])
    return rows


def _cluster_1d(values, tol):
    """Cluster a sorted 1D list into groups where gaps < tol merge."""
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] < tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _looks_like_prose_row(row_words, page_width):
    """Heuristic: does this visual row look like a paragraph of running prose?

    Used to find where a table region ends and body text resumes. Prose
    rows have many words, span most of the page width, begin near the
    left margin, and — crucially — contain multi-character English words
    with lowercase letters. Table data rows can span the full width with
    ten "words", but those words are mostly single-character symbols
    (√, H, M, ___) or short proper nouns, so their average word length
    is short and they carry few lowercase words.
    """
    if len(row_words) < 8:
        return False
    x_min = min(w[0] for w in row_words)
    x_max = max(w[2] for w in row_words)
    if x_min > 100:  # prose starts near the left margin
        return False
    if (x_max - x_min) < page_width * 0.55:  # prose spans wide
        return False
    # Distinguish prose from a full-width data row packed with symbols:
    # real prose has several lowercase-initial words and a non-trivial
    # average word length.
    texts = [w[4] for w in row_words]
    total_chars = sum(len(t) for t in texts)
    avg_len = total_chars / len(texts) if texts else 0
    if avg_len < 3.5:
        return False
    lowercase_words = sum(1 for t in texts if t and t[0].islower())
    if lowercase_words < 3:
        return False
    return True


def _get_page_lines(page):
    """Return a flat list of (y, x, text) triples for every text line on the page.

    Uses PyMuPDF's dict-format extraction so each table cell (which PDF
    producers typically encode as its own text line with its own bbox)
    stays intact as a single entry with its real x-position. This is
    much more faithful than word-level clustering for recovering table
    structure, because multi-word cells like "Teledyne return" arrive
    as one entry rather than two phantom columns.
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return []
    out = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            bbox = line.get("bbox") or (0, 0, 0, 0)
            out.append((bbox[1], bbox[0], text))
    out.sort(key=lambda t: (round(t[0]), t[1]))
    return out


def _cluster_lines_to_rows(lines, y_tol=4):
    """Group (y, x, text) line triples into visual rows by y-coordinate."""
    if not lines:
        return []
    rows = [[lines[0]]]
    for entry in lines[1:]:
        row_y = min(e[0] for e in rows[-1])
        if abs(entry[0] - row_y) <= y_tol:
            rows[-1].append(entry)
        else:
            rows.append([entry])
    for r in rows:
        r.sort(key=lambda e: e[1])
    return rows


def _row_is_decorative(row):
    """True if a row only contains decorative punctuation (e.g., '. . .').

    Book designers often drop an ornamental separator below a table or
    between table and prose. Letting it leak into the final row of the
    markdown grid produces outputs like 'Hedgehog . . .'.
    """
    text = " ".join(e[2] for e in row).strip()
    if not text:
        return True
    if any(ch.isalnum() for ch in text):
        return False
    return True


def _row_looks_like_prose(row, page_width):
    """True if a row of (y,x,text) line entries reads like a prose paragraph.

    Uses `dict`-level line entries. A prose row consists of a single line
    entry — PDF producers encode each full-width paragraph line as one
    line with a single bbox. Table rows, even when cells contain long
    text, have multiple entries at different x positions (one per cell).
    The dict-level structure already separates them for us.
    """
    if len(row) != 1:
        return False
    entry = row[0]
    text = entry[2]
    tokens = text.split()
    if len(tokens) < 8:
        return False
    if entry[1] > 100:  # prose starts near the left margin
        return False
    total_chars = sum(len(t) for t in tokens)
    avg_len = total_chars / len(tokens)
    if avg_len < 3.5:
        return False
    lowercase_words = sum(1 for t in tokens if t and t[0].islower())
    if lowercase_words < 3:
        return False
    return True


def _find_table_regions(page):
    """Locate table regions on a page and return them as markdown.

    Returns a list of (start_y, end_y, markdown) tuples sorted top-to-bottom.
    start_y/end_y are in PDF point coordinates so callers can exclude
    those regions from normal text extraction.

    Detection is anchored on a "TABLE N-N" caption row. The region
    extends from the caption's y down to the first subsequent visual row
    that looks like running prose. Lines inside the region are grouped
    into a grid using x-centers derived from the highest-density data
    rows (headers are then assigned to the nearest data column), with
    sparse continuation rows folded upward into their parent row.
    """
    lines = _get_page_lines(page)
    if not lines:
        return []

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    visual_rows = _cluster_lines_to_rows(lines, y_tol=4)
    if not visual_rows:
        return []

    regions = []
    i = 0
    while i < len(visual_rows):
        row_text = " ".join(e[2] for e in visual_rows[i]).strip()
        if not _TABLE_CAPTION_RE.match(row_text):
            i += 1
            continue

        cap_y = min(e[0] for e in visual_rows[i])
        cap_label = row_text.strip()

        # Optional sub-caption (the table's title) sits on the very next
        # visual row when present. It goes above the grid as a bold line.
        subtitle = None
        data_start = i + 1
        if data_start < len(visual_rows):
            nr = visual_rows[data_start]
            nr_text = " ".join(e[2] for e in nr).strip()
            if (
                nr_text
                and len(nr_text) < 80
                and not nr_text.endswith(".")
                and len(nr) <= 3
                and min(e[1] for e in nr) < page_width * 0.5
            ):
                subtitle = nr_text
                data_start += 1

        # Walk forward until we find a prose row (table end) or hit
        # page bottom. The prose row itself is NOT part of the table.
        # Decorative separators (e.g., ". . .") also terminate the region
        # so they don't get mistaken for cell content in the final row.
        data_end = data_start
        end_y = page_height
        while data_end < len(visual_rows):
            row = visual_rows[data_end]
            if _row_looks_like_prose(row, page_width):
                end_y = min(e[0] for e in row)
                break
            if _row_is_decorative(row):
                end_y = min(e[0] for e in row)
                break
            data_end += 1

        data_rows = visual_rows[data_start:data_end]
        if len(data_rows) < 2:
            i = data_end
            continue

        markdown = _build_markdown_table(data_rows, cap_label, subtitle)
        if markdown is None:
            i = data_end
            continue

        regions.append((cap_y, end_y, markdown))
        i = data_end

    return regions


def _build_markdown_table(visual_rows, caption, subtitle):
    """Reconstruct a markdown table from a list of dict-level visual rows.

    Each `visual_rows[i]` is a list of (y, x, text) triples representing
    the cell-lines on one visual row. Column centers are estimated from
    the most-populated rows (which are typically the primary data rows),
    then every row's cells are assigned to the nearest column. Sparse
    continuation rows (where a cell wraps across visual rows) are merged
    upward into their parent row.

    Returns None if the region does not look like a real table.
    """
    # The columns in the logical table come from the densest data rows;
    # headers can have one fewer cell than the data row they sit above
    # (e.g., because the CEO-name column has no header). Taking column
    # x-positions from only the dense rows avoids polluting the column
    # set with phantom positions from wrapped header words.
    densities = [len(r) for r in visual_rows]
    peak = max(densities) if densities else 0
    if peak < 3:
        return None
    primary_rows = [r for r in visual_rows if len(r) >= max(3, peak - 1)]
    if not primary_rows:
        return None

    primary_x = sorted(e[1] for r in primary_rows for e in r)
    col_centers = _cluster_1d(primary_x, tol=15)
    if len(col_centers) < 2 or len(col_centers) > 12:
        return None

    ncols = len(col_centers)

    def assign_col(x):
        return min(range(ncols), key=lambda k: abs(x - col_centers[k]))

    grid = []
    for row in visual_rows:
        cells = [[] for _ in range(ncols)]
        for _y, x, text in row:
            cells[assign_col(x)].append(text)
        grid.append([" ".join(c).strip() for c in cells])

    merged = _merge_continuation_rows(grid)
    if len(merged) < 2:
        return None

    # Reject grids where no row has multiple filled cells — that's what
    # happens when we mistake a run of body prose for a table.
    if max((sum(1 for c in r if c.strip()) for r in merged), default=0) < 2:
        return None

    return _rows_to_markdown(merged, caption, subtitle)


def _merge_continuation_rows(grid):
    """Merge sparse continuation rows upward.

    When a cell's content wraps across multiple visual rows (e.g., a CEO
    name split as "Henry" / "Singleton"), PyMuPDF emits each wrapped line
    as its own visual row. A continuation row has very few non-empty
    cells relative to a fully populated row; we detect those and fold
    them into the preceding row by appending each cell's content.
    """
    if not grid:
        return grid

    # Determine the "typical" row density so we can call out sparse rows.
    densities = [sum(1 for c in r if c.strip()) for r in grid]
    if not densities:
        return grid
    peak = max(densities)
    sparse_threshold = max(2, peak // 2)

    merged = [list(grid[0])]
    for row, density in zip(grid[1:], densities[1:]):
        prev_density = sum(1 for c in merged[-1] if c.strip())
        is_sparse = density <= sparse_threshold and prev_density >= sparse_threshold + 1
        if is_sparse and merged:
            for idx, cell in enumerate(row):
                if cell.strip():
                    if merged[-1][idx].strip():
                        merged[-1][idx] = merged[-1][idx] + " " + cell.strip()
                    else:
                        merged[-1][idx] = cell.strip()
        else:
            merged.append(list(row))
    return merged


def _rows_to_markdown(rows, caption, subtitle):
    """Emit a markdown table with an optional bold caption line above."""
    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    # Pad short rows so the grid is rectangular.
    for r in rows:
        while len(r) < ncols:
            r.append("")

    def fmt_cell(cell):
        cell = cell.strip().replace("|", "\\|")
        return cell if cell else " "

    def fmt_row(r):
        return "| " + " | ".join(fmt_cell(c) for c in r) + " |"

    lines = []
    header_line = caption
    if subtitle:
        header_line = f"{caption} — {subtitle}" if caption else subtitle
    if header_line:
        lines.append(f"**{header_line}**")
        lines.append("")
    lines.append(fmt_row(rows[0]))
    lines.append("|" + "|".join(["---"] * ncols) + "|")
    for r in rows[1:]:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def _extract_page_text_with_regions(page, regions):
    """Stitch arbitrary markdown regions (tables, images) into page text.

    `regions` is a list of (start_y, end_y, markdown) tuples; they may
    overlap or come in any order. Overlapping regions are merged via
    the tighter of the two bounding y-ranges with markdown concatenated.
    Non-region text is pulled via clipped `page.get_text("text", clip=...)`
    so the flattened column-by-column dump never leaks through.
    """
    import fitz as _fitz

    if not regions:
        return page.get_text()

    # Sort by start_y ascending.
    sorted_regions = sorted(regions, key=lambda r: r[0])

    rect = page.rect
    segments = []
    cursor_y = rect.y0
    for start_y, end_y, md in sorted_regions:
        if start_y < cursor_y:
            # Region starts before cursor: skip overlap. This happens when
            # an image region overlaps a table region on the same page;
            # we preserve the first region and drop the later one.
            continue
        if start_y > cursor_y + 1:
            clip = _fitz.Rect(rect.x0, cursor_y, rect.x1, start_y)
            chunk = page.get_text("text", clip=clip)
            if chunk.strip():
                segments.append(chunk.rstrip())
        segments.append(md)
        cursor_y = end_y
    if cursor_y < rect.y1:
        clip = _fitz.Rect(rect.x0, cursor_y, rect.x1, rect.y1)
        chunk = page.get_text("text", clip=clip)
        if chunk.strip():
            segments.append(chunk.rstrip())
    return "\n\n".join(s for s in segments if s) + "\n"


def _extract_page_text(page, extra_regions=None):
    """Extract text from a page, respecting two-column layouts.

    For single-column pages, returns the default top-to-bottom extraction.
    For two-column pages, processes each block:
      - Blocks that don't cross the column gutter (pure left or pure right)
        are emitted as-is.
      - Short blocks that cross the gutter are treated as full-width
        (title, heading, abstract) and emitted as-is.
      - Tall blocks that cross the gutter are "merged" body blocks where
        PyMuPDF failed to separate the columns. These are split at the word
        level: left-column words are emitted in y-order, then right-column
        words, restoring the proper reading order.
    Blocks are then sorted by (y_top, x_left) so headers, bylines, abstract,
    body, and footnotes appear in the correct order.

    extra_regions: optional list of (y0, y1, markdown_str) tuples for image
    regions extracted by assets.extract_page_assets. Spliced into the page
    text alongside any table regions on single-column pages.
    """
    split = _detect_two_column_split(page)
    if split is None:
        # Single-column page: try to recover any "TABLE N-N" grids first.
        # When a caption is detected and the grid reconstructs cleanly we
        # splice markdown tables into the page text via clipped extraction
        # so the flattened column-by-column dump is replaced with a real
        # grid.
        table_regions = _find_table_regions(page)
        all_regions = list(table_regions)
        if extra_regions:
            all_regions.extend(extra_regions)
        if all_regions:
            return _extract_page_text_with_regions(page, all_regions)
        return page.get_text()

    try:
        blocks = page.get_text("blocks")
        words = page.get_text("words")
    except Exception:
        return page.get_text()

    text_blocks = [
        b for b in blocks
        if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()
    ]
    if not text_blocks:
        return page.get_text()

    # Partition blocks by role so we can keep each column contiguous.
    # PyMuPDF often fragments a 2-column page into one-line blocks per
    # column; sorting everything by (y, x) would interleave the columns
    # row-by-row and reproduce the very problem we're trying to avoid.
    # The correct reading order is: full-width headers, then all of the
    # left column top-to-bottom, then all of the right column top-to-bottom,
    # then full-width footers.
    MERGED_HEIGHT = 80
    GUTTER_TOL = 5

    left_col = []   # list of (y0, text) sorted by y0
    right_col = []
    full_width = []  # list of (y0, text) - short blocks spanning the gutter

    for b in text_blocks:
        x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
        height = y1 - y0
        crosses = (x0 < split - GUTTER_TOL) and (x1 > split + GUTTER_TOL)

        if crosses and height > MERGED_HEIGHT:
            # Merged body block: split at the word level. Append the
            # resulting halves to the column buffers so they read in the
            # correct order alongside any genuine pure-left / pure-right
            # blocks on the same page.
            #
            # Filter by block_no (word[5]) rather than bounding box so we
            # don't double-count words that live in neighboring overlapping
            # blocks. For example, a drop-cap title block often has a bbox
            # that overlaps the first few rows of the body block; the drop
            # cap's own words belong to block_no N while the body belongs
            # to block_no M, so bbox-based filtering would pull in both.
            block_no = b[5] if len(b) > 5 else None
            block_words = [
                w for w in words
                if len(w) >= 6 and isinstance(w[4], str) and w[4].strip()
                and (block_no is None or w[5] == block_no)
            ]
            left_words = [w for w in block_words if w[0] < split]
            right_words = [w for w in block_words if w[0] >= split]
            left_text = _words_to_text(left_words)
            right_text = _words_to_text(right_words)
            if left_text:
                ly0 = min((w[1] for w in left_words), default=y0)
                left_col.append((ly0, left_text))
            if right_text:
                ry0 = min((w[1] for w in right_words), default=y0)
                right_col.append((ry0, right_text))
        elif crosses:
            # Short full-width block (title, heading, caption, footnote).
            full_width.append((y0, btext))
        elif x1 <= split + GUTTER_TOL:
            left_col.append((y0, btext))
        else:
            right_col.append((y0, btext))

    left_col.sort(key=lambda e: e[0])
    right_col.sort(key=lambda e: e[0])
    full_width.sort(key=lambda e: e[0])

    # Decide where each full-width block sits relative to column content.
    # Headers end before the earliest column content; footers start after
    # the latest column content; anything in between is "inline" and we
    # place it after the left column so it still appears before the right
    # column body (best-effort — full-width mid-page elements are rare).
    col_y_min = min((e[0] for e in left_col + right_col), default=None)
    col_y_max = None
    if left_col or right_col:
        col_y_max = max(
            max((e[0] for e in left_col), default=float('-inf')),
            max((e[0] for e in right_col), default=float('-inf')),
        )

    header_fw = []
    footer_fw = []
    inline_fw = []
    for y0, text in full_width:
        if col_y_min is None:
            header_fw.append((y0, text))
        elif y0 < col_y_min:
            header_fw.append((y0, text))
        elif col_y_max is not None and y0 > col_y_max:
            footer_fw.append((y0, text))
        else:
            inline_fw.append((y0, text))

    # Join entries with single newlines within each segment (so clean_text
    # can re-flow wrapped visual lines into paragraphs) but double newlines
    # between segments (so header, columns, and footer stay distinct).
    # PyMuPDF frequently returns one block per visual line on column pages;
    # if we used "\n\n" between blocks, every line would become its own
    # paragraph and clean_text's line-joining heuristic wouldn't fire.
    segments = []
    if header_fw:
        segments.append("\n".join(t.strip() for _, t in header_fw if t.strip()))
    if left_col:
        segments.append("\n".join(t.strip() for _, t in left_col if t.strip()))
    if inline_fw:
        segments.append("\n".join(t.strip() for _, t in inline_fw if t.strip()))
    if right_col:
        segments.append("\n".join(t.strip() for _, t in right_col if t.strip()))
    if footer_fw:
        segments.append("\n".join(t.strip() for _, t in footer_fw if t.strip()))
    return "\n\n".join(s for s in segments if s) + "\n"


def _strip_surrogates(text):
    """Remove lone UTF-16 surrogate code points from extracted text.

    Some PDFs store text that PyMuPDF surfaces with unpaired surrogates
    (U+D800..U+DFFF) -- this happens with overlong UTF-8 sequences or
    non-standard embedded fonts. These code points cannot be encoded as
    valid UTF-8, so they crash the output file write. Strip them.
    """
    if not text:
        return text
    return re.sub(r'[\ud800-\udfff]+', '', text)


_PAGE_POINTER_RE = re.compile(r'^(?:p\.?\s*)?\d+\s*$', re.IGNORECASE)


def _is_useful_toc(toc_entries):
    """Decide whether an embedded TOC is worth emitting in the output.

    Academic papers from JSTOR and similar services embed a fake TOC of
    bare page pointers ("p. 1", "p. 2", ...) plus the whole journal
    issue's article list. These have no value in the output markdown and
    waste tokens.

    Heuristics for rejection:
      - All entries point at page <= 0 (PDF anchor targets, not real pages)
      - More than half of entry titles match "p. N" page-pointer format
      - All entries are at level 1 and the document has fewer than 30 pages
        (short papers don't need a TOC at all)
    """
    if not toc_entries:
        return False

    # All zero/negative page targets = JSTOR-style anchor-only TOC
    real_page_count = sum(1 for e in toc_entries if e[2] > 0)
    if real_page_count == 0:
        return False

    # Count page-pointer-style titles
    pointer_count = 0
    for entry in toc_entries:
        title = _strip_surrogates(entry[1] or '')
        title = re.sub(r'[\t\u2003\u2002\u00a0]+', ' ', title).strip()
        if _PAGE_POINTER_RE.match(title):
            pointer_count += 1
    if pointer_count > len(toc_entries) / 2:
        return False

    return True


def _format_embedded_toc(toc_entries):
    """Format a fitz TOC (list of [level, title, page]) as markdown.

    Each entry is indented by level. Uses a non-numbered list so the output
    reads as a hierarchical outline. Normalizes any tab/ideographic spaces
    in titles (common in bookmarks that prefix chapter numbers) and strips
    unpaired surrogate code points that some PDFs produce. Entries with
    page-pointer-style titles ("p. 5") or non-positive page targets are
    dropped since they come from JSTOR-style anchor TOCs that aren't useful.
    """
    if not toc_entries:
        return ""
    lines = ["## Table of Contents", ""]
    emitted = 0
    for entry in toc_entries:
        # fitz returns [level, title, page] (3-tuple) or with extra dict (4)
        level = max(1, entry[0])
        title = entry[1]
        page = entry[2]
        title = _strip_surrogates(title)
        title = re.sub(r'[\t\u2003\u2002\u00a0]+', ' ', title).strip()
        if not title:
            continue
        # Skip bare page pointers and unresolved anchor targets
        if _PAGE_POINTER_RE.match(title):
            continue
        if page <= 0:
            continue
        indent = "  " * (level - 1)
        lines.append(f"{indent}- {title} (p. {page})")
        emitted += 1
    if emitted == 0:
        return ""
    lines.append("")
    return "\n".join(lines)


def _is_broken_toc_page(text):
    """Detect pages that consist mostly of mangled TOC entries.

    After text extraction + bullet normalization, dotted-leader TOC entries
    look like "- Title ... pagenum" or similar. If most non-empty lines on
    a page match this pattern, we classify it as a broken TOC page and
    skip it in favor of the embedded bookmark-based TOC.
    """
    non_empty = [l.strip() for l in text.split('\n') if l.strip()]
    if len(non_empty) < 5:
        return False

    # Patterns for TOC-like lines:
    #   "- Title ... 42"            (post-normalization)
    #   "Title ....... 42"          (raw dotted leader)
    #   "- Figure ... 1"            (list-of-figures fragments)
    #   "Chapter 1 ... 42"
    toc_line = re.compile(
        r'^(?:-\s+)?(?:.+?)\s+(?:\.{2,}|…)\s*\d+\s*$'
    )
    # Also match lines that are a label + trailing number with no leader
    #   "- Figure ... 7"
    label_num = re.compile(
        r'^-\s+(?:Figure|Table|Chapter|Part|Section)\s*\.*\s*\d+\s*$',
        re.IGNORECASE,
    )
    bare_label_num = re.compile(
        r'^-?\s*(?:Figure|Table|Chapter|Part|Section)\s+\d+\s*$',
        re.IGNORECASE,
    )

    matches = sum(
        1 for line in non_empty
        if toc_line.match(line) or label_num.match(line) or bare_label_num.match(line)
    )
    return matches / len(non_empty) >= 0.5


def convert_with_pymupdf(pdf_path, output_dir, extract_images=True):
    """Convert a PDF using PyMuPDF (fitz) for text extraction.

    Raises ConversionError if the PDF appears to be scanned (too little text).
    `extract_images` defaults to True: figures and diagrams are rendered as
    PNGs into a <stem>_assets/ subdirectory alongside the markdown, and the
    references in the text point at files that are there. Setting it False
    emits neither the assets nor references to them — this backend only
    stitches a reference in when it has just written the file, so text-only
    output here is complete rather than perforated.
    """
    import fitz

    print(f"Converting with PyMuPDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"  {total_pages} pages")

    title = clean_title(pdf_path.stem)
    output_dir = Path(output_dir)
    # Create the output directory here rather than relying on the asset writer
    # to have made it as a side effect. A book with no extracted figures --
    # which is exactly what a scanned book now is, see
    # assets."page scans are not figures" -- otherwise fails at the final
    # write with a FileNotFoundError after doing all of the work.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pdf_path.stem}.md"

    report = ConversionReport(
        source=str(pdf_path),
        output=str(output_file),
        method="pymupdf",
    )

    asset_dir = output_dir / f"{pdf_path.stem}_assets"
    total_assets = 0

    # Extract embedded TOC (bookmark tree) before closing the doc.
    # This sidesteps pdf-text-extraction mangling of dotted-leader TOCs.
    try:
        toc_entries = doc.get_toc()
    except Exception as e:
        log.debug("Could not read embedded TOC: %s", e)
        toc_entries = []

    # Build a map of page number -> bookmark titles that start on that page.
    # _promote_bookmark_headings uses this to promote chapter titles to
    # markdown headings even when PyMuPDF extracts the title as multiple
    # split lines (e.g. "THE" / "CHALLENGE" / "OF THE" / "FUTURE").
    page_bookmarks = {}
    for entry in toc_entries:
        if len(entry) >= 3 and isinstance(entry[2], int) and entry[2] > 0:
            page_bookmarks.setdefault(entry[2], []).append(entry[1])

    pages_with_text = 0
    two_col_pages = 0

    # First pass: extract all page texts. _extract_page_text detects
    # two-column layouts and extracts columns in reading order, which
    # prevents PyMuPDF from interleaving left/right columns mid-sentence
    # on journal-article pages.
    raw_pages = []
    # A scanned book is one full-page bitmap per page with an OCR text layer on
    # top. Decided once for the document, because per page a scanned book's
    # sparse chapter-opener and a prose book's captioned plate are
    # indistinguishable. See assets."page scans are not figures".
    page_scan_document = assets.detect_page_scan_document(doc)
    if page_scan_document:
        print("  scanned book detected: page bitmaps will not be emitted as figures")
    for i in range(total_pages):
        page = doc[i]
        # Track how many pages look two-column so we can hint the user
        # toward --papers / marker-pdf for likely academic papers.
        if _detect_two_column_split(page) is not None:
            two_col_pages += 1
        page_asset_regions = []
        if extract_images:
            extracted = assets.extract_page_assets(
                page, pdf_path.stem, asset_dir, i + 1,
                page_scan_document=page_scan_document,
            )
            total_assets += len(extracted)
            # Convert to (start_y, end_y, markdown) tuples for the
            # region splicer. Each asset gets its own region.
            for rect, md in extracted:
                page_asset_regions.append((rect.y0, rect.y1, md))
        text = _extract_page_text(page, extra_regions=page_asset_regions)
        # Strip any unpaired surrogate code points that PyMuPDF sometimes
        # surfaces from PDFs with overlong UTF-8 or non-standard font encodings
        text = _strip_surrogates(text)
        if text.strip():
            pages_with_text += 1
            raw_pages.append((i + 1, text.strip()))
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total_pages} pages...")

    doc.close()

    # Strip running headers/footers across all pages, capturing the printed
    # page number (the only citation-valid address) as we go.
    cleaned_pages = _strip_running_headers(raw_pages)
    page_printed_by_pdf_index = {
        page_pdf: page_printed
        for page_pdf, _, page_printed in cleaned_pages if page_printed
    }
    page_printed_offset, page_printed_offset_consistent = _derive_page_offset(page_printed_by_pdf_index)
    # Support is measured HERE, on the same samples _derive_page_offset just
    # judged -- not later. The outlier sweep below deletes every capture that
    # contradicts an adopted offset, so a support figure taken after it reads
    # ~1.0 for any book that got one, which is both useless and a false
    # description of the field. (codex round 1, P2)
    page_printed_offset_support = _page_offset_support(page_printed_by_pdf_index)
    # Captured-but-contradicting folios are misreads, exactly as on the
    # marker path. Drop them so interpolation supplies the right number
    # instead of publishing a corrupted one with full confidence, and so
    # the span below is not stretched by a corrupted sample.
    misread_pdf_indices = _page_offset_outliers(
        page_printed_by_pdf_index,
        page_printed_offset if page_printed_offset_consistent else None,
    )
    for page_pdf in sorted(misread_pdf_indices):
        report.warnings.append(
            f"page_pdf={page_pdf}: captured printed page "
            f"'{page_printed_by_pdf_index[page_pdf]}' contradicts the book's "
            f"offset {page_printed_offset:+d}; treated as an OCR misread and "
            f"replaced by the interpolated value"
        )
        del page_printed_by_pdf_index[page_pdf]
    if not page_printed_offset_consistent and page_printed_by_pdf_index:
        report.warnings.append(
            "no page_pdf->page_printed offset could be established: "
            + (_page_offset_refusal(page_printed_by_pdf_index) or "")
            + "; pages without a captured printed page number were left "
            "addressable by pdf page only"
        )
    # Interpolation is permitted only BETWEEN captured samples. Outside that
    # span there is no evidence the offset still holds — endnotes or a second
    # numbering sequence past the last sample contribute no samples, so they
    # cannot disagree, and extrapolating there invents confident wrong page
    # numbers a reader cannot detect.
    arabic_page_pdf_indices = _arabic_page_pdf_indices(page_printed_by_pdf_index)
    page_pdf_span = (
        (min(arabic_page_pdf_indices), max(arabic_page_pdf_indices)) if arabic_page_pdf_indices else None
    )

    skipped_toc_pages = 0
    # page_printed_coverage's two terms MUST be counted over the same
    # population: pages that actually emitted a locator. Blank-after-clean
    # pages and replaced broken-TOC pages are `continue`d below and never
    # get one.
    #
    # Counting the denominator over `cleaned_pages` understates coverage.
    # Counting the numerator over `page_printed_by_pdf_index` (which
    # includes skipped pages) while the denominator excludes them is
    # worse: the ratio can exceed 1.0, and a nonsensical >1.0 value sails
    # straight through the ingestion gate's `>= threshold` check. Both are
    # counted here, in the emit loop, so they cannot drift apart again.
    emitted_locator_pages = 0
    emitted_page_printed_pages = 0

    # errors="replace" is a safety net for any surrogate chars that slip
    # through the explicit _strip_surrogates() calls above
    with open(output_file, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {title}\n\n")
        f.write("*Converted from PDF*\n\n")
        f.write(f"*Source: {pdf_path.name}*\n\n")
        f.write("---\n\n")

        # Emit embedded TOC at the top if it's actually useful. JSTOR-style
        # per-page anchor TOCs and the issue's article list are suppressed.
        emit_toc = _is_useful_toc(toc_entries)
        if emit_toc:
            formatted = _format_embedded_toc(toc_entries)
            if formatted:
                f.write(formatted)
                f.write("\n---\n\n")
            else:
                emit_toc = False

        # Only skip broken-TOC pages in the front matter (first 10% of pages)
        # AND only when we actually have a replacement TOC to offer.
        # Back-of-book indexes share the same line pattern but contain
        # content the user needs for keyword lookup.
        toc_skip_cutoff = max(20, total_pages // 10)

        for page_num, text, page_printed in cleaned_pages:
            # Detect index/glossary-style list pages so we can preserve
            # newlines instead of collapsing every entry into one paragraph.
            # Skip this in the front matter so _format_toc + broken-TOC
            # detection can still replace the printed Contents page with
            # the embedded bookmark TOC.
            in_front_matter = emit_toc and page_num <= toc_skip_cutoff
            is_list = (not in_front_matter) and _looks_like_list_page(text)
            cleaned = clean_text(text, preserve_newlines=is_list)
            # Use the bookmark tree to promote chapter titles to `##` even
            # when PyMuPDF extracts them as multi-line fragments.
            cleaned = _promote_bookmark_headings(
                cleaned, page_bookmarks.get(page_num, [])
            )
            cleaned = _format_headings(cleaned)
            cleaned = _format_toc(cleaned)
            # Skip blank pages (no content after cleaning)
            if not cleaned.strip():
                continue
            # Skip pages that are mostly mangled TOC entries -- the
            # embedded bookmark TOC replaces them. Restricted to front
            # matter so back-of-book indexes are preserved.
            if (emit_toc
                    and page_num <= toc_skip_cutoff
                    and _is_broken_toc_page(cleaned)):
                skipped_toc_pages += 1
                log.debug("Skipping broken TOC page %d", page_num)
                continue
            # A misread folio must not be emitted OR counted as captured.
            page_printed = page_printed_by_pdf_index.get(page_num)
            effective_page_printed = page_printed
            if (effective_page_printed is None
                    and page_printed_offset_consistent
                    and page_pdf_span is not None
                    and page_pdf_span[0] <= page_num <= page_pdf_span[1]):
                candidate = page_num + page_printed_offset
                # Never invent a printed page number for pages that precede
                # printed page 1.
                if candidate >= 1:
                    effective_page_printed = str(candidate)
            report.page_map.append({"page_pdf": page_num, "page_printed": effective_page_printed,
                                    "method": "observed" if page_printed is not None else "inferred" if effective_page_printed is not None else "none"})
            f.write(f"<!-- page_pdf={page_num} page_printed={effective_page_printed or 'none'} -->\n\n")
            emitted_locator_pages += 1
            # `page_printed`, not `effective_page_printed`: page_printed_count
            # counts CAPTURED printed numbers, never interpolated ones.
            if page_printed:
                emitted_page_printed_pages += 1
            f.write(cleaned)
            f.write("\n\n")

    if skipped_toc_pages:
        print(f"  Replaced {skipped_toc_pages} broken TOC page(s) with embedded bookmark TOC")

    report.page_locator_count = emitted_locator_pages
    report.page_printed_count = emitted_page_printed_pages
    report.page_printed_coverage = (
        report.page_printed_count / report.page_locator_count
        if report.page_locator_count else 0.0
    )
    report.page_printed_offset_support = page_printed_offset_support
    report.page_numbering = _decide_page_numbering(
        report.page_printed_count, page_printed_offset_consistent,
        page_printed_offset_support)
    report.page_printed_offset = page_printed_offset
    report.page_printed_offset_consistent = page_printed_offset_consistent

    # Check if we got enough text to consider this a real conversion
    if total_pages > 0 and (pages_with_text / total_pages) < MIN_TEXT_RATIO:
        output_file.unlink(missing_ok=True)
        raise ConversionError(
            f"Only {pages_with_text}/{total_pages} pages had extractable text. "
            f"This PDF may be scanned. Try: --method ocr"
        )

    # Check text quality: catches PDFs with non-standard font encodings that
    # produce extractable-but-garbled output (e.g. "n1ethod" for "method").
    # When quality is low we raise a ConversionError the caller can catch
    # and route to OCR via the auto_ocr fallback.
    extracted = output_file.read_text(encoding="utf-8", errors="replace")
    quality = _text_quality_score(extracted)
    log.debug("Text quality score: %.3f", quality)
    if quality < QUALITY_THRESHOLD:
        output_file.unlink(missing_ok=True)
        raise ConversionError(
            f"Low quality text extraction (score: {quality:.2f}, "
            f"threshold: {QUALITY_THRESHOLD:.2f}). This PDF may have "
            f"non-standard font encoding. Try: --method ocr"
        )

    print(f"  -> {output_file}")

    # Hint: if this document looks like an academic paper (majority of pages
    # detect as two-column AND it's short enough to plausibly be a paper),
    # suggest --papers for users on Python 3.10+. We never auto-switch; the
    # user opts in explicitly to keep behavior predictable.
    if (
        total_pages > 0
        and total_pages <= 60
        and two_col_pages / total_pages >= 0.5
    ):
        py_ok = sys.version_info >= (3, 10)
        if py_ok:
            print(
                "  Note: this looks like an academic paper. For higher-quality "
                "output, try --papers (routes to marker-pdf)."
            )
        else:
            print(
                "  Note: this looks like an academic paper. --papers would use "
                "marker-pdf but requires Python 3.10+ (current: "
                f"{sys.version_info.major}.{sys.version_info.minor})."
            )

    report.total_pages = total_pages
    report.pages_with_text = pages_with_text
    report.two_column_pages = two_col_pages
    report.quality_score = quality
    report.skipped_toc_pages = skipped_toc_pages
    report.extracted_assets = total_assets
    _finalize_assets(report, asset_dir=asset_dir)

    report_path = output_file.with_suffix(".report.json")
    write_report(report_path, report)
    return report


def _rewrite_marker_page_locators(target, report):
    """Replace marker's page separators with typed locator comments.

    Rewrites `{58}-----` into `<!-- page_pdf=58 page_printed=43 -->`, having
    stripped the running head and foot the locator flags asked marker to keep.

    Interpolation follows the same rules as the pymupdf path and for the same
    reason: printed numbers are captured sparsely (a chapter opener or a
    full-page table often shows none), and a gap between two agreeing samples
    is safe to fill while anything outside that span is a guess. Ambiguity
    refuses the whole book rather than inventing an address a reader cannot
    check.

    A no-op when the output is not paginated, which is what happens if a
    caller overrides the locator flags through `--marker-args`.
    """
    text = target.read_text(encoding="utf-8", errors="replace")
    pages, page_printed_by_index = _marker_page_locators(text)
    if pages is None:
        report.page_numbering = "none"
        report.warnings.append(
            "marker output was not paginated, so no page locators were "
            "emitted; the source cannot be cited by page"
        )
        return

    offset, offset_consistent = _derive_page_offset(page_printed_by_index)
    # Same reason as the pymupdf path: measure support on the samples the
    # offset decision just judged, before the outlier sweep below removes the
    # dissenters and makes every surviving book look unanimous.
    offset_support_at_derivation = _page_offset_support(page_printed_by_index)
    # Captured-but-contradicting folios are OCR misreads. Drop them so
    # interpolation can supply the right number, and so the span below is
    # not stretched by a corrupted sample.
    misread = _page_offset_outliers(page_printed_by_index, offset if offset_consistent else None)
    for index in sorted(misread):
        report.warnings.append(
            f"page_pdf={index}: captured printed page "
            f"'{page_printed_by_index[index]}' contradicts the book's "
            f"offset {offset:+d}; treated as an OCR misread and replaced "
            f"by the interpolated value"
        )
        del page_printed_by_index[index]

    if not offset_consistent and page_printed_by_index:
        report.warnings.append(
            "no page_pdf->page_printed offset could be established: "
            + (_page_offset_refusal(page_printed_by_index) or "")
            + "; pages without a captured printed page number were left "
            "addressable by pdf page only"
        )

    arabic = _arabic_page_pdf_indices(page_printed_by_index)
    span = (min(arabic), max(arabic)) if arabic else None

    out = []
    emitted = captured = 0
    for index, body in pages:
        printed = page_printed_by_index.get(index)
        effective = printed
        if (effective is None
                and offset_consistent
                and span is not None
                and span[0] <= index <= span[1]):
            candidate = index + offset
            if candidate >= 1:
                effective = str(candidate)
        report.page_map.append({"page_pdf": index, "page_printed": effective,
                                "method": "observed" if printed is not None else "inferred" if effective is not None else "none"})
        out.append(f"<!-- page_pdf={index} page_printed={effective or 'none'} -->\n")
        emitted += 1
        if printed:
            captured += 1
        if body.strip():
            out.append('\n' + body.strip('\n') + '\n\n')
        else:
            out.append('\n')
    target.write_text(''.join(out), encoding="utf-8")

    report.page_locator_count = emitted
    report.page_printed_count = captured
    report.page_printed_coverage = captured / emitted if emitted else 0.0
    # Measured on the samples the offset decision used; see the pymupdf path.
    offset_support = offset_support_at_derivation
    report.page_printed_offset_support = offset_support
    report.page_numbering = _decide_page_numbering(
        captured, offset_consistent, offset_support)
    report.page_printed_offset = offset
    report.page_printed_offset_consistent = offset_consistent
    print(f"  page locators: {emitted} pages, {captured} printed numbers "
          f"captured, coverage {report.page_printed_coverage:.2f}"
          + (f", offset {offset:+d}" if offset_consistent else ", offset inconsistent"))


# --- memory pre-flight ------------------------------------------------------
#
# marker holds its model weights resident — ~8.6 GB on a book-length scan. On a
# machine with no free RAM those weights get paged in and out continuously, and
# the run does not fail, it crawls: Ross & Nisbett ran ~50x slower per text unit
# than Boyatzis on the same laptop (0.14 vs 7 units/sec) purely because free
# memory had fallen to ~94 MB with 19M swapouts. A 1-hour conversion became a
# 13-hour one.
#
# That is worth a warning and not a block. The estimate is what the operator
# actually needs — "close something" is only actionable next to "or wait 13
# hours" — and a wrong guess about headroom must never stop a conversion the
# user asked for.

MARKER_RESIDENT_GB = 8.6      # measured peak RSS on a 330-page scan


def _pdf_page_count(pdf_path):
    """Page count for the warning message, or None if the PDF won't open.

    Best-effort by design: this exists to make a warning specific, and must
    never be the reason a conversion does not start.
    """
    try:
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)
    except Exception:      # noqa: BLE001 - cosmetic detail, never fatal
        return None


def _free_memory_gb():
    """Memory genuinely available now, in GB, or None if undeterminable.

    Counts `Pages free` + `Pages purgeable`. It deliberately does NOT count
    `Pages inactive`, which the first version of this check did and which made
    it useless: inactive pages on Apple Silicon are largely compressor-backed,
    and reclaiming them costs the very stalls this warning exists to predict.
    Measured on the machine that prompted the fix — 0.21 GB free, 22.6 GB
    compressed, marker crawling — the inactive-counting version reported
    "18.1 GB free" and stayed silent.

    Parses `vm_stat` (macOS). Returns None anywhere else rather than guessing,
    so the caller degrades to saying nothing.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        stats, page_size = {}, 4096
        for line in out.stdout.splitlines():
            m = re.match(r"Mach Virtual Memory Statistics: \(page size of (\d+) bytes\)", line)
            if m:
                page_size = int(m.group(1))
                continue
            m = re.match(r"^(.*?):\s+(\d+)\.?$", line)
            if m:
                stats[m.group(1).strip()] = int(m.group(2))
        available = stats.get("Pages free", 0) + stats.get("Pages purgeable", 0)
        return available * page_size / (1024 ** 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _compressed_gb():
    """Memory held in the compressor, in GB, or None if undeterminable.

    macOS compresses before it swaps, so on Apple Silicon a machine can be
    badly oversubscribed while reporting zero swap activity — which is exactly
    what the first version of this check keyed on and why it missed. A large
    compressor is the tell that the system is already working to make room.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        page_size = 4096
        m = re.search(r"page size of (\d+) bytes", out.stdout)
        if m:
            page_size = int(m.group(1))
        m = re.search(r"Pages occupied by compressor:\s+(\d+)", out.stdout)
        return int(m.group(1)) * page_size / (1024 ** 3) if m else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _swap_used_gb():
    """Swap currently in use, in GB, or None if it cannot be determined."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=5)
        m = re.search(r"used\s*=\s*([\d.]+)([MG])", out.stdout)
        if not m:
            return None
        value = float(m.group(1))
        return value / 1024 if m.group(2) == "M" else value
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# Swap is context, not a trigger. macOS leaves pages swapped out long after the
# pressure that caused them is gone — this machine still showed 1.2 GB of swap
# in use with 33 GB free, right after the thrashing conversion was killed — so
# swapping alone says nothing about whether the next run has room. Free memory
# is the decision variable; swap only sharpens the diagnosis when memory is
# already short.
SWAP_PRESSURE_GB = 1.0

# A compressor this large means the system is already working hard to make
# room. Named separately from swap because Apple Silicon reaches for
# compression first and may never swap at all.
COMPRESSOR_PRESSURE_GB = 8.0


def warn_if_memory_is_tight(page_count=None):
    """Warn before a long conversion when the machine has no room for it.

    Returns the free-GB figure (or None) so callers can record it.
    """
    free_gb = _free_memory_gb()
    swap_gb = _swap_used_gb()
    if free_gb is None or free_gb >= MARKER_RESIDENT_GB:
        return free_gb
    swapping = swap_gb is not None and swap_gb >= SWAP_PRESSURE_GB
    pages = f"{page_count}-page " if page_count else ""
    compressed_gb = _compressed_gb()
    notes = []
    if swapping:
        notes.append(f"{swap_gb:.1f} GB of swap is already in use")
    if compressed_gb is not None and compressed_gb >= COMPRESSOR_PRESSURE_GB:
        notes.append(f"{compressed_gb:.0f} GB is held in the memory compressor")
    swap_note = (" " + "; ".join(notes) + "." if notes else "")
    print(
        f"  WARNING: only {free_gb:.1f} GB of memory is free, and marker needs "
        f"about {MARKER_RESIDENT_GB:.0f} GB resident.{swap_note}\n"
        f"  This {pages}conversion will still finish, but the model weights will "
        f"page in and out of swap the whole way and it can run an order of "
        f"magnitude slower — hours instead of about one.\n"
        f"  Close what you can (browsers and Electron apps are usually the "
        f"biggest holders) and re-run, or leave it going in the background.",
        file=sys.stderr,
    )
    return free_gb


def convert_with_marker(pdf_path, output_dir, marker_args=None,
                        extract_images=True):
    """Convert a text-based PDF using Marker.

    Runs Marker in an isolated temp directory to avoid corrupting existing output.
    `marker_args` is a list of extra flags forwarded verbatim to
    `marker_single` (see `marker_single --help`) — this is how table-quality
    flags like `--use_llm` and `--html_tables_in_markdown` are reached.

    `extract_images` is on by default: marker finds the figures whether we ask
    it to or not, so the only question is whether we keep them, and the
    answer used to be no while the references stayed in the markdown. When
    it is off we tell marker to skip image extraction outright, and the
    finalizer strips any reference marker emits anyway.

    Raises ConversionError on failure.
    """
    print(f"Converting with Marker: {pdf_path.name}")
    warn_if_memory_is_tight(_pdf_page_count(pdf_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Newer marker-pdf releases (≥1.0) take the output path via the
        # `--output_dir` flag rather than as a second positional argument.
        # Passing it positionally raises "Got unexpected extra argument".
        # Validate the interpreter here, at the spawn, and not only in
        # check_dependencies() -- which `--skip-check` skips, and which every
        # documented invocation passes (`ingest-batch.sh`, and four places in
        # source-ingestion.md). A guard that only covers the path nobody takes
        # is not a guard. Proven on 2026-07-29: sourceconvert moved to
        # ~/conductor/repos, all 125 venv scripts kept pointing at the old
        # location, and the check added that same morning could not fire.
        _marker_path = shutil.which("marker_single")
        if _marker_path:
            _dead = _stale_shebang(_marker_path)
            if _dead is not None:
                _old_root = _dead.split("/.venv")[0]
                raise ConversionError(
                    f"`{_marker_path}` points at an interpreter that does not exist:\n"
                    f"    {_dead}\n"
                    "  The venv was built at a different path — this is what a moved or\n"
                    "  renamed project directory looks like. Repair the venv's absolute\n"
                    "  paths (shebangs, and VIRTUAL_ENV in the activate scripts):\n"
                    f"    OLD={_old_root}\n"
                    "    NEW=$(pwd)\n"
                    '    grep -rl "$OLD" .venv/bin .venv-marker/bin '
                    '| while read f; do sed -i "" "s|$OLD|$NEW|g" "$f"; done'
                )
        cmd = ["marker_single", str(pdf_path), "--output_dir", tmpdir]
        # Page-locator flags go on unconditionally: a source filed without a
        # citable address cannot be fixed later without reconverting, and
        # marker is the only backend that works on a scanned book.
        cmd.extend(MARKER_LOCATOR_ARGS)
        if not extract_images:
            # Only sent on the explicit opt-out path. An older marker that
            # does not know this flag would abort the run, and that risk
            # belongs to the caller who asked for text-only, never to the
            # default.
            cmd.append("--disable_image_extraction")
        if marker_args:
            cmd.extend(marker_args)
            print(f"  extra marker args: {' '.join(marker_args)}")

        # Stream marker's output rather than capturing it. A book-length
        # scan is a 1+ hour run and marker's only progress signal is its
        # per-stage progress bars on stderr; capture_output swallowed
        # them entirely, so a backgrounded run looked identical to a hung
        # one until it finished. We keep a rolling tail for the error
        # message instead.
        tail = collections.deque(maxlen=40)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            tail.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = proc.wait()
        if returncode != 0:
            raise ConversionError(f"Marker error: {''.join(tail)}")

        # Find the generated markdown in the temp directory
        tmp_path = Path(tmpdir)
        md_files = list(tmp_path.rglob("*.md"))
        if not md_files:
            raise ConversionError(
                "Marker produced no output. Try --method ocr for scanned PDFs."
            )

        # Move the first markdown file to the target location
        target = output_dir / f"{pdf_path.stem}.md"
        shutil.move(str(md_files[0]), str(target))

        # Marker's figures live in the scratch tree beside the markdown it
        # just wrote. `_finalize_assets` below harvests them into `img_dest`
        # before the TemporaryDirectory takes them to the grave. A previous
        # conversion of the same book owns nothing here, so clear the
        # directory rather than merging two runs' assets into it.
        # Spaces are squeezed out of the asset directory name, matching what the
        # pymupdf4llm backend already does. A space in the directory makes every
        # reference marker emits unusable: `![](A Spaced Name_images/x.jpeg)` is
        # not a valid CommonMark link destination, and consumers stop at the
        # first whitespace — mc-wiki's check-figures.py captures only "A" and
        # reports the figure dangling, while the file sits on disk beside it.
        #
        # Percent-encoding and angle-bracket wrapping are both more standards-
        # correct and both wrong here: that consumer resolves a ref as a literal
        # path with no URL-decoding, so either would still fail to find the file.
        # Renaming the directory is the only fix that needs no consumer changed,
        # and the name is generated rather than meaningful.
        safe_stem = re.sub(r"\s+", "_", pdf_path.stem)
        img_dest = output_dir / f"{safe_stem}_images"
        if img_dest.exists():
            shutil.rmtree(str(img_dest))

        if len(md_files) > 1:
            log.warning("Marker produced %d .md files; only the first was used", len(md_files))

        print(f"  -> {target}")
        report = ConversionReport(
            source=str(pdf_path),
            output=str(target),
            method="marker",
        )
        # Count pages in the source PDF for the report.
        import fitz
        try:
            with fitz.open(str(pdf_path)) as src:
                report.total_pages = len(src)
        except Exception as e:
            report.warnings.append(f"could not read page count: {e}")
        _rewrite_marker_page_locators(target, report)
        apply_table_signals(report, target)
        print(
            f"  tables: {report.tables_emitted} emitted / "
            f"{report.table_captions_seen} captions seen"
        )
        _apply_backend_page_numbering(report)
        # Issue #34 lived in the seven lines this call replaced. They globbed
        # `*.png` and `*.jpg` out of the scratch tree — marker writes `.jpeg`
        # — so the files were never found and were deleted with the temp
        # directory, while the references marker had already written into the
        # markdown stayed put, pointing at nothing. (The rewrite regex was
        # broken too: it required a `/` in the target, and marker's
        # `![](_page_64_Figure_7.jpeg)` has none.) Matching is on a suffix set
        # and on basenames now, and whatever is still unaccounted for gets
        # stripped rather than shipped dangling.
        _finalize_assets(report, scratch_root=tmp_path, asset_dir=img_dest)
        report_path = target.with_suffix(".report.json")
        write_report(report_path, report)
        return report


def convert_with_ocr(pdf_path, output_dir):
    """Convert a scanned PDF using OCR (pdf2image + tesseract).

    Processes pages one at a time to avoid loading all images into memory.
    Raises ConversionError on failure.
    """
    print(f"Converting with OCR: {pdf_path.name}")

    from pdf2image import convert_from_path, pdfinfo_from_path
    import pytesseract

    title = clean_title(pdf_path.stem)
    output_file = output_dir / f"{pdf_path.stem}.md"

    report = ConversionReport(
        source=str(pdf_path),
        output=str(output_file),
        method="ocr",
    )

    # Get page count first, then process one page at a time
    info = pdfinfo_from_path(str(pdf_path))
    total_pages = info["Pages"]
    print(f"  {total_pages} pages")

    pages_with_text = 0

    with open(output_file, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {title}\n\n")
        f.write("*Converted from PDF using OCR*\n\n")
        f.write(f"*Source: {pdf_path.name}*\n\n")
        f.write("---\n\n")

        for i in range(1, total_pages + 1):
            # Convert one page at a time to keep memory bounded
            images = convert_from_path(
                str(pdf_path), dpi=300, first_page=i, last_page=i
            )
            if images:
                text = pytesseract.image_to_string(images[0])
                if text.strip():
                    pages_with_text += 1
                    f.write(f"<!-- page_pdf={i} page_printed=none -->\n\n")
                    cleaned = clean_text(text.strip())
                    cleaned = _format_headings(cleaned)
                    cleaned = _format_toc(cleaned)
                    f.write(cleaned)
                    f.write("\n\n")
            if i % 10 == 0:
                print(f"  Processed {i}/{total_pages} pages...")

    print(f"  -> {output_file}")
    report.total_pages = total_pages
    report.pages_with_text = pages_with_text
    report.ocr_pages = total_pages
    extracted = output_file.read_text(encoding="utf-8", errors="replace")
    report.warnings.extend(_ocr_quality_warnings(extracted))
    # The tesseract path has no table reconstruction at all, so this
    # almost always reports "0 emitted / N captions". That is the point:
    # it makes the OCR backend's table blindness visible in the sidecar
    # instead of leaving it to be discovered during a scholarly pass.
    apply_table_signals(report, output_file)
    _apply_backend_page_numbering(report)
    # Tesseract emits plain text and never an image reference; the sweep is
    # here so the invariant holds for every backend without exception.
    _finalize_assets(report)
    report_path = output_file.with_suffix(".report.json")
    write_report(report_path, report)
    return report


def _strip_pandoc_frontmatter(body):
    """Remove pandoc's YAML frontmatter block if present.

    When pandoc produces gfm output from an epub with metadata (title,
    author, rights), it emits a `---\\n<yaml>\\n---\\n` block at the top.
    We strip it so the sourceconvert header block (title, source, ---) can
    sit at the top of the file in the same shape as the PDF output.
    """
    if not body.startswith("---\n"):
        return body
    end = body.find("\n---\n", 4)
    if end == -1:
        return body
    return body[end + 5:].lstrip()


# HTML comment pattern. Publishers often embed license notices
# (`<!--Licensed to ...-->`) throughout the epub source; these leak
# through even when raw_html is disabled because pandoc preserves HTML
# comments as a separate extension.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Images without alt text are usually decorative (publisher logos,
# section-break glyphs). Pandoc emits them as `![](path)` references.
# We don't extract media, so the paths dangle — drop the references.
_EMPTY_IMAGE_RE = re.compile(r"^!\[\]\([^)]*\)\s*$", re.MULTILINE)

# Collapse runs of 3+ blank lines (left behind by stripping comments
# and images) down to a single blank line.
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _clean_pandoc_output(body):
    """Strip publisher noise from pandoc's epub output.

    Removes HTML comments (license notices), empty-alt image references
    (decorative glyphs), and collapses the resulting blank-line runs.
    """
    body = _HTML_COMMENT_RE.sub("", body)
    body = _EMPTY_IMAGE_RE.sub("", body)
    body = _EXCESS_BLANK_LINES_RE.sub("\n\n", body)
    return body.strip() + "\n"


def convert_with_pandoc(book_path, output_dir, extract_images=None):
    """Convert an EPUB to markdown via pandoc.

    Pandoc maps epub chapter structure to markdown headings, so the
    output keeps the book's natural reading order without page markers.

    Images are **not** extracted unless `extract_images` is explicitly True.
    The default is off because an epub's images are overwhelmingly publisher
    furniture. Measured across five books on 2026-08-13: Groundedness 7 images,
    all furniture (Penguin logos, cover, title page, stock photos); Master of
    Change 3, all furniture; What Editors Do 5, all furniture; Way of Excellence
    4, two of them probable figures. Extracting by default would inflate every
    epub conversion to recover almost nothing.

    But `Peak Performance` in that same batch carried 11 images of which **7
    were page-numbered figures**, and under a hard-coded text-only rule they
    were unrecoverable without knowing to re-run pandoc by hand. Hence the
    escape hatch: `--extract-images` used to be documented as on-by-default
    while this path silently ignored it, so the flag lied. Passing it now works
    here, and `_finalize_assets` harvests the media exactly as the marker path
    does. Unspecified (None) keeps the text-only default.

    Returns a ConversionReport and writes the sidecar, like every other
    backend; its `page_numbering` is always "none" because an epub is
    reflowable and has no pages to address.

    Because there are no page locators, headings are the only addressing an
    epub has — and pandoc can only map headings the source actually carries.
    EPUBs whose chapter openers are styled `<p>` rather than `<h*>` used to
    convert to one flat document with no warning. We now check for semantic
    headings first and, when there are none, hand pandoc a rewritten copy
    with headings derived from the book's own navigation (see
    `epub_structure.py`). The sidecar always declares `heading_source` and
    `headings_emitted` so a structureless conversion is visible at
    conversion time instead of at filing time.

    Raises ConversionError on failure.
    """
    print(f"Converting with pandoc: {book_path.name}")

    title = clean_title(book_path.stem)
    output_file = output_dir / f"{book_path.stem}.md"

    media_scratch = None
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tempfile.TemporaryDirectory(prefix="sourceconvert-epub-") as work:
            try:
                convert_path, heading_source, injected = (
                    epub_structure.prepare_epub(book_path, Path(work))
                )
            except Exception as exc:      # never lose a book to this analysis
                log.warning("epub heading analysis failed: %s", exc)
                convert_path, heading_source, injected = (
                    book_path, epub_structure.SOURCE_NONE, 0
                )
            if heading_source == epub_structure.SOURCE_NAV:
                print(f"  no semantic headings; derived {injected} from the epub nav")
            elif heading_source == epub_structure.SOURCE_CLASS:
                print(
                    f"  no semantic headings or nav; derived {injected} from "
                    "chapter-ish CSS classes"
                )

            # `gfm-raw_html` keeps gfm's nice heading/list/emphasis handling
            # but drops raw HTML passthrough. Many epubs embed publisher
            # `<span>`/`<div>`/`<img>` layout scaffolding that pandoc would
            # otherwise preserve verbatim; disabling raw_html makes pandoc
            # flatten those tags to their text content instead.
            cmd = [
                "pandoc",
                "--from=epub",
                "--to=gfm-raw_html",
                "--wrap=none",
            ]
            # `--extract-media` writes the epub's media next to nothing useful
            # and rewrites every reference to point inside it, so it has to land
            # somewhere that outlives this `with` block: `_finalize_assets`
            # harvests it after the body is written, exactly as it does for
            # marker's scratch tree. Only when extraction was asked for
            # explicitly -- see the docstring for why epub defaults off.
            if extract_images:
                media_scratch = Path(tempfile.mkdtemp(prefix="sourceconvert-epub-media-"))
                cmd.append(f"--extract-media={media_scratch}")
            cmd += [str(convert_path), "-o", str(tmp_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ConversionError(
                    f"pandoc error: {result.stderr.strip() or 'unknown failure'}"
                )

            body = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    body = _strip_pandoc_frontmatter(body)
    body = _clean_pandoc_output(body)

    if not body.strip():
        raise ConversionError("pandoc produced empty output")

    with open(output_file, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {title}\n\n")
        f.write("*Converted from EPUB*\n\n")
        f.write(f"*Source: {book_path.name}*\n\n")
        f.write("---\n\n")
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")

    print(f"  -> {output_file}")

    # EPUB is reflowable: it has no pages, so no locator of any kind is
    # recoverable. That is exactly the case the ingestion gate most needs to
    # detect, so pandoc writes a sidecar declaring `page_numbering: "none"`
    # rather than writing nothing — silence is indistinguishable from a
    # conversion that never ran. Page counts stay at their defaults; a
    # fabricated `total_pages` would be its own small lie.
    report = ConversionReport(
        source=str(book_path),
        output=str(output_file),
        method="pandoc",
    )
    # Count headings in the converted body only. sourceconvert's own `# Title`
    # line is chrome, not structure: counting it would report "1 heading" for
    # exactly the flat-document failure this signal exists to expose.
    report.headings_emitted = epub_structure.count_markdown_headings(body)
    # We only report "none" when we injected nothing. If headings turned up
    # anyway, they came from the source — which means our spine analysis
    # missed them (an exotic container, say), not that the book is flat.
    # Saying "none" over a structured output would be the same silent lie
    # this signal exists to prevent, in the other direction.
    if (heading_source == epub_structure.SOURCE_NONE
            and report.headings_emitted):
        heading_source = epub_structure.SOURCE_SEMANTIC
    report.heading_source = heading_source
    if heading_source != epub_structure.SOURCE_SEMANTIC:
        report.warnings.append(
            "epub exposes no semantic h1-h6 headings; heading_source="
            f"{heading_source}"
        )
    if not report.headings_emitted:
        report.warnings.append(
            "no headings in output: an epub has no page locators, so this "
            "conversion has no structural addressing of any kind"
        )
    _apply_backend_page_numbering(report)
    # Pandoc references the epub's internal media paths (`media/image1.jpg`)
    # for images it never writes out, so an epub conversion dangles by
    # construction for every reference `_clean_pandoc_output` did not already
    # drop. Extraction is NOT the answer here the way it is for the PDF
    # backends: the images an epub carries that survive that filter are
    # overwhelmingly publisher furniture — covers, ornaments, imprint marks —
    # and sourceconvert's epub path is documented as text-only. So epub takes
    # the other route to the same invariant: the references are stripped.
    #
    # When --extract-images IS passed, pandoc has written the media into
    # `media_scratch` and repointed the body at it; hand both to
    # `_finalize_assets` and it harvests them beside the markdown, strips
    # anything still dangling, and writes the manifest -- the same three steps
    # the marker path gets.
    if media_scratch is not None:
        safe_stem = re.sub(r"\s+", "_", book_path.stem)
        img_dest = output_dir / f"{safe_stem}_images"
        if img_dest.exists():
            shutil.rmtree(str(img_dest))
        try:
            _finalize_assets(report, scratch_root=media_scratch, asset_dir=img_dest)
        finally:
            shutil.rmtree(str(media_scratch), ignore_errors=True)
    else:
        _finalize_assets(report)
    write_report(output_file.with_suffix(".report.json"), report)
    return report


def convert_with_pymupdf4llm(pdf_path, output_dir, extract_images=True):
    """Convert a PDF using pymupdf4llm's Markdown exporter.

    pymupdf4llm is PyMuPDF's own LLM-oriented markdown exporter. It
    handles multi-column reading order, image extraction, tables, and
    auto-OCR for scanned pages. This backend is a light wrapper: we
    call `to_markdown()` and stitch the output into sourceconvert's
    standard header format.
    """
    import pymupdf4llm
    import fitz

    print(f"Converting with pymupdf4llm: {pdf_path.name}")

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
    print(f"  {total_pages} pages")

    title = clean_title(pdf_path.stem)
    output_file = output_dir / f"{pdf_path.stem}.md"

    # pymupdf4llm returns the full markdown as a string by default, OR
    # a list of per-page dicts when page_chunks=True. We use page_chunks
    # so we can emit page markers between chunks. Markers use the shared
    # page_pdf/page_printed locator format; pymupdf4llm returns pre-cleaned
    # text, so no printed page number is recoverable here.
    # pymupdf4llm sanitizes spaces in image_path to underscores when it
    # writes the PNGs, so we must sanitize the directory name ourselves
    # before mkdir — otherwise we'd create "Stem With Spaces_images/" and
    # pymupdf4llm would try to write into "Stem_With_Spaces_images/" and
    # error out.
    safe_stem = re.sub(r"\s+", "_", pdf_path.stem)
    image_dir = output_dir / f"{safe_stem}_images"
    if extract_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        write_images=extract_images,
        image_path=str(image_dir),
        image_format="png",
        page_chunks=True,
    )

    # Stitch chunks with `<!-- page_pdf={N} page_printed=none -->` markers between
    # them. pymupdf4llm's metadata.page_number is already 1-indexed (matches the
    # pymupdf backend's convention at line ~2280 above). Use it directly;
    # fall back to a 1-based enumeration if the key is missing for some
    # malformed PDF.
    body_parts = []
    for idx, chunk in enumerate(page_chunks, start=1):
        page_num = chunk.get("metadata", {}).get("page_number", idx)
        chunk_text = chunk.get("text", "")
        body_parts.append(f"<!-- page_pdf={page_num} page_printed=none -->\n\n{chunk_text}")
    markdown = "\n\n".join(body_parts)

    # pymupdf4llm embeds the full `image_path` we passed as the literal ref
    # prefix, e.g. `![](output/Stem_images/x.png)` — or an absolute/symlink-
    # canonicalized path like `/private/tmp/.../Stem_images/x.png`. Either way
    # it is not relative to this markdown file (which lives *inside* the output
    # dir), so refs break the moment the file is opened or moved. Strip any
    # leading directory so every ref is relative to the markdown itself,
    # matching what the default pymupdf backend already emits (`{stem}_images/`
    # at ~line 2392).
    markdown = re.sub(
        r'\]\((?:[^)]*/)?' + re.escape(f"{safe_stem}_images/"),
        f"]({safe_stem}_images/",
        markdown,
    )

    header = (
        f"# {title}\n\n"
        f"*Converted from PDF using pymupdf4llm*\n\n"
        f"*Source: {pdf_path.name}*\n\n"
        f"---\n\n"
    )
    output_file.write_text(header + markdown, encoding="utf-8", errors="replace")
    print(f"  -> {output_file}")

    report = ConversionReport(
        source=str(pdf_path),
        output=str(output_file),
        method="pymupdf4llm",
        total_pages=total_pages,
        pages_with_text=total_pages,  # pymupdf4llm handles its own detection
    )
    _apply_backend_page_numbering(report)
    _finalize_assets(report, asset_dir=image_dir)
    write_report(output_file.with_suffix(".report.json"), report)
    return report


def convert_with_docling(pdf_path, output_dir):
    """Convert a PDF using IBM's Docling pipeline.

    Docling does layout analysis, reading order, tables, formulas, image
    classification, and markdown export. This backend is a light wrapper
    around `DocumentConverter.convert(source).document.export_to_markdown()`.
    """
    from docling.document_converter import DocumentConverter
    import fitz

    print(f"Converting with docling: {pdf_path.name}")

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
    print(f"  {total_pages} pages")

    title = clean_title(pdf_path.stem)
    output_file = output_dir / f"{pdf_path.stem}.md"

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    header = (
        f"# {title}\n\n"
        f"*Converted from PDF using docling*\n\n"
        f"*Source: {pdf_path.name}*\n\n"
        f"---\n\n"
    )
    output_file.write_text(header + markdown, encoding="utf-8", errors="replace")
    print(f"  -> {output_file}")

    report = ConversionReport(
        source=str(pdf_path),
        output=str(output_file),
        method="docling",
        total_pages=total_pages,
        pages_with_text=total_pages,
    )
    _apply_backend_page_numbering(report)
    # Docling's default export mode emits `<!-- image -->` placeholders rather
    # than file references, so there is normally nothing here to enforce. It
    # gets the same sweep anyway: the invariant is a property of sourceconvert's
    # output, not a favour we do the backends we happen to distrust.
    _finalize_assets(report)
    write_report(output_file.with_suffix(".report.json"), report)
    return report



def _finalize_assets(report, scratch_root=None, asset_dir=None):
    """Land every asset, kill every dead reference, and record the manifest.

    The last thing each backend does. Three steps, in order:

    1. **Harvest.** If the backend wrote its assets into a scratch directory
       it is about to lose (marker's temp tree), move them next to the
       markdown and repoint the references at their new home. Backends that
       write straight into the output directory pass `asset_dir` alone.
    2. **Enforce.** Strip any reference still pointing at a file that does not
       exist. This is the invariant — *the output never contains a reference
       to a file that does not exist* — and it holds no matter which backend
       ran, whether extraction was on, or what the backend believed it wrote.
       Extraction-by-default (step 1 and the `--extract-images` default) is
       what keeps satisfying the invariant from being a loss; this step is
       what makes it true.
    3. **Manifest.** Record every asset and the references pointing at it, so
       a consumer can relocate assets without pattern-matching a backend's
       private filename convention.

    See issue #34: marker emitted `![](_page_64_Figure_7.jpeg)` while writing
    no such file, 1,140 times across 49 sources, and nothing in the output
    said so.
    """
    if not isinstance(report, ConversionReport) or not report.output:
        return report
    md_path = Path(report.output)
    if not md_path.exists():
        return report
    asset_dir = Path(asset_dir) if asset_dir is not None else None

    if scratch_root is not None:
        harvested = assets.collect_asset_files(scratch_root)
        if harvested:
            if asset_dir is None:
                asset_dir = md_path.parent / f"{md_path.stem}_images"
            mapping = assets.harvest_assets(harvested, asset_dir)
            text = md_path.read_text(encoding="utf-8", errors="replace")
            text, rewrites = assets.rewrite_asset_refs(
                text, mapping, asset_dir.name
            )
            md_path.write_text(text, encoding="utf-8", errors="replace")
            print(
                f"  assets: {len(mapping)} written to {asset_dir.name}/ "
                f"({rewrites} reference(s) repointed)"
            )

    stripped = assets.enforce_reference_invariant(md_path)
    report.assets = assets.build_asset_manifest(md_path, asset_dir)
    report.extracted_assets = len(report.assets)
    report.dangling_refs_stripped = len(stripped)
    if stripped:
        report.warnings.append(
            f"stripped {len(stripped)} image reference(s) to files that were "
            f"not written (first: {stripped[0]}). The output is intact; the "
            f"figures are not in it."
        )
        print(
            f"  WARNING: {len(stripped)} image reference(s) pointed at files "
            f"that were never written; stripped so the output does not dangle."
        )
    return report


def _refresh_asset_manifest(report):
    """Rebuild the manifest against the markdown as it now stands on disk.

    Used after a pass that rewrites the file (cleanup), because the manifest
    carries per-reference line numbers. Asset directories are recovered from
    the existing manifest so assets nothing references stay listed.
    """
    md_path = Path(report.output)
    asset_dirs = {
        md_path.parent / Path(entry["path"]).parent
        for entry in report.assets
        if Path(entry["path"]).parent != Path(".")
    }
    report.assets = assets.build_asset_manifest(md_path, asset_dirs)
    report.extracted_assets = len(report.assets)


def _apply_cleanup(result):
    """Run the post-conversion cleanup pass on a backend's markdown output.

    Reads the markdown named by `result.output`, repairs extraction artifacts
    (see cleanup.clean_markdown), rewrites the file, and records what changed
    on the report (including a rewritten sidecar). Verbatim-safe and
    idempotent; a no-op if there is nothing to fix. Returns the report.
    """
    if not isinstance(result, ConversionReport) or not result.output:
        return result
    md_path = Path(result.output)
    if md_path.suffix.lower() != ".md" or not md_path.exists():
        return result

    original = md_path.read_text(encoding="utf-8", errors="replace")
    cleaned, stats = cleanup.clean_markdown(original)
    if cleaned != original:
        md_path.write_text(cleaned, encoding="utf-8", errors="replace")

    result.cleaned = True
    result.cleanup = stats
    # Cleanup can unwrap a picture-text table (a ToC rendered as a grid),
    # which changes the table counts the backend recorded a moment ago.
    # Recount against the post-cleanup text so the sidecar describes the
    # file that actually landed on disk.
    if cleaned != original:
        result.tables_emitted, result.table_captions_seen = count_table_signals(cleaned)
        # The manifest records a line number per reference. Cleanup rewrites
        # the file underneath it, so recompute rather than ship a manifest
        # that describes the pre-cleanup text.
        _refresh_asset_manifest(result)
    if not stats.get("dejoin_available"):
        result.warnings.append(
            "cleanup: de-join pass skipped (pyspellchecker not installed; "
            "`pip install -r requirements.txt` to enable dropped-space repair)"
        )
    write_report(md_path.with_suffix(".report.json"), result)
    return result


# --- concurrent conversions -------------------------------------------------
#
# One sourceconvert checkout is shared by several agent sessions and by Andy.
# The realistic collision is not two people converting the same book: it is one
# workspace running a two-hour OCR of a scanned book while another converts a
# handful of papers for an unrelated project. Both should proceed — serialising
# them would make the small job wait hours for no reason — but neither should
# be surprised by the other.
#
# Surprise is the actual cost. A concurrent run halves the CPU each gets and
# doubles resident memory, which turns an unexplained slowdown into a
# diagnosis that took an hour of process archaeology to reach: two markers,
# ~7 GB each, on a machine whose free memory had already gone.
#
# So this registry is advisory and never blocks. It answers one question at
# the moment it matters — "is anything else running, and for how long?" — and
# composes with the memory warning above, because "4 GB free" means something
# different when another conversion already holds 8 of them.

ACTIVE_CONVERSIONS = Path(__file__).resolve().parent / ".active-conversions.json"


def _process_alive(pid):
    """Whether `pid` is still running. Stale entries are the normal case:
    a killed or crashed conversion never gets to clean up after itself."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def read_active_conversions(exclude_pid=None):
    """Live entries from the registry, pruning any whose process has gone."""
    try:
        raw = json.loads(ACTIVE_CONVERSIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [e for e in raw
            if isinstance(e, dict)
            and isinstance(e.get("pid"), int)
            and e["pid"] != exclude_pid
            and _process_alive(e["pid"])]


def _write_active_conversions(entries):
    try:
        ACTIVE_CONVERSIONS.write_text(json.dumps(entries, indent=1) + "\n",
                                      encoding="utf-8")
    except OSError:
        pass                 # advisory only; never fail a conversion over it


def register_conversion(pdf_path, method, started_at):
    """Record this run, and return what was already running.

    Returns the list of other live conversions so the caller can report them.
    """
    others = read_active_conversions(exclude_pid=os.getpid())
    entry = {"pid": os.getpid(), "book": Path(pdf_path).name,
             "method": method, "started": started_at}
    _write_active_conversions(others + [entry])
    return others


def unregister_conversion():
    _write_active_conversions(read_active_conversions(exclude_pid=os.getpid()))


def describe_other_conversions(others, now):
    """One line per concurrent run, or None when there are none."""
    if not others:
        return None
    lines = ["  NOTE: another conversion is already running on this checkout:"]
    for e in others:
        try:
            mins = max(0, int((now - float(e.get("started", now))) // 60))
            age = f"{mins // 60}h{mins % 60:02d}m" if mins >= 60 else f"{mins}m"
        except (TypeError, ValueError):
            age = "unknown"
        lines.append(f"    - {e.get('book', '?')} ({e.get('method', '?')}), "
                     f"running {age}, pid {e.get('pid')}")
    lines.append("  Both will finish. Expect each to be slower, and expect "
                 "roughly 8 GB more memory in use than you would otherwise.")
    return "\n".join(lines)


def convert_book(book_path, output_dir, method="pymupdf", auto_ocr=False,
                 extract_images=True, clean=True, marker_args=None):
    """Convert a single book (PDF or EPUB) to markdown.

    Returns True on success, False on failure. Never raises.
    EPUB files route through pandoc regardless of `method`. For PDFs,
    if auto_ocr is True and pymupdf/marker fails due to scanned content,
    automatically retries with OCR.
    `extract_images` is tri-state, because the right default differs by
    format. **None (unspecified)** means "use the format's default": every PDF
    backend writes the figures it finds, and epub does not, since an epub's
    images are overwhelmingly publisher furniture (see `convert_with_pandoc`).
    **True** forces extraction everywhere, including epub -- the escape hatch
    for an epub that really does carry figures. **False** converts text only.
    Either way the references to skipped figures are stripped, so the output
    never dangles (issue #34).

    The resolution happens here rather than in each backend so that None can
    never reach one: every PDF backend tests the flag with a plain truthiness
    check, and None is falsy, so passing it straight through would silently
    disable PDF image extraction while looking like a default.
    When clean is True (default), a verbatim-safe post-conversion pass repairs
    common extraction artifacts (dropped-space joins, stray-consonant citation
    ghosts, picture-text garble) on the emitted markdown.
    `marker_args` is forwarded to the marker backend only; other backends
    ignore it.
    """
    # None means "not specified". Resolve it for the PDF backends here: they
    # test the flag with `if extract_images:` / `if not extract_images:`, and
    # None is falsy, so an unresolved None would turn image extraction OFF for
    # every PDF while reading like a default. Caught by converting a PDF and
    # finding no assets directory.
    pdf_extract_images = True if extract_images is None else extract_images

    book_path = Path(book_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = book_path.suffix.lower()
    is_pdf = suffix == ".pdf"

    try:
        if suffix == ".epub":
            # Pandoc now returns a ConversionReport like its siblings and
            # writes its own sidecar. The cleanup pass is deliberately NOT
            # run here: it repairs PDF *extraction* artifacts (dropped-space
            # joins, stray-consonant citation ghosts, pymupdf4llm
            # picture-text garble) — see cleanup.py's module docstring, "PDF
            # text extraction (all backends)". Pandoc's text comes from the
            # epub's HTML, which carries none of those defects, so running
            # the pass would be spending a dictionary sweep on text that
            # cannot need it.
            # epub keeps the raw tri-state: None means text-only here.
            return bool(convert_with_pandoc(book_path, output_dir,
                                             extract_images=extract_images))
        if method == "ocr":
            result = convert_with_ocr(book_path, output_dir)
        elif method == "marker":
            result = convert_with_marker(book_path, output_dir,
                                         marker_args=marker_args,
                                         extract_images=pdf_extract_images)
        elif method == "pymupdf4llm":
            result = convert_with_pymupdf4llm(book_path, output_dir,
                                              extract_images=pdf_extract_images)
        elif method == "docling":
            result = convert_with_docling(book_path, output_dir)
        else:
            result = convert_with_pymupdf(book_path, output_dir,
                                          extract_images=pdf_extract_images)
        # Backends return either True (legacy) or a ConversionReport.
        # Treat any non-False truthy value as success.
        if clean:
            result = _apply_cleanup(result)
        return bool(result)
    except ConversionError as e:
        error_msg = str(e).lower()
        auto_ocr_triggers = ("scanned", "low quality")
        if (
            auto_ocr
            and is_pdf
            and method != "ocr"
            and any(t in error_msg for t in auto_ocr_triggers)
        ):
            chosen = pick_ocr_backend()
            print(f"  Text extraction failed, auto-retrying with {chosen}...")
            try:
                check_dependencies(chosen)
                if chosen == "marker":
                    result = convert_with_marker(book_path, output_dir,
                                                 marker_args=marker_args,
                                                 extract_images=pdf_extract_images)
                else:
                    result = convert_with_ocr(book_path, output_dir)
                if clean:
                    result = _apply_cleanup(result)
                return bool(result)
            except DependencyError as dep_e:
                print(f"  OCR fallback unavailable: {dep_e}")
                return False
            except Exception as ocr_e:
                partial = output_dir / f"{book_path.stem}.md"
                if partial.exists():
                    partial.unlink()
                print(f"  OCR fallback also failed: {ocr_e}")
                log.debug(traceback.format_exc())
                return False
        print(f"  FAILED: {e}")
        return False
    except Exception as e:
        partial = output_dir / f"{book_path.stem}.md"
        if partial.exists():
            partial.unlink()
        print(f"  FAILED (unexpected): {e}")
        log.debug(traceback.format_exc())
        return False


# Public alias used by tests and future callers who deal exclusively with PDFs.
convert_pdf = convert_book

_SUPPORTED_BOOK_EXTS = {".pdf", ".epub"}


def collect_books(input_path):
    """Collect PDF and EPUB files from a path (file or directory).

    Extension matching is case-insensitive so .PDF and .EPUB both work.
    """
    input_path = Path(input_path)

    if (
        input_path.is_file()
        and input_path.suffix.lower() in _SUPPORTED_BOOK_EXTS
    ):
        return [input_path]
    elif input_path.is_dir():
        books = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_BOOK_EXTS
        )
        if not books:
            print(f"No PDF or EPUB files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(books)} book(s) to convert.\n")
        return books
    else:
        print(f"Not a PDF/EPUB file or directory: {input_path}")
        sys.exit(1)


def clean_markdown_file(md_path):
    """Apply clean_text to an existing markdown file in place.

    Preserves the header (everything before the first <!-- Page marker
    or the first blank line after ---).
    """
    md_path = Path(md_path)
    print(f"Cleaning: {md_path.name}")

    content = md_path.read_text(encoding="utf-8")
    before = content.count('\n')
    cleaned = clean_text(content)
    after = cleaned.count('\n')

    md_path.write_text(cleaned, encoding="utf-8")
    removed = before - after
    print(f"  {removed} lines joined ({before} -> {after})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF books to clean Markdown."
    )
    parser.add_argument(
        "input",
        help="PDF file, EPUB file, markdown file, or directory to convert/clean",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["pymupdf", "pymupdf4llm", "marker", "docling", "ocr"],
        default="pymupdf",
        help="Conversion method (default: pymupdf). pymupdf4llm and docling require Python 3.10+.",
    )
    # Keep --ocr as a shortcut for backwards compatibility
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Shortcut for --method ocr",
    )
    # --papers routes academic papers through marker-pdf, which has much
    # better layout analysis than the default pymupdf path. Requires
    # Python 3.10+ (enforced by check_dependencies) because marker-pdf
    # uses PEP 604 type-hint syntax.
    parser.add_argument(
        "--papers",
        action="store_true",
        help="Shortcut for --method marker (for academic papers; requires Python 3.10+)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing markdown file(s) -- fix orphaned lines and hyphenation. "
             "Pass a .md file or directory of .md files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have a markdown file in the output directory",
    )
    parser.add_argument(
        "--auto-ocr",
        action="store_true",
        help="Automatically retry with OCR if text extraction fails (scanned PDFs)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip dependency check",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the post-conversion cleanup pass (dropped-space de-join, "
             "stray-consonant citation fixes, picture-text garble removal). "
             "Cleanup runs by default and is verbatim-safe.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="After a successful conversion, move the source book from "
             "input/ into archive/ (created if missing). Failed conversions "
             "and --skip-existing skips are left in place.",
    )
    parser.add_argument(
        "--archive-dir",
        default="archive",
        help="Directory to move successfully-converted books into when "
             "--archive is set (default: archive/)",
    )
    parser.add_argument(
        "--extract-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write figures, diagrams, and raster images alongside the "
             "markdown. ON by default for the PDF backends: a backend that "
             "emits a reference to a figure has already done the work of "
             "finding it, and throwing the file away is pure loss. "
             "--no-extract-images converts text only — the references to the "
             "discarded figures are stripped, never left dangling. "
             "EPUB is the exception and defaults OFF, because an epub's images "
             "are overwhelmingly publisher furniture (covers, imprint marks, "
             "promo cards); pass --extract-images explicitly to harvest them "
             "from an epub that really does carry figures.",
    )
    parser.add_argument(
        "--marker-args",
        default="",
        help="Extra flags forwarded verbatim to marker_single (marker backend "
             "only). Quote the whole string, e.g. "
             "--marker-args '--use_llm --html_tables_in_markdown'. Run "
             "`marker_single --help` for the full list; the table-quality "
             "flags are --use_llm (fixes merged headers and split rows, "
             "needs an LLM service configured) and "
             "--html_tables_in_markdown (preserves colspan/rowspan that GFM "
             "cannot express).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging output",
    )

    args = parser.parse_args()

    # Configure logging based on --verbose flag
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="  [%(levelname)s] %(message)s",
    )

    # Clean mode: process existing markdown files
    if args.clean:
        input_path = Path(args.input)
        if input_path.is_file() and input_path.suffix.lower() == ".md":
            clean_markdown_file(input_path)
        elif input_path.is_dir():
            md_files = sorted(input_path.glob("*.md"))
            if not md_files:
                print(f"No .md files found in {input_path}")
                sys.exit(1)
            for md_file in md_files:
                clean_markdown_file(md_file)
            print(f"\nDone. Cleaned {len(md_files)} file(s).")
        else:
            print(f"Not a .md file or directory: {input_path}")
            sys.exit(1)
        sys.exit(0)

    if args.ocr and args.papers:
        print("Error: --ocr and --papers are mutually exclusive")
        sys.exit(1)
    if args.ocr:
        method = "ocr"
    elif args.papers:
        method = "marker"
    else:
        method = args.method

    # shlex so a quoted --marker-args string splits the way a shell would.
    marker_args = shlex.split(args.marker_args) if args.marker_args else None
    if marker_args and method != "marker":
        print(f"Warning: --marker-args is ignored by the {method} backend")

    output_dir = Path(args.output)
    books = collect_books(args.input)

    # Dependency check runs AFTER collection so we only validate the
    # toolchains we actually need: the PDF method only if a PDF is
    # present, pandoc only if an EPUB is present.
    if not args.skip_check:
        needs_pdf_method = any(
            b.suffix.lower() == ".pdf" for b in books
        )
        needs_pandoc = any(
            b.suffix.lower() == ".epub" for b in books
        )
        try:
            if needs_pdf_method:
                check_dependencies(method)
            if needs_pandoc:
                check_dependencies("pandoc")
        except DependencyError as e:
            print(e)
            sys.exit(1)

    # Tell the operator what else is running before anything slow starts.
    # `try/finally` around the loop so a crash or Ctrl-C still de-registers;
    # a stale entry would misreport the next run.
    others = register_conversion(books[0] if books else "?", method, time.time())
    notice = describe_other_conversions(others, time.time())
    if notice:
        print(notice, file=sys.stderr)

    success = 0
    failed = 0
    skipped = 0
    converted_books = []  # successfully converted source paths, for --archive

    try:
        for book in books:
            if args.skip_existing:
                existing = output_dir / f"{book.stem}.md"
                if existing.exists():
                    print(f"Skipping (already exists): {book.name}")
                    skipped += 1
                    continue

            if convert_book(
                book,
                output_dir,
                method=method,
                auto_ocr=args.auto_ocr,
                extract_images=args.extract_images,
                clean=not args.no_clean,
                marker_args=marker_args,
            ):
                success += 1
                converted_books.append(book)
            else:
                failed += 1
            print()
    finally:
        unregister_conversion()

    parts = [f"{success} converted", f"{failed} failed"]
    if skipped:
        parts.append(f"{skipped} skipped")
    print(f"Done. {', '.join(parts)}.")

    # --archive: move successfully-converted source books into the archive
    # dir. Only runs on success; failed conversions stay in input/ so the
    # user can retry. Skip collisions by appending a timestamp suffix so
    # we never overwrite an existing archived file.
    if args.archive and converted_books:
        archive_dir = Path(args.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        collisions = 0
        for src in converted_books:
            dest = archive_dir / src.name
            if dest.exists():
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                dest = archive_dir / f"{src.stem}.{ts}{src.suffix}"
                collisions += 1
            try:
                shutil.move(str(src), str(dest))
                moved += 1
            except Exception as e:
                print(f"  Archive failed for {src.name}: {e}")
        msg = f"Archived {moved} book(s) to {archive_dir}/"
        if collisions:
            msg += f" ({collisions} renamed to avoid collision)"
        print(msg)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
