"""Synthetic PDF builders for tests.

Every test that needs a PDF calls one of these builders instead of
committing a binary fixture to the repo. Each builder returns a Path
to a file inside the tmp_path fixture directory.
"""
import zipfile
from pathlib import Path

import fitz


def build_text_pdf(tmp_path: Path, pages: int = 3, body: str = "Lorem ipsum dolor sit amet.") -> Path:
    """Build a plain-text PDF with N pages of the given body text."""
    out = tmp_path / "text.pdf"
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i + 1}")
        page.insert_text((72, 120), body)
    doc.save(str(out))
    doc.close()
    return out


def build_figure_pdf(tmp_path: Path) -> Path:
    """Build a PDF with one page containing body text, a drawn rectangle
    standing in for a figure, and a 'Figure 1.1' caption below it.
    """
    out = tmp_path / "figure.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Introduction to Thinking")
    page.insert_text((72, 120),
                     "This chapter presents the basic model we will use.")
    # Draw a rectangle that stands in for a figure.
    rect = fitz.Rect(150, 200, 460, 450)
    page.draw_rect(rect, color=(0, 0, 0), width=2)
    page.draw_line((150, 325), (460, 325), color=(0, 0, 0), width=1)
    page.draw_line((305, 200), (305, 450), color=(0, 0, 0), width=1)
    # Caption.
    page.insert_text((72, 480), "Figure 1.1 The four-quadrant model.")
    page.insert_text((72, 520),
                     "The quadrants represent the two primary axes.")
    doc.save(str(out))
    doc.close()
    return out


def build_page_printed_pdf(
    tmp_path: Path,
    pages: int = 8,
    offset: int = -4,
    skip_page_printed_on: tuple = (),
    body_only_footer_on: tuple = (),
    misread_page_printed_on: dict = None,
    name: str = "page_printed.pdf",
) -> Path:
    """Build a PDF whose pages carry a printed page number in the footer.

    PDF page i (1-based) prints page number i + offset. With the default
    offset of -4, PDF page 5 prints "1" — i.e. four pages of unnumbered
    front matter, the common real-world shape. Pages whose computed
    printed number is < 1 print no footer at all, standing in for a cover
    and title page.

    `skip_page_printed_on` names PDF pages that print NO footer even
    though their printed number would be >= 1 — standing in for the
    real-world case where the printed number was lost to extraction (OCR
    miss, figure-covered footer). Those pages are the ones interpolation
    must fill in.

    `body_only_footer_on` names PDF pages that carry their printed footer
    but NO body text — the numbered-but-blank verso of a part divider.
    After header stripping those pages clean to nothing and are dropped
    before a locator is emitted, so they must not count toward the
    page_printed_coverage denominator.

    `misread_page_printed_on` maps a PDF page to the WRONG string printed
    in its footer — a digit OCR'd badly, or a body numeral lifted off an
    appendix opener. Those pages are captured with full confidence and are
    the ones the consensus rule must suppress rather than publish.
    """
    out = tmp_path / name
    doc = fitz.open()
    skip = set(skip_page_printed_on)
    footer_only = set(body_only_footer_on)
    misread = dict(misread_page_printed_on or {})
    for i in range(1, pages + 1):
        page = doc.new_page(width=612, height=792)
        if i not in footer_only:
            page.insert_text((72, 100), f"Body text for page {i}.")
        page_printed = i + offset
        if i in misread:
            page.insert_text((300, 740), str(misread[i]))
        elif page_printed >= 1 and i not in skip:
            page.insert_text((300, 740), str(page_printed))
    doc.save(str(out))
    doc.close()
    return out


def build_trailing_number_pdf(
    tmp_path: Path,
    pages: int = 8,
    name: str = "trailing.pdf",
) -> Path:
    """Build a PDF whose body text legitimately ENDS in a number.

    No page carries a printed page number. Every page's last line of prose
    ends with a year, standing in for the very common real-world shape
    where a page's closing sentence ends in a number and the real footer
    never survived extraction. Nothing here was ever printed as a page
    number, so every page must report `page_printed=none`.

    The closing sentences deliberately DIFFER from page to page. An
    identical closing line would be detected as a repeating running footer
    and removed whole, so the trailing-number branch this fixture exists to
    exercise would never be reached.
    """
    prose = [
        "The study was published in",
        "The print run sold roughly",
        "The trial ran for",
        "The firm employed some",
        "The division grew to",
        "The strike lasted about",
        "The audience reached nearly",
        "The revision shipped in",
        "The survey covered exactly",
        "The programme spanned some",
        "The refit cost about",
        "The agency hired around",
    ]
    out = tmp_path / name
    doc = fitz.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Body text for page {i}.")
        page.insert_text(
            (72, 130), f"{prose[(i - 1) % len(prose)]} {1990 + i}"
        )
    doc.save(str(out))
    doc.close()
    return out


def build_roman_wordlike_pdf(tmp_path: Path, name: str = "romanish.pdf") -> Path:
    """Build a PDF with standalone lines that LOOK like roman numerals.

    "I", "ill" and "civil" are all matched by the loose roman stripping
    regex `[ivxlc]{1,7}`, but they are English words sitting on their own
    line (a dropped caption, a hyphenation artifact, a one-word line). No
    page here carries a printed number, so the book must come out
    pdf_only.
    """
    out = tmp_path / name
    doc = fitz.open()
    words = ["I", "ill", "civil", "ill", "civil", "I", "ill", "civil"]
    for i in range(1, 9):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), words[(i - 1) % len(words)])
        page.insert_text((72, 130), f"Body text for page {i}.")
    doc.save(str(out))
    doc.close()
    return out


def build_renumbering_pdf(tmp_path: Path, name: str = "renumber.pdf") -> Path:
    """Build a PDF whose printed numbering RESTARTS partway through.

    Pages 3-8 print page numbers 1-6 (offset -2). Page 9 onward is a
    second section restarting at 1 (offset -8), the shape of endnotes or a
    part-opener that resets. No single constant offset explains both runs,
    so `_derive_page_offset` must refuse and NOTHING may be interpolated.

    Pages 12 and 14 print no footer at all — those are the uncaptured
    pages a buggy implementation would fill with confident wrong numbers.
    """
    out = tmp_path / name
    doc = fitz.open()
    for i in range(1, 17):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Body text for page {i}.")
        if i in (12, 14):
            continue  # printed number lost to extraction
        if 3 <= i <= 8:
            page.insert_text((300, 740), str(i - 2))
        elif i >= 9:
            page.insert_text((300, 740), str(i - 8))
    doc.save(str(out))
    doc.close()
    return out


def build_raster_image_pdf(tmp_path: Path) -> Path:
    """Build a PDF with one page containing a real embedded raster image.

    Uses a tiny 4x4 solid-colored PNG so the test runs fast.
    """
    out = tmp_path / "raster.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Chapter 1")
    # Create a 4x4 red PNG in-memory via a pixmap.
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4))
    pix.clear_with(255)  # white background
    img_bytes = pix.tobytes("png")
    rect = fitz.Rect(200, 150, 400, 350)
    page.insert_image(rect, stream=img_bytes)
    page.insert_text((72, 380), "Figure 1.1 A square.")
    page.insert_text((72, 420), "The image above is the square we will study.")
    doc.save(str(out))
    doc.close()
    return out


def build_scanned_pdf(tmp_path: Path) -> Path:
    """Build a PDF whose pages contain ONLY images (no extractable text).

    Used to test that extraction-only backends fail cleanly and that the
    OCR fallback routing picks it up.
    """
    out = tmp_path / "scanned.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 612, 792))
        pix.clear_with(240)
        img_bytes = pix.tobytes("png")
        page.insert_image(page.rect, stream=img_bytes)
    doc.save(str(out))
    doc.close()
    return out


def build_minimal_epub(tmp_path: Path, name: str = "minimal.epub") -> Path:
    """Build the smallest valid EPUB 2 that pandoc will convert.

    An EPUB is a zip with an uncompressed `mimetype` entry first, a
    META-INF/container.xml pointing at the OPF package, and at least one
    XHTML content document. Synthesized rather than committed so the repo
    keeps no binary fixtures.
    """
    out = tmp_path / name

    container = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )
    opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
        'unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:title>Minimal Book</dc:title>\n'
        '    <dc:language>en</dc:language>\n'
        '    <dc:identifier id="bookid">urn:uuid:sourceconvert-test</dc:identifier>\n'
        '  </metadata>\n'
        '  <manifest>\n'
        '    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>\n'
        '    <item id="ncx" href="toc.ncx" '
        'media-type="application/x-dtbncx+xml"/>\n'
        '  </manifest>\n'
        '  <spine toc="ncx">\n'
        '    <itemref idref="ch1"/>\n'
        '  </spine>\n'
        '</package>\n'
    )
    ncx = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
        '  <head><meta name="dtb:uid" content="urn:uuid:sourceconvert-test"/></head>\n'
        '  <docTitle><text>Minimal Book</text></docTitle>\n'
        '  <navMap>\n'
        '    <navPoint id="np1" playOrder="1">\n'
        '      <navLabel><text>Chapter One</text></navLabel>\n'
        '      <content src="ch1.xhtml"/>\n'
        '    </navPoint>\n'
        '  </navMap>\n'
        '</ncx>\n'
    )
    ch1 = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        '  <head><title>Chapter One</title></head>\n'
        '  <body>\n'
        '    <h1>Chapter One</h1>\n'
        '    <p>An epub is reflowable, so it has no printed page numbers.</p>\n'
        '  </body>\n'
        '</html>\n'
    )

    with zipfile.ZipFile(out, "w") as zf:
        # The mimetype entry must be first and stored uncompressed.
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/toc.ncx", ncx)
        zf.writestr("OEBPS/ch1.xhtml", ch1)
    return out


# --- configurable EPUB builder ------------------------------------------
#
# `build_minimal_epub` above is the one-chapter smoke fixture. The builder
# below takes explicit content documents and an explicit nav so a test can
# construct the exact structural shape it needs: semantic headings, no
# headings but a good toc.ncx, an EPUB 3 nav document, chapter-ish CSS
# classes, or nothing at all. All content is invented for the tests.

_CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <rootfiles>\n'
    '    <rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/>\n'
    '  </rootfiles>\n'
    '</container>\n'
)


def xhtml_doc(title: str, body: str) -> str:
    """Wrap a body fragment in a minimal XHTML content document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops">\n'
        '  <head><title>%s</title></head>\n'
        '  <body>\n%s\n  </body>\n'
        '</html>\n' % (title, body)
    )


def _nest(entries):
    """Turn a flat [(depth, label, href), ...] list into a nested tree."""
    root = []
    stack = [(0, root)]
    for depth, label, href in entries:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        children = []
        stack[-1][1].append((label, href, children))
        stack.append((depth, children))
    return root


def _ncx_points(tree, counter=None):
    counter = counter if counter is not None else [0]
    out = []
    for label, href, children in tree:
        counter[0] += 1
        out.append(
            '<navPoint id="np%d" playOrder="%d">'
            '<navLabel><text>%s</text></navLabel>'
            '<content src="%s"/>%s</navPoint>'
            % (counter[0], counter[0], label, href,
               "".join(_ncx_points(children, counter)))
        )
    return out


def _nav_list(tree):
    items = "".join(
        '<li><a href="%s">%s</a>%s</li>'
        % (href, label, _nav_list(children) if children else "")
        for label, href, children in tree
    )
    return "<ol>%s</ol>" % items


def build_epub(
    tmp_path: Path,
    docs,
    nav=None,
    nav_style: str = "ncx",
    name: str = "synthetic.epub",
    title: str = "Synthetic Book",
    nav_in_spine: bool = False,
) -> Path:
    """Build an EPUB from explicit content documents and an explicit nav.

    `docs` is a list of (filename, body markup) pairs, spine order.
    `nav` is a flat list of (depth, label, href) where depth is 1-based
    nesting and href is relative to OEBPS/ (may carry a `#anchor`).
    `nav_style` is "ncx" (EPUB 2), "nav" (EPUB 3 nav document), or "none".
    `nav_in_spine` puts the EPUB 3 nav document in the spine as a readable
    contents page, which is what most real EPUB 3s do — and which is how the
    nav's own `<h2>Contents</h2>` can masquerade as a semantic heading.
    """
    out = tmp_path / name
    tree = _nest(nav or [])

    manifest = [
        '<item id="doc%d" href="%s" media-type="application/xhtml+xml"/>'
        % (i, filename)
        for i, (filename, _) in enumerate(docs)
    ]
    spine_items = ['<itemref idref="doc%d"/>' % i for i in range(len(docs))]
    extra_files = {}
    spine_attr = ""

    if nav_style == "ncx":
        manifest.append(
            '<item id="ncx" href="toc.ncx" '
            'media-type="application/x-dtbncx+xml"/>'
        )
        spine_attr = ' toc="ncx"'
        extra_files["OEBPS/toc.ncx"] = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" '
            'version="2005-1">\n'
            '  <head><meta name="dtb:uid" content="urn:uuid:bc-test"/></head>\n'
            '  <docTitle><text>%s</text></docTitle>\n'
            '  <navMap>%s</navMap>\n'
            '</ncx>\n' % (title, "".join(_ncx_points(tree)))
        )
    elif nav_style == "nav":
        manifest.append(
            '<item id="navdoc" href="nav.xhtml" properties="nav" '
            'media-type="application/xhtml+xml"/>'
        )
        extra_files["OEBPS/nav.xhtml"] = xhtml_doc(
            "Contents",
            '<nav epub:type="toc" id="toc"><h2>Contents</h2>%s</nav>'
            % _nav_list(tree),
        )
        if nav_in_spine:
            spine_items.insert(0, '<itemref idref="navdoc"/>')

    version = "3.0" if nav_style == "nav" else "2.0"
    opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="%s" '
        'unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:title>%s</dc:title>\n'
        '    <dc:language>en</dc:language>\n'
        '    <dc:identifier id="bookid">urn:uuid:bc-test</dc:identifier>\n'
        '  </metadata>\n'
        '  <manifest>%s</manifest>\n'
        '  <spine%s>%s</spine>\n'
        '</package>\n'
        % (version, title, "".join(manifest), spine_attr,
           "".join(spine_items))
    )

    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", _CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", opf)
        for path, text in extra_files.items():
            zf.writestr(path, text)
        for filename, body in docs:
            zf.writestr("OEBPS/" + filename, xhtml_doc(filename, body))
    return out


# Three canonical structural shapes, used across the heading-fallback tests.

def build_semantic_epub(tmp_path: Path, name: str = "semantic.epub") -> Path:
    """An EPUB that carries real `<h1>` chapter openers — the normal path."""
    docs = [
        ("ch1.xhtml",
         '    <h1>The Opening Move</h1>\n'
         '    <p>A manager decides what the team will not do.</p>'),
        ("ch2.xhtml",
         '    <h1>The Second Move</h1>\n'
         '    <p>Then she tells them, in words they can repeat.</p>'),
    ]
    nav = [
        (1, "The Opening Move", "ch1.xhtml"),
        (1, "The Second Move", "ch2.xhtml"),
    ]
    return build_epub(tmp_path, docs, nav, "ncx", name=name)


def build_navless_headingless_epub(
    tmp_path: Path, name: str = "nav_only.epub", nav_style: str = "ncx"
) -> Path:
    """No `h1`-`h6` anywhere, but a complete and correct nav.

    This is the Landsberg shape from issue #27: chapter openers styled as
    `<p class="chaphead">`, with the real chapter list living only in the
    navigation. Nesting is two-deep so heading depth can be checked.
    """
    docs = [
        ("front.xhtml",
         '    <p class="chaphead" id="intro">Introduction: A Broader Repertoire</p>\n'
         '    <p>Most managers own one move and use it everywhere.</p>'),
        ("part1.xhtml",
         '    <p class="chaphead" id="part1">Part One: Foundations</p>\n'
         '    <p>Two ideas do most of the work in this book.</p>\n'
         '    <p class="chaphead" id="c1">Chapter 1: Attention</p>\n'
         '    <p>Where a manager looks is where the team looks.</p>'),
        ("part2.xhtml",
         '    <p class="chaphead" id="c2">Chapter 2: Rhythm</p>\n'
         '    <p>A weekly beat beats a quarterly heroic.</p>'),
    ]
    nav = [
        (1, "Introduction: A Broader Repertoire", "front.xhtml#intro"),
        (1, "Part One: Foundations", "part1.xhtml#part1"),
        (2, "Chapter 1: Attention", "part1.xhtml#c1"),
        (2, "Chapter 2: Rhythm", "part2.xhtml#c2"),
    ]
    return build_epub(tmp_path, docs, nav, nav_style, name=name)


def build_structureless_epub(
    tmp_path: Path, name: str = "flat.epub", chapterish: bool = False
) -> Path:
    """No headings and no usable nav — the worst case.

    With `chapterish=True` the chapter openers still carry a recognizable
    `class="chaphead"`, so the class heuristic has something to find; with
    `chapterish=False` there is nothing to recover at all.
    """
    cls = ' class="chaphead"' if chapterish else ' class="bodytext"'
    docs = [
        ("ch1.xhtml",
         '    <p%s>Chapter 1: The Flat Book</p>\n'
         '    <p>Nothing here announces itself as a heading.</p>' % cls),
        ("ch2.xhtml",
         '    <p%s>Chapter 2: Still Flat</p>\n'
         '    <p>Nor here. The nav is empty too.</p>' % cls),
    ]
    return build_epub(tmp_path, docs, nav=None, nav_style="none", name=name)


# --- marker stand-in -------------------------------------------------------
#
# A real marker run on a book is 25+ minutes and needs model weights. What the
# asset invariant depends on is not marker's cleverness but its *output shape*:
# a markdown file that references figures by bare `.jpeg` filename, and the
# figure files sitting beside it in marker's own scratch directory. That shape
# is what these fixtures reproduce.

MARKER_FIGURE_PAGE = """\
{{{page}}}------------------------------------------------

## Chapter {page}

The model below is the one the rest of the chapter argues from.

![]({figure})

*Figure {page}.1 The four-quadrant model.*

Managers who skip it tend to skip the argument with it.
"""


def fake_marker_output(out_dir: Path, stem: str, pages=(3, 7),
                       write_images: bool = True,
                       suffix: str = ".jpeg") -> Path:
    """Write a directory shaped like a real marker_single run.

    marker writes `<out_dir>/<stem>/<stem>.md` plus its figure files as
    siblings of that markdown, named `_page_N_Figure_M.jpeg`. The references
    in the markdown are bare filenames with no directory component — which is
    why a rewrite regex that requires a `/` never matched them.

    With `write_images=False` the references are emitted and the files are
    not, which is precisely the state issue #34 describes.
    """
    doc_dir = Path(out_dir) / stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    body = []
    for page in pages:
        figure = f"_page_{page}_Figure_1{suffix}"
        body.append(MARKER_FIGURE_PAGE.format(page=page, figure=figure))
        if write_images:
            # A one-pixel PNG is a real file with real bytes; the suffix is
            # what the harvesting code keys on, and lying about it is the
            # point of the test.
            (doc_dir / figure).write_bytes(_ONE_PIXEL_PNG)
    md = doc_dir / f"{stem}.md"
    md.write_text("\n".join(body), encoding="utf-8")
    return md


# Smallest valid PNG: 1x1, opaque white.
_ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000c4944415408d763f8cfc0000003010100189dd6e1000000"
    "0049454e44ae426082"
)


def patch_marker(monkeypatch, convert_module, **kwargs):
    """Make `convert_with_marker` run against `fake_marker_output`.

    Stubs the subprocess.Popen call so no marker, no model weights, and no
    25-minute wait are involved. Extra kwargs are forwarded to
    `fake_marker_output`; `--disable_image_extraction` on the command line
    overrides `write_images`, the way real marker would.
    """
    def fake_popen(cmd, *args, **popen_kwargs):
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        stem = Path(cmd[1]).stem
        opts = dict(kwargs)
        if "--disable_image_extraction" in cmd:
            opts["write_images"] = False
            opts["emit_refs"] = opts.get("emit_refs", False)
        if opts.pop("emit_refs", True):
            fake_marker_output(out_dir, stem, **opts)
        else:
            # marker with image extraction disabled drops the references too.
            doc_dir = out_dir / stem
            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / f"{stem}.md").write_text(
                "{3}------------------------------------------------\n\n"
                "## Chapter 3\n\nText only.\n",
                encoding="utf-8",
            )
        return _FakeProc()

    monkeypatch.setattr(convert_module.subprocess, "Popen", fake_popen)


class _FakeProc:
    """Just enough of Popen for convert_with_marker's streaming loop."""

    def __init__(self, lines=("Loaded detection model\n", "Saved output\n")):
        self.stdout = iter(lines)

    def wait(self):
        return 0


def build_figure_epub(tmp_path: Path, name: str = "figure.epub") -> Path:
    """An EPUB whose chapters carry `<img>` tags with alt text.

    Pandoc turns these into markdown image references pointing at the epub's
    internal media paths — for images sourceconvert never writes out. Alt text
    matters: `_clean_pandoc_output` already drops empty-alt references on
    their own line, so an alt-bearing image is the one that survives to
    dangle. This is the epub half of issue #34.
    """
    docs = [
        ("ch1.xhtml",
         '    <h1>The Opening Move</h1>\n'
         '    <p>She names the work before anyone touches it.</p>\n'
         '    <p><img src="images/fig1.png" alt="Figure 1.1 The loop"/></p>'),
        ("ch2.xhtml",
         '    <h1>The Second Move</h1>\n'
         '    <p>Then she tells them, in words they can repeat.</p>\n'
         '    <p><img src="images/fig2.png" alt="Figure 2.1 The ladder"/></p>'),
    ]
    nav = [
        (1, "The Opening Move", "ch1.xhtml"),
        (1, "The Second Move", "ch2.xhtml"),
    ]
    return build_epub(tmp_path, docs, nav, "ncx", name=name)


def build_scanned_ocr_pdf(
    tmp_path: Path,
    pages: int = 3,
    body: str = "The real sound of the cannon is the sensation it makes.",
    name: str = "scanned_ocr.pdf",
) -> Path:
    """Build a PDF shaped like a home-scanned book with an OCR text layer.

    Every page is one full-page raster image (the scan) with a text layer
    sitting on top of it -- exactly what a Brother ADS-4700W scan that has
    been OCR'd elsewhere looks like, and what archive.org serves. The
    distinguishing feature is that the image bbox covers the WHOLE page,
    so a naive figure-region splicer replaces the page's text with an
    image reference and silently drops the book.
    """
    out = tmp_path / name
    doc = fitz.open()
    # A small grey PNG, stretched to fill the page. Content does not matter;
    # the bbox covering the full page is the whole point.
    src = fitz.open()
    tmp_page = src.new_page(width=100, height=160)
    tmp_page.draw_rect(fitz.Rect(0, 0, 100, 160), color=(0.5, 0.5, 0.5),
                       fill=(0.5, 0.5, 0.5))
    png = tmp_page.get_pixmap(dpi=72).tobytes("png")
    src.close()

    for i in range(pages):
        page = doc.new_page(width=334, height=559)
        page.insert_image(fitz.Rect(0, 0, 334, 559), stream=png)
        # The OCR text layer: a running folio and a page's worth of prose.
        # The volume matters -- the detector asks whether a page carries a
        # real text layer, and a caption-sized string is not one.
        page.insert_text((20, 20), str(36 + i))
        # Varied per page: an identical line on every page is a running
        # header, and the converter is right to strip it.
        page.insert_text((20, 60), f"{body} ({i})")
        y = 90
        for n in range(14):
            page.insert_text(
                (20, y),
                f"line {n} of ordinary body prose running the measure of "
                f"the page as scanned book text does",
                fontsize=8,
            )
            y += 14
    doc.save(str(out))
    doc.close()
    return out


def build_striped_scan_pdf(
    tmp_path: Path,
    pages: int = 3,
    strips: int = 21,
    body: str = "The real sound of the cannon is the sensation it makes.",
    name: str = "striped_scan.pdf",
) -> Path:
    """Build a scan whose page bitmap is SLICED into horizontal strips.

    Acrobat Capture (and several journal-archive pipelines) store a page scan
    as a stack of full-width bands rather than one image. Each band is a
    legitimate raster region; none of them covers the page on its own, but
    merged they are the page. A detector that tests the unmerged regions sees
    no full-page bitmap and calls the book born-digital, while the extractor
    -- which merges first -- splices the merged band over the whole page and
    eats its text.
    """
    out = tmp_path / name
    doc = fitz.open()
    src = fitz.open()
    tmp_page = src.new_page(width=100, height=16)
    tmp_page.draw_rect(fitz.Rect(0, 0, 100, 16), color=(0.5, 0.5, 0.5),
                       fill=(0.5, 0.5, 0.5))
    png = tmp_page.get_pixmap(dpi=72).tobytes("png")
    src.close()

    width, height = 334.0, 559.0
    band = height / strips
    for i in range(pages):
        page = doc.new_page(width=width, height=height)
        for s in range(strips):
            # keep_proportion=False: the band must span the FULL page width,
            # the way a real capture's bands do. Letterboxed bands merge to a
            # rect narrower than the page and the fixture stops reproducing
            # the bug it exists for.
            page.insert_image(
                fitz.Rect(0, s * band, width, (s + 1) * band), stream=png,
                keep_proportion=False,
            )
        page.insert_text((20, 20), str(36 + i))
        page.insert_text((20, 60), f"{body} ({i})")
        y = 90
        for n in range(14):
            page.insert_text(
                (20, y),
                f"line {n} of ordinary body prose running the measure of "
                f"the page as scanned book text does",
                fontsize=8,
            )
            y += 14
    doc.save(str(out))
    doc.close()
    return out
