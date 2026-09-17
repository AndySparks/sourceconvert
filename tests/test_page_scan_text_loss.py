"""A full-page scan image must never suppress the page's text layer.

WHY THIS EXISTS

2026-08-31. Andy dropped William James's *Principles of Psychology* Vol 1
(704 pages, a scan with a good OCR text layer) into the ingest inbox. The
pymupdf backend produced 9,350 words -- roughly 1% of the book -- and
emitted every single page as `![Figure on page N]` with no prose at all.

The mechanism: `assets.find_raster_regions` reported the page-scan bitmap as
a figure region covering the entire page, and `_extract_page_text_with_regions`
walks the page top-to-bottom emitting region markdown in place of the text
underneath it. A region spanning the full page therefore consumed every
character on it.

What makes this the dangerous class of bug rather than an annoying one is
that NOTHING caught it. `quality_score` read 1.0, the degenerate-text gate
was clean, and the locator gate was clean. `--no-clean` produced byte-identical
output, so the cleanup pass was not involved. The board was green on a 99%
content loss.
"""
import fitz
import pytest

import assets
from tests.fixtures import (
    build_scanned_ocr_pdf,
    build_figure_pdf,
    build_striped_scan_pdf,
)


def test_full_page_scan_is_not_reported_as_a_figure(tmp_path):
    """The bad case: a page-scan bitmap over a text layer is not a figure."""
    pdf = build_scanned_ocr_pdf(tmp_path)
    doc = fitz.open(str(pdf))
    page = doc[0]
    # Sanity: the fixture really does have a full-page raster and real text.
    raster = assets.find_raster_regions(page)
    assert raster, "fixture must carry a raster image"
    assert len(page.get_text().strip()) > 200, "fixture must carry a text layer"

    assert assets.detect_page_scan_document(doc), (
        "a book of page bitmaps with a text layer is a scanned document"
    )
    extracted = assets.extract_page_assets(
        page, "scan", tmp_path / "a", 1, page_scan_document=True
    )
    doc.close()
    assert extracted == [], (
        "a full-page scan bitmap over a text layer is the page itself, not a "
        "figure; emitting it as one costs the page its text"
    )


def test_scanned_book_conversion_keeps_its_text(tmp_path):
    """End to end: the words in the book survive into the markdown."""
    import convert

    pdf = build_scanned_ocr_pdf(tmp_path, pages=5)
    outdir = tmp_path / "out"
    convert.convert_with_pymupdf(pdf, outdir, extract_images=True)
    md = (outdir / f"{pdf.stem}.md").read_text()

    assert "cannon" in md, "body text was dropped by the figure splicer"
    # One occurrence per page, not one image reference per page.
    assert md.count("cannon") == 5, f"expected 5 pages of body, got {md.count('cannon')}"
    assert "Figure on page" not in md, (
        "page scans must not be emitted as figures"
    )


def test_a_real_figure_is_still_extracted(tmp_path):
    """The other half of the gate: legitimate figures must still be found.

    Without this, 'drop full-page regions' could be implemented as 'drop
    every region' and the bad-case test above would still pass.
    """
    pdf = build_figure_pdf(tmp_path)
    doc = fitz.open(str(pdf))
    page = doc[0]
    assert not assets.detect_page_scan_document(doc), (
        "a prose page with a drawn figure is not a scanned document"
    )
    extracted = assets.extract_page_assets(page, "fig", tmp_path / "b", 1)
    doc.close()
    assert len(extracted) == 1, (
        "a genuine sub-page figure must still be extracted"
    )
    _rect, md = extracted[0]
    assert "Figure 1.1" in md, "the caption must still be matched"


def test_a_sub_page_figure_survives_inside_a_scanned_document(tmp_path):
    """The half the first draft of these tests did NOT cover.

    Every other test here exercises `page_scan_document=False` on the
    legitimate cases, so an implementation of "drop every region when the flag
    is set" passed all of them. The filter must drop the page bitmap and
    nothing else, so a genuine sub-page figure has to survive with the flag ON.
    """
    pdf = build_figure_pdf(tmp_path)
    doc = fitz.open(str(pdf))
    extracted = assets.extract_page_assets(
        doc[0], "fig", tmp_path / "d", 1, page_scan_document=True,
    )
    doc.close()
    assert len(extracted) == 1, (
        "the page-scan filter must drop only regions that cover the page; a "
        "sub-page figure inside a scanned book is still a figure"
    )


def test_sampling_spans_the_whole_document(tmp_path):
    """A scan whose front matter is text-only must still be classified.

    `range(0, total, total // 40)` truncated to 40 entries samples only the
    front of the document when the step rounds down to 1, so a 79-page book
    was judged entirely on pages 0-39.
    """
    assert assets._sample_indices(79, 40)[-1] == 78, (
        "sampling must reach the last page"
    )
    assert assets._sample_indices(159, 40)[-1] == 158
    assert assets._sample_indices(3, 40) == [0, 1, 2]
    assert assets._sample_indices(0, 40) == []


def test_a_short_scan_with_sparse_pages_is_still_a_scan(tmp_path):
    """Blank and sparse pages must not drag a real scan below the bar.

    A 3-page scan with one text-heavy page and two sparse ones scored 1/3
    under the first draft and was classified not-a-scan -- which would have
    replaced all three text layers with image references.
    """
    pdf = build_scanned_ocr_pdf(tmp_path, pages=1, name="short.pdf")
    doc = fitz.open(str(pdf))
    # Append two sparse scan pages: full-page bitmap, almost no text.
    src = fitz.open()
    tmp_page = src.new_page(width=100, height=160)
    tmp_page.draw_rect(fitz.Rect(0, 0, 100, 160), color=(0.5, 0.5, 0.5),
                       fill=(0.5, 0.5, 0.5))
    png = tmp_page.get_pixmap(dpi=72).tobytes("png")
    src.close()
    for _ in range(2):
        page = doc.new_page(width=334, height=559)
        page.insert_image(fitz.Rect(0, 0, 334, 559), stream=png)
        page.insert_text((20, 40), "CHAPTER II")
    assert assets.detect_page_scan_document(doc), (
        "sparse pages are not evidence against a scan"
    )
    doc.close()


def test_an_oversized_banner_does_not_cover_the_page(tmp_path):
    """Area alone is not coverage.

    A region twice the page width and 45% of its height has the AREA of 90%
    of the page while covering none of it, and a region hanging off the crop
    box is larger than the page it sits on.
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    banner = fitz.Rect(-200, 0, 600, 270)      # 800 x 270 = 90% of page area
    assert not assets._covers_page(banner, page)
    overhang = fitz.Rect(-50, -50, 450, 650)   # bigger than the page
    assert assets._covers_page(overhang, page), (
        "a bitmap bleeding past the crop box still covers the page"
    )
    doc.close()


def test_image_only_page_still_emits_its_image(tmp_path):
    """A full-page image with NO text layer has nothing else to emit.

    A plate, a fold-out, or a scan that was never OCR'd. Dropping the image
    there would lose the only content the page has.
    """
    pdf = tmp_path / "plate.pdf"
    doc = fitz.open()
    page = doc.new_page(width=334, height=559)
    src = fitz.open()
    tmp_page = src.new_page(width=100, height=160)
    tmp_page.draw_rect(fitz.Rect(0, 0, 100, 160), color=(0.2, 0.2, 0.2),
                       fill=(0.2, 0.2, 0.2))
    png = tmp_page.get_pixmap(dpi=72).tobytes("png")
    src.close()
    page.insert_image(fitz.Rect(0, 0, 334, 559), stream=png)
    doc.save(str(pdf))
    doc.close()

    doc = fitz.open(str(pdf))
    assert not assets.detect_page_scan_document(doc), (
        "no text layer means this is not a scanned-and-OCR'd book, so its "
        "images are the only content it has"
    )
    extracted = assets.extract_page_assets(
        doc[0], "plate", tmp_path / "c", 1,
        page_scan_document=assets.detect_page_scan_document(doc),
    )
    doc.close()
    assert len(extracted) == 1, (
        "an un-OCR'd full-page image is the page's only content and must "
        "still be emitted"
    )


def test_an_unreadable_image_table_is_not_evidence_against_a_scan(tmp_path,
                                                                  monkeypatch):
    """A page we could not inspect must leave the denominator alone.

    Counting it as "no full-page raster" biases the detector toward
    not-a-scan, which is the answer that deletes the text.
    """
    pdf = build_scanned_ocr_pdf(tmp_path, pages=6, name="probe.pdf")
    doc = fitz.open(str(pdf))
    assert assets.detect_page_scan_document(doc), "control: this is a scan"

    real = assets.find_raster_regions
    calls = {"n": 0}

    def flaky(page):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise assets.RasterProbeFailed("simulated")
        return real(page)

    monkeypatch.setattr(assets, "find_raster_regions", flaky)
    assert assets.detect_page_scan_document(doc), (
        "pages whose image table could not be read must be skipped, not "
        "counted as evidence that this is not a scan"
    )
    doc.close()


def test_extract_page_assets_survives_an_unreadable_image_table(tmp_path,
                                                                monkeypatch):
    """Figure extraction keeps its old tolerant behaviour."""
    pdf = build_figure_pdf(tmp_path)
    doc = fitz.open(str(pdf))

    def boom(page):
        raise assets.RasterProbeFailed("simulated")

    monkeypatch.setattr(assets, "find_raster_regions", boom)
    # The vector figure is still found; the unreadable raster table is not
    # allowed to take the whole page down.
    extracted = assets.extract_page_assets(doc[0], "fig", tmp_path / "e", 1)
    doc.close()
    assert len(extracted) == 1


def test_a_scan_sliced_into_strips_is_still_a_scan(tmp_path):
    """A page bitmap stored as a stack of bands must still read as a scan.

    2026-09-17. Sahlman (1990), 'The structure and governance of
    venture-capital organizations' (Journal of Financial Economics 27,
    473-521), came out of Acrobat 3.0 Capture with each page's scan stored as
    21 full-width horizontal strips. `detect_page_scan_document` tested the
    UNMERGED raster regions, so no single region covered the page and the
    document was called born-digital; `extract_page_assets` then merged those
    same strips into one page-covering region and the text splicer dropped the
    text of 44 of 49 pages. Every gate was green: quality_score 1.0,
    degenerate-text clean, 44/44 figures.

    The two functions must test the same geometry.
    """
    pdf = build_striped_scan_pdf(tmp_path)
    doc = fitz.open(str(pdf))
    page = doc[0]
    rasters = assets.find_raster_regions(page)
    assert len(rasters) > 1, "fixture must store the page as several bands"
    assert not any(assets._covers_page(r, page) for r in rasters), (
        "fixture is only interesting while no SINGLE band covers the page"
    )
    merged = assets._merge_rects(rasters, gap=assets.REGION_PADDING)
    assert any(assets._covers_page(r, page) for r in merged), (
        "fixture is only interesting while the MERGED bands do cover the "
        "page -- that asymmetry is the bug"
    )
    assert len(page.get_text().strip()) > 200, "fixture must carry a text layer"

    assert assets.detect_page_scan_document(doc), (
        "bands that merge to the whole page are the page's own bitmap"
    )
    extracted = assets.extract_page_assets(
        page, "striped", tmp_path / "s", 1,
        page_scan_document=assets.detect_page_scan_document(doc),
    )
    doc.close()
    assert extracted == [], "a sliced page scan is the page, not a figure"


def test_striped_scan_conversion_keeps_its_text(tmp_path):
    """End to end on the striped shape: the prose survives."""
    import convert

    pdf = build_striped_scan_pdf(tmp_path, pages=5)
    outdir = tmp_path / "out"
    convert.convert_with_pymupdf(pdf, outdir, extract_images=True)
    md = (outdir / f"{pdf.stem}.md").read_text()

    assert md.count("cannon") == 5, (
        f"expected 5 pages of body, got {md.count('cannon')}"
    )
    assert "Figure on page" not in md, "page scans must not be emitted as figures"
