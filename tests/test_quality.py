"""Tests for the text quality scoring module."""
import convert


# --- Quality scorer tests ---

CLEAN_ENGLISH = """
The organization's leadership must consider the behavior of employees
across every department. A manager who understands the people on the
team can build trust through clear communication. Performance improves
when workers feel their contributions are recognized and their ideas
are taken seriously by the executives. This principle appears in every
major study of management practice over the past fifty years, from the
earliest research on motivation to contemporary work on psychological
safety and team effectiveness. Organizations that ignore it pay a cost
in turnover, engagement, and productivity.
""" * 3  # Repeat to ensure >=100 tokens


MANGLED_TEXT = """
The organi7ation leadersh1p n1ust consider the behavi0r of en1ployees
across every departn1ent. A n1anager vvho understands the peop1e on the
tean1 can bui1d trust through c1ear con1n1unication. Perforn1ance in1proves
vvhen vvorkers fee1 their contribution1s are recogn1ized and their ideas
are taken serious1y by the executi\\/es. This princip1e appears in every
n1ajor study of managen1ent practice over the past fifty years, from the
ear1iest research on n1otivation to conten1porary vvork on psycho1ogica1
safety and tean1 effectiveness. Organi7ations that ignore it pay a cost
in turnover, engagen1ent, and productivity.
""" * 3


def test_quality_clean_english_scores_high():
    """A clean English paragraph scores >= 0.7."""
    score = convert._text_quality_score(CLEAN_ENGLISH)
    assert score >= 0.7, f"Expected >=0.7 for clean English, got {score:.3f}"


def test_quality_mangled_text_scores_low():
    """A mangled paragraph (ligature artifacts) scores < 0.5."""
    score = convert._text_quality_score(MANGLED_TEXT)
    assert score < 0.5, f"Expected <0.5 for mangled text, got {score:.3f}"


def test_quality_short_text_returns_one():
    """Text with fewer than 100 usable tokens returns exactly 1.0."""
    # "the" x 50: 50 tokens total, below the 100 threshold
    short = "the " * 50
    score = convert._text_quality_score(short)
    assert score == 1.0, f"Expected 1.0 for short text, got {score}"


def test_quality_strips_markdown():
    """Markdown syntax characters don't pollute the score."""
    markdown = "# Heading\n\n" + CLEAN_ENGLISH + "\n\n* bullet\n* another\n"
    score = convert._text_quality_score(markdown)
    assert score >= 0.7, f"Markdown should not drag score down, got {score:.3f}"


def test_quality_strips_trailing_punctuation():
    """Words with trailing punctuation (word., word,) are still counted."""
    text = ("organization, management, leadership, business, strategy, "
            "performance, team, people, process, system. " * 15)
    score = convert._text_quality_score(text)
    # Every token after stripping punctuation is a real word -> ~1.0
    assert score >= 0.9, f"Expected >=0.9 for punctuated real words, got {score:.3f}"


def test_quality_strips_page_markers():
    """HTML page markers (<!-- Page N -->) don't contribute tokens."""
    text = "<!-- Page 1 -->\n\n" + CLEAN_ENGLISH
    score = convert._text_quality_score(text)
    assert score >= 0.7


def test_ocr_warnings_detect_rough_text():
    """Text with a high density of OCR-shape errors should produce warnings."""
    rough = """
    BARBAIA SWIX Ine. OSA
    The n1anagen1ent Ine. decision. SWIX OSA was nnade by Ine.
    BARBAIA is a well-known BARBAIA figure. SWIX, Ine., OSA, BARBAIA.
    """ * 10
    warnings = convert._ocr_quality_warnings(rough)
    assert warnings, "Expected at least one OCR warning"


def test_ocr_warnings_empty_for_clean_text():
    clean = ("The organization is well-run and the leader is trusted. " * 30)
    assert convert._ocr_quality_warnings(clean) == []


# --- digit-acronym false positives (2026-09-20) -----------------------------

B2B_PAPER = """
We study how venture capitalists evaluate startups. Firms were classified as
B2B or B2C based on their stated customer. The B2B sample includes enterprise
software, while the B2C sample covers consumer marketplaces. Angels in the B2B
group responded at a higher rate than those in the B2C group, and the P2P
lending startups drew the fewest responses of all. Across every specification
the B2B and B2C coefficients are statistically indistinguishable from zero,
which is the paper's central negative result about sorting on customer type.
""" * 3


MANGLED_ALL_CAPS = """
THE ORGANI7ATION LEADERSH1P N1UST CONSIDER THE BEHAVI0R OF EN1PLOYEES
ACROSS EVERY DEPARTN1ENT. A N1ANAGER VVHO UNDERSTANDS THE PEOP1E ON THE
TEAN1 CAN BUI1D TRUST THROUGH C1EAR CON1N1UNICATION. PERFORN1ANCE IN1PROVES
VVHEN VVORKERS FEE1 THEIR CONTRIBUTION1S ARE RECOGN1IZED AND THEIR IDEAS
ARE TAKEN SERIOUS1Y BY THE EXECUTIVES. THIS PRINCIP1E APPEARS IN EVERY
N1AJOR STUDY OF MANAGEN1ENT PRACTICE OVER THE PAST FIFTY YEARS.
""" * 3


def test_quality_digit_acronyms_are_not_artifacts():
    """B2B/B2C/P2P are business vocabulary, not font-encoding damage.

    Origin 2026-09-20: Gornall & Strebulaev's VC field experiment failed
    conversion on three separate nights. The extraction was perfect -- 156,679
    clean characters off a born-digital pdfTeX file -- but it says "B2B" or
    "B2C" 37 times, every one of which matched the letter-digit-letter
    artifact pattern under re.IGNORECASE. Score 0.00, hard fail, and the error
    told the operator to re-run it through OCR, which would have degraded a
    clean text layer.
    """
    score = convert._text_quality_score(B2B_PAPER)
    assert score >= 0.7, f"Expected >=0.7 for a B2B/B2C paper, got {score:.3f}"


def test_quality_still_catches_mangled_all_caps():
    """The fix must not become "ignore uppercase".

    Dropping re.IGNORECASE would fix the B2B case and silently stop flagging
    mangled text in headings and all-caps pages. The exclusion is scoped to
    known digit-acronym TOKENS, so genuine damage is still caught in any case.
    """
    score = convert._text_quality_score(MANGLED_ALL_CAPS)
    assert score < 0.5, f"Expected <0.5 for all-caps mangled text, got {score:.3f}"
