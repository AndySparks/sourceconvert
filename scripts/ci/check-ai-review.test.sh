#!/usr/bin/env bash
# check-ai-review.test.sh — verification-gate-symmetry for the AI-review
# gate predicate: RED on a conversion-code diff+body missing the section,
# on wrong round counts (1, 12, 2+, 2.5), on empty engine/verdict values,
# on fields scattered outside the one `## AI review` section, on a rename
# laundered into docs/, and on a truncated file listing; GREEN on a
# compliant body AND on an exempt docs-only PR.
set -euo pipefail
GUARD="$(cd "$(dirname "$0")" && pwd)/check-ai-review.sh"

fail() { echo "FAIL: $*"; exit 1; }

COMPLIANT_BODY='Fixes the folio probe.

## AI review — rounds: 2, engine: codex, verdict: pass

Round 1 found 3 actionables; round 2 verified the fixes.

## Notes

Unrelated section.'

# 1. RED: convert.py diff, body with no AI-review section -> exit 1,
#    message names the required format and the ruling (codex round 1, P1:
#    conversion code is the gate's primary target here).
set +e
out=$(printf 'convert.py\n' | PR_BODY='Just a quick fix.' "$GUARD" 2>&1)
rc=$?
set -e
[ "$rc" -eq 1 ] || fail "convert.py diff exited $rc, want 1"
printf '%s\n' "$out" | grep -q '## AI review' || fail "required format not quoted; got: $out"
printf '%s\n' "$out" | grep -q '9.2' || fail "ruling not named; got: $out"
echo "ok 1 - convert.py diff without section fails, quoting format + ruling"

# 2. RED: non-exempt paths, alone — nested .md outside docs/, non-md under
#    docs/, CLAUDE.md, code, corpus
for p in "scripts/fix_frontmatter.py" "tests/test_cleanup.py" \
         ".github/workflows/x.yml" "output/Management/book.md" \
         "8-DECISIONS/2026-01-01-x.md" "docs/notes.txt" "CLAUDE.md" \
         "requirements.txt"; do
  set +e
  printf '%s\n' "$p" | PR_BODY='No review.' "$GUARD" >/dev/null 2>&1 \
    && fail "non-exempt path '$p' passed as docs-only"
  set -e
done
echo "ok 2 - nested .md, docs/ non-md, CLAUDE.md, code + corpus paths never exempt"

# 3. GREEN: conversion-code diff, compliant body (with a trailing unrelated
#    section — the slice must end at the next `## `) -> pass
printf 'convert.py\n' \
  | PR_BODY="$COMPLIANT_BODY" "$GUARD" >/dev/null || fail "compliant body refused"
echo "ok 3 - code diff with compliant section passes"

# 4. GREEN: docs-only diff (docs/**/*.md + root *.md), no section
printf '%s\n' "docs/notes.md" "docs/sub/guide.md" "README.md" "1-ROADMAP.md" \
  | PR_BODY='No review here.' "$GUARD" >/dev/null || fail "docs-only diff refused"
echo "ok 4 - docs-only diff without section is exempt"

# 5. RED: one code file among docs revokes the exemption
set +e
printf '%s\n' "README.md" "report.py" \
  | PR_BODY='No review here.' "$GUARD" >/dev/null 2>&1 && fail "mixed diff passed"
set -e
echo "ok 5 - mixed diff without section fails"

# 6. RED: wrong round counts — 1, 12, 2+, 2.5 (codex r1 P2 + r2 P2)
for bad in \
  '## AI review — rounds: 0, engine: codex, verdict: pass' \
  '## AI review — rounds: 11, engine: codex, verdict: pass' \
  '## AI review — rounds: 12, engine: codex, verdict: pass' \
  '## AI review — rounds: 2+, engine: codex, verdict: pass' \
  '## AI review — rounds: 2.5, engine: codex, verdict: pass' \
  '## AI review — rounds: 2.foo, engine: codex, verdict: pass'; do
  set +e
  printf 'convert.py\n' | PR_BODY="$bad" "$GUARD" >/dev/null 2>&1 \
    && fail "wrong round count passed: $bad"
  set -e
done
echo "ok 6 - rounds: 0 / 11 / 12 / 2+ / 2.5 / 2.foo all fail"

# 6b. GREEN: one round is the current ruling (Andy, 2026-09-01). The gate
# demanded exactly 2 until this fix, which failed sourceconvert PR #59 --
# a correctly reviewed PR -- while mc-wiki's copy of this same predicate had
# already been updated. The ruling covers BOTH repos; only one gate knew.
printf 'convert.py\n' \
  | PR_BODY='## AI review — rounds: 1, engine: codex, verdict: pass' \
    bash "$GUARD" >/dev/null || fail "rounds: 1 refused, but one round is the ruling"
echo "ok 6b - rounds: 1 passes"

# 7. GREEN: sentence-final "rounds: 2." is exactly two (codex round 2, P2)
printf 'convert.py\n' \
  | PR_BODY='## AI review
The review ran for rounds: 2.
engine: codex
verdict: pass' "$GUARD" >/dev/null || fail "sentence-final 'rounds: 2.' refused"
echo "ok 7 - sentence-final 'rounds: 2.' passes"

# 8. RED: empty field VALUES — 'engine: ,' and bare 'verdict:' must not
#    count as records (codex round 2, P1)
for bad in \
  '## AI review — rounds: 2, engine: , verdict: pass' \
  '## AI review — rounds: 2, engine: -, verdict: pass' \
  '## AI review — rounds: 2, engine:,verdict:pass' \
  '## AI review — rounds: 2, engine: codex, verdict:' \
  '## AI review — rounds: 2, verdict: pass' \
  '## AI review — rounds: 2, engine: codex'; do
  set +e
  printf 'convert.py\n' | PR_BODY="$bad" "$GUARD" >/dev/null 2>&1 \
    && fail "incomplete record passed: $bad"
  set -e
done
# GREEN counterpart: the compact no-space form with real values passes.
printf 'convert.py\n' \
  | PR_BODY='## AI review — rounds: 2, engine:codex,verdict:pass' \
    "$GUARD" >/dev/null || fail "compact engine:codex,verdict:pass refused"
echo "ok 8 - empty engine values (incl. engine:,verdict:pass) fail; compact real values pass"

# 9. RED: fields OUTSIDE the one AI-review section supply nothing —
#    split across two sections, or in commented template text
#    (codex round 2, P1)
set +e
printf 'convert.py\n' | PR_BODY='## AI review — rounds: 2

## Notes
engine: codex, verdict: pass' "$GUARD" >/dev/null 2>&1 && fail "split-section fields passed"
set -e
set +e
printf 'convert.py\n' | PR_BODY='<!-- template:
## AI review — rounds: 2, engine: codex, verdict: pass
-->' "$GUARD" >/dev/null 2>&1 && fail "commented template passed"
set -e
# Same-line HTML comment INSIDE the section supplies nothing either
# (#52 direct round, P2).
set +e
printf 'convert.py\n' | PR_BODY='## AI review
<!-- rounds: 2, engine: codex, verdict: pass -->' \
  "$GUARD" >/dev/null 2>&1 && fail "single-line commented template inside section passed"
set -e
echo "ok 9 - fields split across sections / commented templates (block + same-line) fail"

# 10. RED: rename laundering — report.py renamed to docs/report.md; the
#     workflow feeds BOTH sides, and the old side must revoke the exemption
#     (codex round 1, P2)
set +e
printf '%s\n' "docs/report.md" "report.py" \
  | PR_BODY='No review.' "$GUARD" >/dev/null 2>&1 && fail "rename into docs/ passed"
set -e
echo "ok 10 - rename old-side path (report.py -> docs/report.md) revokes exemption"

# 11. RED: truncated file listing (pulls/files caps at 3,000) — a docs-only
#     PAGE of a larger PR must fail closed as mandatory (codex round 2, P2)
set +e
printf 'docs/notes.md\n' \
  | PR_BODY='No review.' PR_CHANGED_FILES=3500 PR_LISTED_FILES=3000 \
    "$GUARD" >/dev/null 2>&1 && fail "truncated listing passed as docs-only"
set -e
echo "ok 11 - truncated listing (listed < changed) fails closed as mandatory"

# 12. GREEN: truncated listing + compliant body still passes (fail-closed
#     means mandatory-review, not unconditional failure); and a matching
#     count keeps the docs-only exemption
printf 'docs/notes.md\n' \
  | PR_BODY="$COMPLIANT_BODY" PR_CHANGED_FILES=3500 PR_LISTED_FILES=3000 \
    "$GUARD" >/dev/null || fail "truncated + compliant body refused"
printf 'docs/notes.md\n' \
  | PR_BODY='No review.' PR_CHANGED_FILES=1 PR_LISTED_FILES=1 \
    "$GUARD" >/dev/null || fail "matching counts lost the exemption"
echo "ok 12 - truncation is mandatory-not-fatal; matching counts stay exempt"

# 13. GREEN: lenient formatting — lowercase heading, fields on separate lines
printf 'convert.py\n' \
  | PR_BODY='## ai review
rounds: 2 (codex round 1 found, round 2 verified)
engine: codex
verdict: pass' "$GUARD" >/dev/null || fail "leniently-formatted body refused"
echo "ok 13 - lenient formatting still passes"

# 14. GREEN: empty diff passes
printf '' | PR_BODY='' "$GUARD" >/dev/null || fail "empty diff refused"
echo "ok 14 - empty diff passes"

# 15. RED: newline-in-filename (#52 verify round, P1) — in base64 mode ONE
#     entry stays ONE path; a filename containing \n that covers a
#     mandatory path is classified intact, never split into exempt lines.
enc=$(printf 'scripts/evil.py\nREADME.md' | base64 | tr -d '\n')
set +e
printf '%s\n' "$enc" \
  | PR_BODY='No review.' PR_PATHS_BASE64=1 "$GUARD" >/dev/null 2>&1 \
  && fail "newline-bearing mandatory filename passed"
set -e
# GREEN counterpart: base64 mode still exempts a plain docs path.
enc=$(printf 'docs/notes.md' | base64 | tr -d '\n')
printf '%s\n' "$enc" \
  | PR_BODY='No review.' PR_PATHS_BASE64=1 "$GUARD" >/dev/null \
  || fail "base64 docs path lost the exemption"
echo "ok 15 - base64 mode: newline-bearing mandatory filename fails; docs path stays exempt"

# 16. RED: labels inside larger words supply nothing (#52 verify, P2)
set +e
printf 'convert.py\n' | PR_BODY='## AI review
backgrounds: 2, notengine: codex, nonverdict: pass' \
  "$GUARD" >/dev/null 2>&1 && fail "embedded labels (backgrounds/notengine/nonverdict) passed"
set -e
echo "ok 16 - backgrounds:/notengine:/nonverdict: are not rounds/engine/verdict"

# 17. RED: heading boundaries (#52 verify, P2) — '## AI reviewer notes'
#     must NOT open the section; a tab-headed '##<TAB>Notes' MUST close it.
set +e
printf 'convert.py\n' \
  | PR_BODY='## AI reviewer notes — rounds: 2, engine: codex, verdict: pass' \
    "$GUARD" >/dev/null 2>&1 && fail "'## AI reviewer notes' opened the section"
set -e
set +e
printf 'convert.py\n' | PR_BODY="$(printf '## AI review — rounds: 2\n##\tNotes\nengine: codex, verdict: pass')" \
  "$GUARD" >/dev/null 2>&1 && fail "tab-headed heading failed to close the section"
set -e
echo "ok 17 - suffixed heading does not open; tab-headed heading closes"

# 18. GREEN: oversized body (#52 verify, P2) — the check must not be
#     killable by its own plumbing (no early-exiting pipeline / SIGPIPE).
#     10k lines (~350 KB) is far past the 64 KB pipe buffer that made an
#     early-exiting awk SIGPIPE its feeder — and >5x GitHub's own 64 KB
#     PR-body cap, so this over-covers anything the workflow can see.
big_tail=$( { yes 'filler line for the oversized-body case'; } | head -n 10000 || true)
printf 'convert.py\n' \
  | PR_BODY="$(printf '## AI review — rounds: 2, engine: codex, verdict: pass\nround detail\n## Tail\n%s' "$big_tail")" \
    "$GUARD" >/dev/null || fail "oversized body refused (plumbing killed the check?)"
echo "ok 18 - oversized body (10k-line tail after the section) passes"


# 19. GREEN: heading depth is cosmetic -- `### AI review` (H3) and `#### AI
#     review` (H4) with all three fields pass. Ported from mc-wiki's copy of
#     this predicate, which hit this as a recurring false-negative (#77
#     there): a fully-reviewed PR rejected only because its section used
#     ### to match its sibling ### blocks.
printf 'convert.py\n' \
  | PR_BODY='### AI review -- rounds: 1, engine: codex, verdict: pass' \
    "$GUARD" >/dev/null || fail "H3 '### AI review' with all fields refused"
printf 'convert.py\n' \
  | PR_BODY='#### AI review -- rounds: 1, engine: codex, verdict: pass' \
    "$GUARD" >/dev/null || fail "H4 '#### AI review' with all fields refused"
echo "ok 19 - ### / #### AI review (deeper headings) with fields pass"

# 20. RED: a same-or-shallower heading still closes the section, at ANY depth
#     (the close must stay depth-aware, not just accept deeper openings).
set +e
printf 'convert.py\n' | PR_BODY='### AI review -- rounds: 1

### Notes
engine: codex, verdict: pass' "$GUARD" >/dev/null 2>&1 \
  && fail "fields after a same-depth (###) closing heading passed"
set -e
set +e
printf 'convert.py\n' | PR_BODY='### AI review -- rounds: 1

## Notes
engine: codex, verdict: pass' "$GUARD" >/dev/null 2>&1 \
  && fail "fields after a shallower (##) closing heading passed"
set -e
echo "ok 20 - a same-or-shallower heading closes the section at any depth"

# 21. GREEN: a DEEPER sub-heading does NOT close the section -- fields under a
#     `#### round detail` inside a `### AI review` still count.
printf 'convert.py\n' | PR_BODY='### AI review -- rounds: 1
#### round detail
engine: codex, verdict: pass' "$GUARD" >/dev/null \
  || fail "a deeper sub-heading wrongly closed the section"
echo "ok 21 - a deeper sub-heading stays inside the section"

echo "PASS check-ai-review"

# 22. RED: a heading with more than six hashes is not a real ATX heading
#     (CommonMark caps at 6), so it must not open the AI-review section --
#     codex round 2, P2, against the generalized /#+/ matcher.
set +e
printf 'convert.py\n' | PR_BODY='####### AI review -- rounds: 1, engine: codex, verdict: pass' \
  "$GUARD" >/dev/null 2>&1 && fail "a seven-hash line was accepted as an AI-review heading"
set -e
echo "ok 22 - a heading with more than six hashes does not open the section"
