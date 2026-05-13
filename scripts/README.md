# Scripts

Utility scripts for the narrative-latency-tw capstone project.

## Files

| File | Purpose |
|---|---|
| `clean.py` | Load raw IORG CSV, validate schema, save processed CSV |
| `latency.py` | Compute Stage 1->4 deltas, summary stats by cluster, save chart |
| `scrape_iorg.py` | (Stub) IORG scraper — rebuild from fresh session if needed |

## How to run

    cd ~/narrative-latency-tw
    python3 scripts/clean.py
    python3 scripts/latency.py

Outputs land in `data/processed/` and `viz/`.

## Dependencies

    python3 -m pip install --user -r requirements.txt

Current deps: requests, beautifulsoup4, lxml, pandas, matplotlib.

Always use `python3 -m pip` (not `pip3`) — see Problem 7.

## Lessons learned (2026-05-13 session)

### Problem 1: pbpaste truncates multi-line code from Notion chat

Symptom: pasting via `pbpaste > scripts/foo.py` produces a 15-16 line file from a 90+ line source.

Working fix: bypass pbpaste and the terminal entirely. Commit files via the GitHub web editor:

1. github.com/natharzu/narrative-latency-tw → navigate to target folder
2. "Add file" → "Create new file" (or pencil icon to edit existing)
3. Paste content directly into the GitHub editor
4. Commit with a descriptive message
5. `git pull origin main` locally

Proven across 5+ deliverables on 2026-05-13.

### Problem 2: __future__ markdown mangling

Symptom: `from __future__ import annotations` arrives as `from **future** import annotations`.

Cause: Notion renders `__text__` as bold and copies the rendered form.

Fix: GitHub web editor preserves text literally.

### Problem 3: Heredoc stuck at heredoc> prompt

Symptom: pasting `cat << 'PYEOF' ... PYEOF` into zsh leaves the shell waiting.

Cause: closing delimiter must be on its own line with no leading whitespace.

Fix: Ctrl+C to abort. Use the GitHub web path instead.

### Problem 4: Nested folder after re-running clone

Symptom: `narrative-latency-tw/narrative-latency-tw/`.

Recovery:

    mv narrative-latency-tw /tmp/inner-real
    rm -rf narrative-latency-tw
    mv /tmp/inner-real narrative-latency-tw

### Problem 5: git pull --rebase blocked by unstaged changes

Symptom: `error: cannot pull with rebase: You have unstaged changes.`

Fix:

    git add -A
    git commit -m "WIP: local changes before rebase"
    git pull --rebase origin main
    git push

### Problem 6: Push rejected, fetch first

Symptom: `! [rejected] main -> main (fetch first)` — remote has commits you don't have (e.g. from GitHub web edits).

Fix:

    git pull --rebase origin main
    git push

### Problem 7: pip3 installs into wrong Python version

Symptom: `pip3 install matplotlib` reports "Requirement already satisfied" but `python3 -c "import matplotlib"` still fails with ModuleNotFoundError.

Cause: on macOS with multiple Python versions, `pip3` and `python3` may point to different interpreters with separate site-packages. Example seen on this setup:
- `pip3` → Python 3.9 (`Library/Python/3.9/`)
- `python3` → Python 3.13 (`Library/Python/3.13/`)

Fix: always install via the same Python that runs the scripts:

    python3 -m pip install --user matplotlib
    python3 -c "import matplotlib; print(matplotlib.__file__)"

The verify line confirms the install landed in the expected site-packages directory. Use `python3 -m pip` for every install in this project — never bare `pip3`.

## Bug in chat code blocks

When code blocks contain other code blocks (e.g. triple-backtick fences inside a markdown file), the outer fence can close at the first inner triple-backtick. Workaround: use a 4-backtick outer fence, or use 4-space indented code for inner blocks.

## The golden rule

Notion chat → GitHub web editor → `git pull` locally.
Do not paste multi-line code from Notion directly into the terminal.
Tested across 5+ deliverables on 2026-05-13 — works every time.
