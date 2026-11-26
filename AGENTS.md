# live-listen

KpiX sovereign live listening: PipeWire capture + local whisper STT + progressive summary via the local OpenVINO server.

## Purpose

Follow a film, course or conference in real time: sound captured in chunks (`pw-record` 16 kHz mono), transcribed locally with whisper.cpp, one progressive summary maintained via the OpenVINO server. Two `.txt` outputs only.

## Structure

```text
src/live_listen/
  __init__.py  # version
  config.py    # defaults + XDG ensure + precedence + path resolution
  pipeline.py  # record/transcribe/summarize/think-filter
  cli.py       # live-listen CLI (argparse, main entry)
tests/
  test_import.py
  test_thinking.py            # strip_thinking + detector
  test_reasoning_regression.py # the two REAL leaks of 2026-09-23, verbatim
  test_audio.py               # RMS energy gate + repetition collapse
  test_robustness.py          # missing binary, atomic write, recorder thread
configs/en-small.json        # the example the README references
Makefile       # install, check, smoke, build, publish, push, release
```

## Rules

- `uv` only, never `pip`.
- Zero runtime dependencies (stdlib only) so `uv tool install` stays instant.
  No `audioop` either : it was removed in Python 3.13.
- Config lives in `~/.local/share/live-listen/config.json` (XDG data dir), created at install (`make install` chains `uv tool install -e .` then `live-listen init`) and on first run (ensure-on-startup). Never in the project dir.
- Models never vendored: shared Whisper store under `~/.local/share/whisper/models/ggml/`, summary model served by OpenVINO (`http://127.0.0.1:11437`).
- No sessions: outputs are `--transcript-path` / `--summary-path` only (defaults `transcription.txt` + `resume.txt` in CWD). `sessions/` is gitignored archive evidence from before that change, never touched by the tool.

## Invariants (a change that breaks one is a bug)

- The capture never blocks on STT: the recorder thread produces chunks while
  the consumer transcribes. `run()` never raises, it returns 127/126/124.
- `resume.txt` is only ever written atomically, and only from a candidate
  that passes the weighted reasoning detector. On any failure the previous
  summary is kept.
- Every config key has a same-named CLI flag, and a loud validation with
  explicit bounds. No silent default, no hardcoded value that also exists in
  the config.
- A regression test must hold the artefact that caused the bug, not a
  paraphrase of it.
