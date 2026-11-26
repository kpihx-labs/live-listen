"""Robustness and fluidity : every fix for a crash, a hang, or a silent trap.

Covers : a missing binary returning a code instead of a traceback, the atomic
summary write, the prompt bound, sink parsing, the /no_think directive, the
STT energy gate, and the recorder thread that keeps the microphone busy while
the previous chunk is transcribed.
"""

import os
import threading
import time

from live_listen.config import BUILTIN_FALLBACK
from live_listen.pipeline import (
    EXIT_NOT_FOUND,
    EXIT_TIMEOUT,
    Recorder,
    bound_new_segments,
    build_system_prompt,
    default_sink_name,
    require_binary,
    run,
    vad_args,
    warn_sink_mismatch,
    whisper_device,
    whisper_device_args,
    write_atomic,
)


def test_whisper_device_cpu_is_explicit():
    assert whisper_device("cpu") == "cpu"
    assert whisper_device_args("cpu") == ("cpu", ["--no-gpu"])


def test_whisper_device_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        whisper_device("magic")

# --- run() never raises -----------------------------------------------------

def test_run_missing_binary_returns_code_not_exception():
    """A missing whisper-cli used to kill the whole capture loop with a
    traceback. It must be a return code the caller can branch on."""
    code, _out, err = run(["definitely-not-a-real-binary-xyz"])
    assert code == EXIT_NOT_FOUND
    assert "introuvable" in err


def test_run_timeout_returns_124():
    code, _out, _err = run(["sleep", "5"], timeout=1)
    assert code == EXIT_TIMEOUT


def test_run_captures_output():
    code, out, _err = run(["echo", "salut"])
    assert code == 0
    assert out.strip() == "salut"


def test_require_binary_true_false():
    assert require_binary("sh") is True
    assert require_binary("definitely-not-a-real-binary-xyz") is False


# --- write_atomic -----------------------------------------------------------

def test_write_atomic_creates_file_with_content(tmp_path):
    target = str(tmp_path / "resume.txt")
    write_atomic(target, "contenu")
    with open(target, encoding="utf-8") as f:
        assert f.read() == "contenu"


def test_write_atomic_overwrites_and_leaves_no_temp_file(tmp_path):
    target = str(tmp_path / "resume.txt")
    write_atomic(target, "un")
    write_atomic(target, "deux")
    with open(target, encoding="utf-8") as f:
        assert f.read() == "deux"
    assert [n for n in os.listdir(str(tmp_path)) if n != "resume.txt"] == []


def test_write_atomic_creates_missing_directory(tmp_path):
    target = str(tmp_path / "sous" / "dossier" / "resume.txt")
    write_atomic(target, "contenu")
    assert os.path.exists(target)


# --- bound_new_segments -----------------------------------------------------

def test_bound_new_segments_keeps_short_text():
    text, truncated = bound_new_segments("court", 100)
    assert text == "court"
    assert truncated is False


def test_bound_new_segments_keeps_the_tail():
    """A live summary cares about what was just said, so the tail is kept."""
    text, truncated = bound_new_segments("0123456789", 4)
    assert text == "6789"
    assert truncated is True


def test_bound_new_segments_zero_disables():
    text, truncated = bound_new_segments("0123456789", 0)
    assert text == "0123456789"
    assert truncated is False


def test_bound_new_segments_bounds_an_unreachable_server_growth():
    """Without this, a server that stays down makes the prompt grow on every
    cycle : quadratic tokens, then a hard context error."""
    growing = "x" * 50000
    text, truncated = bound_new_segments(growing, 4000)
    assert truncated is True
    assert len(text) == 4000


# --- default_sink_name ------------------------------------------------------

WPCTL_STATUS = r"""
PipeWire 'pipewire-0' [1.0.0, root@host, cookie:1234]
 |- Sinks:
 |  *   51.alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink
 |
 \- Sources:
    54.alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source

Settings
 \- Default Configured Devices:
         0. Audio/Sink    bluez_output.41:42:2A:F6:F7:9A
         1. Audio/Source    alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source
"""


def test_default_sink_name_finds_the_configured_device():
    assert default_sink_name(WPCTL_STATUS) == "bluez_output.41:42:2A:F6:F7:9A"


def test_default_sink_name_none_when_absent():
    assert default_sink_name("rien ici") is None
    assert default_sink_name("") is None


def test_warn_sink_mismatch_flags_the_bluetooth_trap(capsys):
    """The exact configuration measured on kpihx-ubuntu : the default sink was
    Bluetooth while monitor_target pointed at the internal Intel speaker, so a
    session would record pure silence for its whole duration."""
    cfg = dict(BUILTIN_FALLBACK)
    cfg["monitor_target"] = (
        "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic"
        ".HiFi__Speaker__sink.monitor")
    warn_sink_mismatch(cfg)
    out = capsys.readouterr().out
    assert "COHERENCE SON" in out
    assert "bluez_output." in out


# --- build_system_prompt ----------------------------------------------------

def test_build_system_prompt_prefixes_no_think():
    """enable_thinking=false is not authoritative on the served Qwen3
    checkpoint, which still self-emits a <think> block. /no_think is the
    template directive that survives it."""
    cfg = dict(BUILTIN_FALLBACK, thinking=False, think_directive=True)
    assert build_system_prompt(cfg, "consigne").startswith("/no_think\n")


def test_build_system_prompt_omits_directive_when_thinking_requested():
    cfg = dict(BUILTIN_FALLBACK, thinking=True, think_directive=True)
    assert build_system_prompt(cfg, "consigne") == "consigne"


def test_build_system_prompt_omits_directive_when_disabled():
    cfg = dict(BUILTIN_FALLBACK, thinking=False, think_directive=False)
    assert build_system_prompt(cfg, "consigne") == "consigne"


def test_build_system_prompt_prefixes_only_once():
    """build_system_prompt is applied by the CLI, and the retry inside
    summarize_once only appends : the directive must never stack."""
    cfg = dict(BUILTIN_FALLBACK, thinking=False, think_directive=True)
    once = build_system_prompt(cfg, "consigne")
    assert once.count("/no_think") == 1


# --- native whisper-cli VAD wiring ------------------------------------------

def test_vad_args_empty_when_not_configured():
    cfg = dict(BUILTIN_FALLBACK)
    assert vad_args(cfg) == []


def test_vad_args_empty_when_model_missing(tmp_path):
    """Emitting --vad without a real model would make whisper-cli fail on
    every single chunk, so a dangling path must stay inert."""
    cfg = dict(BUILTIN_FALLBACK,
               vad_model=str(tmp_path / "absent-vad.bin"),
               vad_threshold=0.5)
    assert vad_args(cfg) == []


def test_vad_args_emitted_when_model_present(tmp_path):
    model = tmp_path / "ggml-silero-v5.1.2.bin"
    model.write_bytes(b"IR")
    cfg = dict(BUILTIN_FALLBACK, vad_model=str(model), vad_threshold=0.4)
    args = vad_args(cfg)
    assert "--vad" in args
    assert "--vad-model" in args
    assert args[args.index("--vad-model") + 1] == str(model)
    assert args[args.index("--vad-threshold") + 1] == "0.4"


def test_vad_args_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "vad.bin").write_bytes(b"IR")
    cfg = dict(BUILTIN_FALLBACK, vad_model="~/vad.bin", vad_threshold=0.5)
    assert str(tmp_path / "vad.bin") in vad_args(cfg)


# --- Recorder : the live capture thread -------------------------------------

def test_recorder_keeps_recording_while_the_consumer_is_slow(tmp_path):
    """The fluidity fix. With a sequential loop the microphone idled for the
    whole STT latency between two chunks. Here the fake recorder is much
    faster than the consumer, so several chunks must already be queued and
    recorded by the time the first one is even read."""
    stop = threading.Event()
    made = []

    def fast_record(wav, seconds, target):
        with open(wav, "wb") as f:
            f.write(b"x" * 64)
        made.append(wav)
        return True

    rec = Recorder(str(tmp_path), 0.01, None, stop,
                   record_fn=fast_record, backoff=0, max_failures=3,
                   max_queue=3)
    rec.start()
    time.sleep(0.4)
    # Consumer has read nothing at all.
    assert len(made) >= 2, "la capture doit avancer sans attendre le lecteur"
    stop.set()
    rec.join(timeout=5)
    assert not rec.is_alive()


def test_recorder_stops_and_signals_end(tmp_path):
    stop = threading.Event()

    def fast_record(wav, seconds, target):
        with open(wav, "wb") as f:
            f.write(b"x" * 64)
        return True

    rec = Recorder(str(tmp_path), 0.01, None, stop,
                   record_fn=fast_record, backoff=0, max_queue=2)
    rec.start()
    time.sleep(0.15)
    stop.set()
    rec.join(timeout=5)
    assert not rec.is_alive()
    # The consumer loop terminates on this sentinel.
    assert rec.queue.get(timeout=2)[0] in ("wav", "end")


def test_recorder_never_blocks_the_stop_sentinel(tmp_path):
    """Regression : a bounded queue made the producer block on put(), so the
    "end" sentinel could never be delivered and both threads deadlocked on
    shutdown. The queue is unbounded and the producer counts instead."""
    stop = threading.Event()

    def fast_record(wav, seconds, target):
        with open(wav, "wb") as f:
            f.write(b"x" * 64)
        return True

    rec = Recorder(str(tmp_path), 0.01, None, stop,
                   record_fn=fast_record, backoff=0, max_queue=2)
    rec.start()
    time.sleep(0.3)  # nothing consumed : the bound is saturated
    stop.set()
    rec.join(timeout=5)
    assert not rec.is_alive()
    assert rec.dropped > 0, "la saturation doit avoir abandonne des chunks"
    # The sentinel is reachable even though the queue was never drained.
    seen = []
    while True:
        item = rec.queue.get(timeout=2)
        seen.append(item[0])
        if item[0] == "end":
            break
    assert seen[-1] == "end"


def test_recorder_abandons_after_consecutive_failures(tmp_path):
    """The old loop had no backoff and no cap : a device that disappeared made
    it spin at full speed appending error lines until the disk filled."""
    stop = threading.Event()
    calls = []

    def failing_record(wav, seconds, target):
        calls.append(1)
        return False

    rec = Recorder(str(tmp_path), 0.01, None, stop,
                   record_fn=failing_record, backoff=0, max_failures=3)
    rec.start()
    rec.join(timeout=5)
    assert not rec.is_alive()
    assert rec.abandoned is True
    assert rec.reason
    assert len(calls) == 3
    kinds = []
    while True:
        item = rec.queue.get_nowait()
        kinds.append(item[0])
        if item[0] == "end":
            break
    assert kinds.count("error") == 3


def test_recorder_resets_failure_counter_after_a_success(tmp_path):
    stop = threading.Event()
    state = {"n": 0}

    def flaky(wav, seconds, target):
        state["n"] += 1
        if state["n"] % 2 == 1:
            return False
        with open(wav, "wb") as f:
            f.write(b"x" * 64)
        return True

    rec = Recorder(str(tmp_path), 0.01, None, stop,
                   record_fn=flaky, backoff=0, max_failures=3, max_queue=8)
    rec.start()
    time.sleep(0.3)
    stop.set()
    rec.join(timeout=5)
    assert rec.abandoned is False
