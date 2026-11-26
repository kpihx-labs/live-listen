"""Audio helpers : the energy gate and the whisper repetition collapse.

Both exist because of a measured failure, not a theory :
  * whisper transcribed a silent chunk into "Sous-titrage Societe Radio-Canada"
    repeated 7 times, so silent chunks are now gated by RMS energy.
  * whisper looped on the video title into
    "El Espiritu de la Camara El Espiritu del Espiritu de la Camara ...",
    which then poisoned the summary with invented repeated content.
"""

import math
import os
import struct
import wave

from live_listen.pipeline import collapse_repetition, wav_rms

SAMPLE_RATE = 16000


def write_wav(path, samples, channels=1, width=2):
    """Write raw int samples to a 16-bit PCM wav."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return path


def make_wav(tmpdir, name, samples, **kw):
    return write_wav(os.path.join(tmpdir, name), samples, **kw)


def test_wav_rms_silence_is_zero(tmp_path):
    path = make_wav(str(tmp_path), "silence.wav", [0] * 4000)
    assert wav_rms(path) == 0.0


def test_wav_rms_scales_with_amplitude(tmp_path):
    quiet = make_wav(str(tmp_path), "q.wav", [100] * 4000)
    loud = make_wav(str(tmp_path), "l.wav", [8000] * 4000)
    rq, rl = wav_rms(quiet), wav_rms(loud)
    assert 0 < rq < rl <= 1.0
    # 8000/32768 is the expected normalised RMS for a constant full-ish signal.
    assert abs(rl - 8000 / 32768.0) < 1e-6


def test_shipped_silence_gate_rejects_silence_and_keeps_quiet_speech(tmp_path):
    """Calibrated on the real recorded chunks of 2026-09-23, not guessed.

    The chunk that carried the actual spoken test ("Quand on est a
    l'interieur, on enleve son echarpe") measures RMS 0.0079. A gate anywhere
    near 0.008 throws real speech away, which is the opposite of the intent.
    The shipped default must therefore sit below that measured speech value
    while still rejecting a dead input.
    """
    from live_listen.config import BUILTIN_FALLBACK

    gate = BUILTIN_FALLBACK["silence_rms"]
    measured_quiet_speech = 0.0079
    assert gate < measured_quiet_speech, (
        "le seuil doit rester sous la parole douce mesuree (0.0079), "
        "sinon il rejette de la vraie parole")
    dead = make_wav(str(tmp_path), "dead.wav", [0] * 8000)
    assert wav_rms(dead) < gate
    speech = make_wav(str(tmp_path), "v.wav",
                      [int(3000 * math.sin(i / 20.0)) for i in range(8000)])
    assert wav_rms(speech) > gate


def test_gate_is_not_claimed_to_separate_speech_from_a_jingle(tmp_path):
    """A station stinger is real audio with real energy. The repetition
    collapse handles that case, not the gate : this test pins the measured
    overlap so nobody raises the threshold to try."""
    jingle = make_wav(str(tmp_path), "j.wav",
                      [int(2500 * math.sin(i / 8.0)) for i in range(8000)])
    quiet_speech = make_wav(str(tmp_path), "q.wav",
                            [int(1500 * math.sin(i / 25.0)) for i in range(8000)])
    assert wav_rms(jingle) > wav_rms(quiet_speech)
    # Both are above the gate : energy alone cannot tell them apart.


def test_wav_rms_stereo_takes_one_channel(tmp_path):
    mono = make_wav(str(tmp_path), "m.wav", [4000] * 2000)
    left = [4000] * 1000
    right = [4000] * 1000
    stereo = make_wav(str(tmp_path), "s2.wav", left + right, channels=2)
    assert abs(wav_rms(mono) - wav_rms(stereo)) < 0.2


def test_wav_rms_unreadable_returns_none(tmp_path):
    bogus = os.path.join(str(tmp_path), "not-a-wav.bin")
    with open(bogus, "wb") as f:
        f.write(b"ceci n'est pas un wav")
    assert wav_rms(bogus) is None


def test_wav_rms_missing_file_returns_none(tmp_path):
    assert wav_rms(os.path.join(str(tmp_path), "absent.wav")) is None


def test_wav_rms_empty_wav_returns_zero(tmp_path):
    path = make_wav(str(tmp_path), "empty.wav", [])
    assert wav_rms(path) == 0.0


def test_collapse_repetition_folds_an_exact_loop_entirely():
    """The common case : a perfectly repeating loop is removed completely.
    This is the "Sous-titrage Societe Radio-Canada" shape that whisper emitted
    on the silent 2026-09-23 chunks."""
    raw = ("Sous-titrage Societe Radio-Canada Sous-titrage Societe Radio-Canada "
           "Sous-titrage Societe Radio-Canada")
    assert collapse_repetition(raw) == "Sous-titrage Societe Radio-Canada"


def test_collapse_repetition_shortens_a_drifting_loop():
    """The recorded Spanish title loop drifts ("de la Camara" becomes "del
    Espiritu") so no exact block repeats. The documented guarantee is a
    substantial shortening, not a full collapse : see the function docstring."""
    raw = ("El Espiritu de la Camara El Espiritu del Espiritu de la Camara "
           "El Espiritu del Espiritu del Espiritu de la Camara")
    out = collapse_repetition(raw)
    assert len(out.split()) < len(raw.split())
    assert len(out.split()) <= 0.75 * len(raw.split())


def test_collapse_repetition_kills_word_level_loop():
    raw = ("Je fais un deuxieme test Deuxieme test Deuxieme test Deuxieme test")
    assert collapse_repetition(raw) == "Je fais un deuxieme test Deuxieme test"


def test_collapse_repetition_folds_a_twice_repeated_jingle():
    """"!Suscribete al canal!" twice is a YouTube sting carrying no
    information beyond its own existence : one occurrence is the right
    summary of it, not two."""
    raw = "Suscribete al canal! Suscribete al canal!"
    assert collapse_repetition(raw) == "Suscribete al canal!"


def test_collapse_repetition_keeps_two_distinct_occurrences():
    """A phrase said once, then something else, then said again, is two real
    events. Only consecutive repetition is a loop."""
    raw = "Maria dit bonjour. Jean repond. Maria dit bonjour."
    assert collapse_repetition(raw) == raw


def test_collapse_repetition_leaves_normal_sentence_untouched():
    raw = ("Le personnage entre dans la piece, hesite, puis parle a Maria "
           "qui repond aussitot depuis le couloir.")
    assert collapse_repetition(raw) == raw


def test_collapse_repetition_keeps_repeated_content_across_a_gap():
    """Two mentions separated by other words are information, not a loop."""
    raw = "Maria parle. Jean se tait. Maria reparle."
    assert collapse_repetition(raw) == raw


def test_collapse_repetition_handles_empty_and_short():
    assert collapse_repetition("") == ""
    assert collapse_repetition(None) == ""
    assert collapse_repetition("bonjour") == "bonjour"
