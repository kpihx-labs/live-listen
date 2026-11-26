"""live-listen pipeline : record + transcribe + summarize + think-filter.

pw-record (chunks 16 kHz mono, whisper-optimal) -> whisper-cli
-> transcription.txt -> progressive summary via the local OpenVINO
server (resume.txt, loop in the same process).

Two .txt outputs only. Chunk wavs go to a temp dir (removed on exit).
Stdlib only, zero pip dependencies.
"""

import array
import datetime
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave

from live_listen.config import log

# A missing binary is a setup problem, not a crash : 127/126 are the shell
# conventions for "not found" / "not executable" and let the caller branch on
# the return code like on any other failure.
EXIT_NOT_FOUND = 127
EXIT_NOT_EXECUTABLE = 126
EXIT_TIMEOUT = 124


def run(cmd, timeout=None):
    """Run a command, return (code, stdout, stderr) as text.

    Never raises : a missing binary or a bad path is a return code, so a
    session degrades to a clear error instead of a traceback that kills the
    whole capture loop.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return EXIT_TIMEOUT, out, err
    except FileNotFoundError:
        return EXIT_NOT_FOUND, "", "binaire introuvable : %s" % cmd[0]
    except OSError as e:
        return EXIT_NOT_EXECUTABLE, "", "%s : %s" % (type(e).__name__, e)


def require_binary(name):
    """True if the binary is on PATH, False with a logged reason if not."""
    if shutil.which(name):
        return True
    log("[ERREUR pretraitement] binaire requis absent du PATH : %s" % name)
    return False


def whisper_backends(binary="whisper-cli"):
    """Compiled whisper.cpp backends from the binary's resolved libraries."""
    path = shutil.which(binary)
    if not path:
        return set()
    code, out, _err = run(["ldd", path])
    if code != 0:
        return set()
    return set(re.findall(r"libggml-([a-z0-9_-]+)\.so", out))


def whisper_device(requested="auto"):
    """Resolve auto/xpu/cuda/cpu; auto prioritises the local Intel GPU."""
    backends = whisper_backends()
    pci = run(["lspci"])[1] if shutil.which("lspci") else ""
    has_xpu = "vulkan" in backends and bool(
        re.search(r"(?:VGA|3D|Display).*Intel", pci, re.IGNORECASE)
    )
    has_cuda = "cuda" in backends and shutil.which("nvidia-smi") is not None
    if requested == "auto":
        if has_xpu:
            return "xpu"
        if has_cuda:
            return "cuda"
        if "vulkan" in backends:
            return "gpu"
        return "cpu"
    if requested == "xpu" and not has_xpu:
        raise RuntimeError("XPU demande mais GPU Intel/backend Vulkan indisponible")
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("CUDA demande mais GPU NVIDIA/backend CUDA indisponible")
    if requested not in {"xpu", "cuda", "cpu"}:
        raise ValueError("stt_device doit etre auto, xpu, cuda ou cpu")
    return requested


def whisper_device_args(requested="auto"):
    selected = whisper_device(requested)
    return selected, (["--no-gpu"] if selected == "cpu" else ["--device", "0"])


def wav_rms(path):
    """Normalised RMS energy of a 16-bit PCM wav, in 0..1. None if unreadable.

    Pure stdlib on purpose : `audioop` was removed in Python 3.13 and this
    project must keep working on every interpreter the .python-version pin
    and the stray 3.14 cache suggest it has been run under.
    """
    try:
        with wave.open(path, "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    if width != 2:
        return None
    if not raw:
        return 0.0  # a valid but empty wav is silence, not a read failure
    samples = array.array("h")
    samples.frombytes(raw[:len(raw) - (len(raw) % 2)])
    if channels > 1:
        samples = samples[::channels]
    if not samples:
        return 0.0
    total = 0
    for v in samples:
        total += v * v
    return math.sqrt(total / len(samples)) / 32768.0


def collapse_repetition(text, max_period=12, min_repeats=2):
    """Collapse an immediate repetition loop to a single copy of the block.

    whisper.cpp loops on non-speech and on music, producing things like
    "El Espiritu de la Camara El Espiritu del Espiritu de la Camara ..." which
    then poisons the summary with invented repeated content. A short block
    repeated back to back keeps one copy.

    min_repeats is 2 because the recorded loop only ever repeated twice (and
    drifted while doing so, "de la Camara" becoming "del Espiritu"). Requiring
    3 missed it entirely. Collapsing a twice-repeated jingle down to one
    occurrence is also the right outcome: "!Suscribete al canal!" is a
    YouTube sting, it carries no information beyond its own existence.
    Repetition that is not consecutive is information and is never touched.

    Known limit, stated rather than hidden : this matches blocks EXACTLY, so a
    perfectly repeating loop is removed entirely ("Sous-titrage Societe
    Radio-Canada" x3 : 9 words to 3) while a loop that drifts as it repeats
    is only shortened (the 21-word title loop goes to 14). A fuzzy matcher
    would catch the drifting case too, but it cannot be told apart from real
    speech with enough confidence to be worth the risk of eating words that
    were actually said. The transcript is additionally bounded by
    max_new_chars, so a drifting loop cannot grow without limit.
    """
    text = text or ""
    words = text.split()
    if len(words) < min_repeats:
        return text
    best = None
    for period in range(1, max_period + 1):
        stop = len(words) - period * min_repeats + 1
        for i in range(max(stop, 0)):
            block = words[i:i + period]
            if len(block) < period:
                break
            reps = 1
            j = i + period
            while j + period <= len(words) and words[j:j + period] == block:
                reps += 1
                j += period
            if reps < min_repeats:
                continue
            candidate = words[:i] + block + words[j:]
            if best is None or len(candidate) < len(best):
                best = candidate
    return " ".join(best) if best is not None else text


def elapsed_hms(start):
    s = int(time.time() - start)
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


def stamp_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_source(mode, monitor_target, silence_rms=0.002, probe_seconds=2):
    """Return (target or None, label). None = default source (mic).

    The `auto` probe now requires actual signal energy, not merely a non-empty
    wav : a monitor that exists but carries the wrong (silent) sink used to
    pass the probe and then record nothing for the whole session.
    """
    if mode == "monitor":
        return monitor_target, "monitor film : %s" % monitor_target
    if mode == "mic":
        return None, "micro par defaut (sans --target)"
    fd, probe = tempfile.mkstemp(prefix="live-listen-autoprobe-", suffix=".wav")
    os.close(fd)
    try:
        code, _out, err = run(
            ["timeout", str(probe_seconds), "pw-record", "--target", monitor_target,
             "--rate", "16000", "--channels", "1", probe],
            timeout=probe_seconds + 15)
        exists = os.path.exists(probe) and os.path.getsize(probe) > 44
        rms = wav_rms(probe) if exists else None
        if not exists:
            log("[source] auto : monitor muet ou absent (code=%s stderr=%s), "
                "repli sur micro." % (code, (err.strip() or "(vide)")[-300:]))
            return None, "micro par defaut (sans --target)"
        if silence_rms > 0 and rms is not None and rms < silence_rms:
            log("[source] auto : monitor joignable mais SILENCIEUX "
                "(RMS=%.5f < %.5f), repli sur micro. Cause probable : le son "
                "sort par un autre sink que celui du monitor." % (rms, silence_rms))
            return None, "micro par defaut (sans --target)"
        log("[source] auto : monitor OK (RMS=%.5f), capture du film." % (rms or 0.0))
        return monitor_target, "monitor film : %s" % monitor_target
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def record_chunk(path, seconds, target):
    """Record one chunk. True if non-empty wav, else False (error logged)."""
    cmd = ["timeout", str(seconds), "pw-record"]
    if target:
        cmd += ["--target", target]
    cmd += ["--rate", "16000", "--channels", "1", path]
    code, out, err = run(cmd, timeout=seconds + 15)
    if os.path.exists(path) and os.path.getsize(path) > 44:
        return True
    log("[ERREUR chunk] pw-record a echoue (code=%s) sur %s." % (code, path))
    if out.strip():
        log("[ERREUR chunk] stdout pw-record :\n%s" % out.strip())
    if err.strip():
        log("[ERREUR chunk] stderr pw-record :\n%s" % err.strip())
    else:
        log("[ERREUR chunk] stderr pw-record : (vide)")
    # Single retry via ffmpeg before giving up.
    log("[chunk] retry unique via ffmpeg...")
    fsrc = target if target else "default"
    fcmd = ["ffmpeg", "-y", "-f", "pulse", "-i", fsrc,
            "-ar", "16000", "-ac", "1", "-t", str(seconds), path]
    code2, out2, err2 = run(fcmd, timeout=seconds + 30)
    if os.path.exists(path) and os.path.getsize(path) > 44:
        log("[chunk] retry ffmpeg OK.")
        return True
    log("[ERREUR chunk] ECHEC DEFINITIF %s (code ffmpeg=%s)." % (path, code2))
    log("[ERREUR chunk] stdout ffmpeg :\n%s" % (out2.strip() or "(vide)"))
    log("[ERREUR chunk] stderr ffmpeg :\n%s" % (err2.strip() or "(vide)"))
    return False


TIMESTAMP_RE = re.compile(r"^\s*\[.*?-->.*?\]\s*(.*)\s*$")
LOG_PREFIXES = ("whisper_", "system_info", "main:", "sampling", "encode",
                "decode", "mel:", "model:", "ls ", "ggml", "print_")


def parse_whisper_stdout(stdout):
    """Extract the useful text from whisper-cli output."""
    timed = []
    for ln in (stdout or "").splitlines():
        m = TIMESTAMP_RE.match(ln)
        if m:
            txt = m.group(1).strip()
            if txt and txt != "[Silence?]" and "[BLANK" not in txt:
                timed.append(txt)
    if timed:
        return " ".join(timed).strip()
    kept = []
    for ln in (stdout or "").splitlines():
        s = ln.strip()
        if not s or s.startswith(LOG_PREFIXES):
            continue
        kept.append(s)
    return " ".join(kept).strip()


def vad_args(cfg):
    """whisper-cli native VAD flags, only when the model is really there.

    Emitting `--vad` without `--vad-model` would make whisper-cli fail on every
    chunk, so the capability is wired but only armed when the file resolves.
    """
    model = (cfg.get("vad_model") or "").strip()
    if not model:
        return []
    path = os.path.expanduser(model)
    if not os.path.exists(path):
        return []
    return ["--vad", "--vad-model", path,
            "--vad-threshold", str(cfg.get("vad_threshold", 0.5))]


def transcribe(wav, stt_model, lang, silence_rms=0.0, timeout=600, threads="8",
               device="auto", extra_args=None):
    """Transcribe one chunk. Returns the text, or None on a hard STT error.

    Returns "" (empty, not None) when the chunk is below the energy gate : an
    empty result means "nothing was said", which is a normal outcome, while
    None means "whisper failed". The caller keeps the two apart.
    """
    if silence_rms > 0:
        rms = wav_rms(wav)
        if rms is not None and rms < silence_rms:
            log("[stt] chunk ignore (RMS=%.5f < %.5f, silence ou bruit de fond)"
                % (rms, silence_rms))
            return ""
    try:
        selected, device_args = whisper_device_args(device)
    except (RuntimeError, ValueError) as exc:
        log("[ERREUR stt] %s" % exc)
        return None
    cmd = ["whisper-cli", "-m", stt_model, "-f", wav, "-l", lang, "-t", str(threads)]
    cmd += device_args
    cmd += list(extra_args or [])
    log("[stt] device=%s (demande=%s)" % (selected, device))
    code, out, err = run(cmd, timeout=timeout)
    if code == EXIT_NOT_FOUND:
        log("[ERREUR stt] whisper-cli absent du PATH : transcription impossible.")
        return None
    if code != 0:
        log("[ERREUR stt] whisper-cli code=%s sur %s" % (code, wav))
        log("[ERREUR stt] stdout :\n%s" % (out.strip() or "(vide)"))
        log("[ERREUR stt] stderr :\n%s" % (err.strip() or "(vide)"))
        return None
    return collapse_repetition(parse_whisper_stdout(out))


THINK_TAG_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<(think|thinking|reasoning)>", re.IGNORECASE)


def strip_thinking(content):
    """Remove model reasoning before writing the summary.

    qwen3 does not always honor enable_thinking=false : its reasoning
    (English, verbose) can arrive TAGGED or not and eat the whole
    max_tokens budget, leaving no room for the answer. Without this
    filter, resume.txt receives reasoning instead of the summary.
    Returns "" if nothing usable remains (the caller then raises and
    the old summary is kept for the next cycle).
    """
    text = content or ""
    text = THINK_TAG_RE.sub("", text)
    # Tag opened but never closed : everything after is reasoning.
    m = THINK_OPEN_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


# Reasoning detection.
#
# A flat ">= 2 substrings" threshold was too fragile : one of the two real
# leaks recorded on 2026-09-23 (sessions/20260923-203051-fr/resume.txt)
# matched exactly 2 markers, so a single rewording of the model would have
# let it through. Markers are now weighted, and the score is reinforced by
# three structural anchors that do not depend on the exact wording the model
# happens to pick.
#
# Strong markers are phrases a summary essentially never contains. Medium
# ones are common enough that they must combine. The threshold of 3 means a
# real French summary needs either three medium markers or one strong plus
# one medium before it is even suspect.

REASONING_THRESHOLD = 3

STRONG_MARKERS = (
    "as an ai",
    "let's start by",
    "the user wants",
    "the user provided",
    "the user mentioned",
    "i need to think",
    "the previous summary was",
    "the new transcript",
)

MEDIUM_MARKERS = (
    "i need to",
    "i should also",
    "i'll begin",
    "i'll just",
    "i'll start",
    "i'll check",
    "let me structure",
    "let me break",
    "next, i'll",
    "step by step",
    "that should cover",
    "first, the previous",
    "okay, let's",
    "keep track of what's",
)

# (label, pattern, weight) : each counted at most once. Weight 3 means the
# anchor is conclusive on its own, which is why the threshold is 3 : a legacy
# English summary can plausibly contain one medium marker, and two of them,
# but no summary ever opens with "Okay, let's", says "I'll just add", or talks
# about writing "the summary" in the future tense.
REASONING_ANCHORS = (
    ("opener+let's", re.compile(r"^\s*(okay|ok|alright|and)\b[^.!?\n]{0,24}\blet'?s\b",
                                re.IGNORECASE), 3),
    ("i'll <verb>", re.compile(
        r"\bi'?ll\s+(begin|start|just\s+add|structure|check|note|review|add)\b",
        re.IGNORECASE), 3),
    ("future about summary", re.compile(
        r"\b(i\s+will|i'?ll|we'?ll)\s+(add|write|keep|maintain|preserve)\b"
        r"[^.!?\n]{0,40}\b(summary|transcription|resume)\b",
        re.IGNORECASE), 3),
)

BLUNT_RETRY_LINE = ("IGNORE tout raisonnement. "
                    "Reponds UNIQUEMENT par le resume final.")


def reasoning_score(text):
    """(score, [labels]) for a candidate summary. Higher means worse."""
    lowered = (text or "").lower()
    score = 0
    labels = []
    for m in STRONG_MARKERS:
        if m in lowered:
            score += 2
            labels.append("fort:%s" % m)
    for m in MEDIUM_MARKERS:
        if m in lowered:
            score += 1
            labels.append("moyen:%s" % m)
    for label, pattern, weight in REASONING_ANCHORS:
        if pattern.search(text or ""):
            score += weight
            labels.append("ancre:%s" % label)
    return score, labels


def looks_like_reasoning(text):
    """True when the stripped content still smells like model reasoning."""
    return reasoning_score(text)[0] >= REASONING_THRESHOLD


def matched_markers(text):
    """Human readable list of what fired, for the console log."""
    return reasoning_score(text)[1]


def post_summary(cfg, system, user_text, first_call):
    """One HTTP round to the OpenVINO server. Returns the raw content string.

    `enable_thinking` is always sent : the server decides from the chat
    template text whether the model family actually consumes it. The sampling
    cut (top_p / top_k) is always sent too, because OpenVINO's own defaults
    are no cut at all and a non-zero temperature then samples the entire
    vocabulary into multilingual noise.
    """
    model = cfg["summary_model"]
    if first_call:
        log("[resume] premier appel : chargement eventuel du modele '%s' "
            "sur le serveur (25-60 s possibles), patience..." % model)
    url = cfg["server"].rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "enable_thinking": bool(cfg["thinking"]),
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=cfg["http_timeout"]) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # 429 and 502 are not "the model refused" : measured on kpihx-ubuntu,
        # they mean the OpenVINO worker subprocess died. The server log showed
        # `CL_OUT_OF_RESOURCES` from clFinish, i.e. the Arc iGPU ran out of
        # OpenCL resources when a second model was loaded on top of the first,
        # and the worker was terminated by terminate(). Name it, or the user
        # only sees a generic HTTP error and blames live-listen.
        if e.code in (429, 502, 503):
            raise RuntimeError(
                "serveur OpenVINO en erreur transitoire (HTTP %d) sur '%s'. "
                "Cause probable mesuree : le worker GPU a crash, le GPU Arc "
                "n'a plus assez de ressources OpenCL (CL_OUT_OF_RESOURCES) "
                "lors du chargement d'un second modele. Verifier avec "
                "`openvino ps` (un modele doit etre charge) et "
                "`journalctl --user -u openvino.service -n 40`."
                % (e.code, model))
        raise RuntimeError("HTTP %d %s -> %s : %s"
                           % (e.code, e.reason, url, e.read()[:300].decode(
                               "utf-8", "replace")))
    except Exception as e:
        raise RuntimeError("HTTP POST %s -> %s : %s"
                           % (url, type(e).__name__, e))
    try:
        payload = json.loads(raw)
        message = payload["choices"][0]["message"]
        content = message.get("content")
        reasoning = message.get("reasoning_content")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError("reponse inattendue du serveur (%s : %s) : %s"
                           % (type(e).__name__, e, raw[:1000]))
    if not (content or "").strip():
        # A model that answers only in the reasoning channel leaves nothing to
        # write : name it instead of writing an empty file.
        if (reasoning or "").strip():
            raise RuntimeError(
                "le serveur n'a renvoye que reasoning_content (%d caracteres), "
                "content vide. Modele unsuitable ou thinking force."
                % len(reasoning))
        raise RuntimeError("le serveur a renvoye un contenu vide : %s" % raw[:500])
    return content.strip()


def build_system_prompt(cfg, base_prompt):
    """The prompt actually sent, with the no-think directive when it applies.

    `enable_thinking=false` is not authoritative on the Qwen3 checkpoint
    served here : the model still self-emits a <think> block despite the empty
    pre-filled block (measured, recorded in the k-openvino skill). "/no_think"
    prepended to the system prompt is the template-level directive that
    survives that, and it is the single cheapest fix for both the leak and
    the 1x-3x generation-time penalty thinking costs on a live session.
    """
    if cfg["think_directive"] and not cfg["thinking"]:
        return "/no_think\n" + base_prompt
    return base_prompt


def summarize_once(cfg, system, user_text, first_call):
    """One summary round : POST, strip tags, score reasoning, retry once.

    `system` must already be the built prompt (see build_system_prompt) : the
    blunter retry only appends its line, it never rebuilds, so the /no_think
    directive is not stacked twice.

    Keeps enable_thinking=false in the POST body, strips reasoning tags
    (closed + unclosed), then scores the result with the weighted detector.
    On detection, ONE automatic retry fires with a blunter system line. If
    still polluted, raises (old summary kept, retry next cycle).
    """
    content = post_summary(cfg, system, user_text, first_call)
    content = strip_thinking(content)
    if not content:
        raise RuntimeError(
            "le serveur n'a renvoye que du raisonnement "
            "(vide apres suppression des balises)")
    if looks_like_reasoning(content):
        found = matched_markers(content)
        log("[resume] RAISONNEMENT DETECTE (score=%d : %s), "
            "nouvel essai avec consigne directe..."
            % (reasoning_score(content)[0], ", ".join(found)))
        blunt = system + "\n" + BLUNT_RETRY_LINE
        content = post_summary(cfg, blunt, user_text, False)
        content = strip_thinking(content)
        if not content:
            raise RuntimeError(
                "retry anti-raisonnement vide "
                "(vide apres suppression des balises)")
        if looks_like_reasoning(content):
            found = matched_markers(content)
            log("[resume] RAISONNEMENT TOUJOURS PRESENT APRES RETRY "
                "(score=%d : %s), cycle abandonne."
                % (reasoning_score(content)[0], ", ".join(found)))
            raise RuntimeError(
                "resume pollue par du raisonnement apres retry "
                "(score=%d)" % reasoning_score(content)[0])
        log("[resume] retry OK, raisonnement elimine.")
    return content


def default_sink_name(wpctl_output):
    """Parse `wpctl status` for the default Audio/Sink node name."""
    for line in (wpctl_output or "").splitlines():
        m = re.search(r"Audio/Sink\s+(\S+)", line)
        if m:
            return m.group(1)
    return None


def warn_sink_mismatch(cfg):
    """Warn when the monitor target does not belong to the default sink.

    Measured on kpihx-ubuntu : the default sink was a Bluetooth speaker while
    the configured monitor was the internal Intel speaker. The session records
    in perfect silence for its whole duration and only the STT hallucination
    hints at it. Purely a warning : an explicit --monitor-target may be exactly
    what the user wants, so the decision stays with them.
    """
    code, out, _err = run(["wpctl", "status"], timeout=15)
    if code != 0:
        log("[AVERTISSEMENT] wpctl status indisponible (code=%s) : "
            "coherence sink non verifiee." % code)
        return
    sink = default_sink_name(out)
    if not sink:
        log("[AVERTISSEMENT] sink par defaut introuvable dans wpctl status.")
        return
    target = cfg["monitor_target"] or ""
    if not target.endswith(".monitor"):
        return
    if not target.startswith(sink):
        log("[AVERTISSEMENT COHERENCE SON] sink par defaut = %s" % sink)
        log("[AVERTISSEMENT COHERENCE SON] monitor_target = %s" % target)
        log("[AVERTISSEMENT COHERENCE SON] le monitor N'APPARTIENT PAS au sink "
            "par defaut : la capture sera muette. Passer par --source mic, ou "
            "--monitor-target %s.monitor, ou brancher les/ecouter les "
            "haut-parleurs internes." % sink)


def server_models(cfg):
    """List the model ids the OpenVINO server advertises, or [] on failure."""
    try:
        with urllib.request.urlopen(cfg["server"].rstrip("/") + "/v1/models",
                                    timeout=15) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return [m.get("id") for m in payload.get("data", []) if m.get("id")]
    except Exception as e:
        log("[ERREUR] serveur injoignable : %s : %s" % (type(e).__name__, e))
        return []


def preflight(cfg):
    """Everything that can be checked before recording the first chunk.

    Returns True when the session may start. A session that only discovers a
    missing binary, an unserved model or a silent monitor after three minutes
    of recording has wasted the recording and confused the user.
    """
    log("--- pretraitement ---")
    ok = True
    for binary in ("pw-record", "whisper-cli"):
        if not require_binary(binary):
            ok = False
    if not os.path.exists(cfg["stt_model"]):
        log("[ERREUR pretraitement] modele whisper introuvable : %s"
            % cfg["stt_model"])
        ok = False
    else:
        log("[pretraitement] modele whisper : %s" % cfg["stt_model"])
    try:
        selected = whisper_device(cfg["stt_device"])
        log("[pretraitement] device whisper : %s (demande=%s)"
            % (selected, cfg["stt_device"]))
    except (RuntimeError, ValueError) as exc:
        log("[ERREUR pretraitement] %s" % exc)
        ok = False
    if cfg["source"] in ("monitor", "auto"):
        warn_sink_mismatch(cfg)
    ids = server_models(cfg)
    if not ids:
        log("[ERREUR pretraitement] serveur OpenVINO injoignable sur %s : "
            "verifier avec `openvino server status` / `openvino ps`."
            % cfg["server"])
        ok = False
    else:
        log("[pretraitement] serveur : %s modeles : %s"
            % (cfg["server"], ", ".join(ids)))
        if cfg["summary_model"] not in ids:
            log("[ERREUR pretraitement] summary_model '%s' n'est PAS servi. "
                "Modeles disponibles : %s" % (cfg["summary_model"], ", ".join(ids)))
            ok = False
    if cfg["think_directive"] and not cfg["thinking"]:
        log("[pretraitement] /no_think actif sur '%s'" % cfg["summary_model"])
    if cfg["silence_rms"] > 0:
        log("[pretraitement] porte de silence RMS=%.5f" % cfg["silence_rms"])
    else:
        log("[pretraitement] porte de silence desactivee")
    if vad_args(cfg):
        log("[pretraitement] VAD natif whisper-cli actif (modele %s, seuil %s)"
            % (cfg["vad_model"], cfg["vad_threshold"]))
    elif (cfg.get("vad_model") or "").strip():
        log("[AVERTISSEMENT] vad_model configure mais introuvable : %s "
            "(VAD desactive, repli sur la porte d'energie)"
            % cfg["vad_model"])
    else:
        log("[pretraitement] VAD natif whisper-cli inactif (pas de "
            "vad_model). Porte d'energie seule.")
    if ok:
        log("[pretraitement] OK")
    else:
        log("[ERREUR pretraitement] echec : corriger ci-dessus ou passer "
            "--no-preflight pour lancer quand meme.")
    return ok


def write_atomic(path, text):
    """Write a text file atomically.

    The summary file used to be written with three successive f.write calls,
    so a Ctrl-C landing in the middle (or the worker being abandoned when the
    join deadline expires) left a truncated resume.txt. Write to a sibling
    temp file, fsync, then rename.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".live-listen-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def do_list_sources(cfg):
    log("=== live-listen --list-sources ===")
    code, out, err = run(["wpctl", "status"], timeout=15)
    log("--- wpctl status (code=%s) ---" % code)
    log(out.strip() or "(vide)")
    if err.strip():
        log("[wpctl stderr] %s" % err.strip())
    log("--- cible monitor configuree ---")
    log(cfg["monitor_target"])
    warn_sink_mismatch(cfg)
    fd, probe = tempfile.mkstemp(prefix="live-listen-listprobe-", suffix=".wav")
    os.close(fd)
    try:
        code, _o, err = run(
            ["timeout", "2", "pw-record", "--target", cfg["monitor_target"],
             "--rate", "16000", "--channels", "1", probe], timeout=15)
        if os.path.exists(probe) and os.path.getsize(probe) > 44:
            rms = wav_rms(probe)
            log("[sonde] monitor joignable : %d octets, RMS=%s (code=%s ; "
                "code=124 est ATTENDU, la sonde coupe deliberately a 2 s)"
                % (os.path.getsize(probe),
                   ("%.5f" % rms) if rms is not None else "illisible", code))
            if cfg["silence_rms"] > 0 and rms is not None and rms < cfg["silence_rms"]:
                log("[sonde] VERDICT : SILENCE. Rien ne sera transcrit. "
                    "Cause probable : sink par defaut != sink du monitor_target.")
        else:
            log("[ERREUR sonde] monitor muet ou absent : pw-record code=%s "
                "stderr=%s" % (code, err.strip() or "(vide)"))
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
    log("--- micro par defaut ---")
    code, out, _e = run(["wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@"], timeout=10)
    log(out.strip().splitlines()[0] if code == 0 and out.strip() else
        "(inspect @DEFAULT_AUDIO_SOURCE@ impossible, code=%s)" % code)
    log("--- modeles whisper locaux ---")
    wdir = os.path.join(
        os.path.expanduser("~"), ".local", "share", "whisper", "models", "ggml"
    )
    if os.path.isdir(wdir):
        for f in sorted(os.listdir(wdir)):
            p = os.path.join(wdir, f)
            if os.path.isfile(p):
                log("  %s (%d octets)" % (f, os.path.getsize(p)))
    else:
        log("  repertoire absent : %s" % wdir)
    log("[stt] modele configure : %s" % cfg["stt_model"])
    log("--- serveur resume : %s ---" % cfg["server"])
    ids = server_models(cfg)
    for m in ids:
        log("  modele : %s" % m)
    if ids and cfg["summary_model"] not in ids:
        log("[AVERTISSEMENT] summary_model '%s' n'est PAS servi par %s"
            % (cfg["summary_model"], cfg["server"]))


def bound_new_segments(text, max_chars):
    """Keep the tail of the new transcript, never more than max_chars.

    Without a bound, a server that stays unreachable makes the prompt grow
    without limit on every cycle (quadratic tokens, then a hard context
    error). The tail is kept because a live summary cares about what was just
    said.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[-max_chars:], True


def summary_worker_loop(cfg, system, transcript_txt, summary_txt, stop_event, state):
    """Summary loop (same process) : every summary_every s, if the
    transcript grew, POST then rewrite the .txt summary file atomically.
    Errors shown in full, old summary kept, next cycle backs off."""
    last = time.time()
    consecutive_failures = 0
    while not stop_event.is_set():
        if stop_event.wait(5):
            break
        # A server that keeps failing must not be hammered every cycle.
        effective_every = cfg["summary_every"] * min(
            1 + consecutive_failures, 6)
        if (time.time() - last) < effective_every:
            continue
        last = time.time()
        try:
            with open(transcript_txt, encoding="utf-8") as f:
                full = f.read()
        except OSError as e:
            log("[ERREUR resume] transcript illisible : %s" % e)
            continue
        if len(full) <= state["summarized_len"]:
            continue  # nothing new, wait for next cycle
        new_segments, truncated = bound_new_segments(
            full[state["summarized_len"]:], cfg["max_new_chars"])
        if truncated:
            log("[resume] transcription nouvelle tronquee a %d caracteres "
                "(les plus anciens sont laisses de cote)."
                % cfg["max_new_chars"])
        prev = state["prev"]
        num = state["count"] + 1
        state["in_progress"] = True
        try:
            user_text = ("RESUME PRECEDENT :\n%s\n\nNOUVELLE TRANSCRIPTION :\n%s"
                         % (prev if prev else "(aucun)", new_segments))
            content = summarize_once(cfg, system, user_text, state["first"])
            state["first"] = False
            state["prev"] = content
            state["count"] = num
            state["summarized_len"] = len(full)
            write_atomic(summary_txt,
                         "# Resume progressif live-listen\n\n"
                         "Mise a jour #%d - %s (modele %s)\n\n%s\n"
                         % (num, stamp_now(), cfg["summary_model"], content))
            log("=== RESUME #%d ===\n%s" % (num, content))
            consecutive_failures = 0
        except Exception as e:
            state["first"] = False
            consecutive_failures += 1
            log("[ERREUR resume] %s : %s" % (type(e).__name__, e))
            log("[resume] ancien resume conserve, nouvel essai au prochain cycle "
                "(cadence x%d)." % min(1 + consecutive_failures, 6))
        finally:
            state["in_progress"] = False


class Recorder(threading.Thread):
    """Records chunks on its own thread and hands them to the consumer queue.

    The capture used to run strictly between two transcriptions, so the
    effective cadence was chunk_seconds + stt_latency : a slow model silently
    paused the microphone and dropped audio without ever saying so. Recording
    on its own thread keeps the microphone busy while the previous chunk is
    transcribed, which is what makes the session live rather than batched.

    The queue itself is unbounded (see __init__ for why) ; the number of chunks
    waiting for STT is capped by this class, which drops the surplus.
    """

    def __init__(self, chunks_dir, seconds, target, stop_event,
                 record_fn=record_chunk, backoff=5, max_failures=10,
                 max_queue=3):
        super().__init__(daemon=True, name="live-listen-recorder")
        self.chunks_dir = chunks_dir
        self.seconds = seconds
        self.target = target
        self.stop_event = stop_event
        self.record_fn = record_fn
        self.backoff = backoff
        self.max_failures = max_failures
        # Unbounded on purpose. A bounded queue blocks the producer on put(),
        # which deadlocks the stop path : the "end" sentinel could not be
        # delivered and the consumer would wait forever. The bound is enforced
        # by the producer counting pending items itself, via release().
        self.queue = queue.Queue()
        self.max_pending = max_queue
        self._pending = 0
        self._lock = threading.Lock()
        self.recorded = 0
        self.dropped = 0
        self.failures = 0
        self.abandoned = False
        self.reason = ""

    def release(self):
        """Called by the consumer once it has taken an item off the queue."""
        with self._lock:
            if self._pending > 0:
                self._pending -= 1

    def run(self):
        idx = 0
        try:
            while not self.stop_event.is_set():
                idx += 1
                wav = os.path.join(self.chunks_dir, "chunk_%03d.wav" % idx)
                if not self.record_fn(wav, self.seconds, self.target):
                    self.failures += 1
                    self.queue.put(("error", idx))
                    if self.failures >= self.max_failures:
                        self.abandoned = True
                        self.reason = ("%d echecs d'enregistrement consecutifs"
                                       % self.failures)
                        log("[ABANDON] %s : capture arretee." % self.reason)
                        break
                    self.stop_event.wait(self.backoff)
                    continue
                self.failures = 0
                self.recorded = idx
                with self._lock:
                    over = self._pending >= self.max_pending
                    if not over:
                        self._pending += 1
                if over:
                    # STT is slower than realtime. Dropping the chunk we just
                    # recorded keeps the capture moving; blocking here would
                    # lose audio instead.
                    self.dropped += 1
                    log("[ALERTE FLUIDITE] transcription en retard, "
                        "chunk_%03d abandonne (%d au total)."
                        % (idx, self.dropped))
                    try:
                        os.remove(wav)
                    except OSError:
                        pass
                else:
                    self.queue.put(("wav", idx, wav))
        finally:
            self.queue.put(("end",))


def run_session(cfg, system, transcript_txt, summary_txt):
    """Main listening session : record chunks, transcribe, summarize."""
    log("=== live-listen ===")
    log("transcription : %s" % transcript_txt)
    log("resume        : %s" % summary_txt)
    log("chunk=%ss resume toutes les %ss modele=%s thinking=%s think_directive=%s "
        "lang=%s" % (cfg["chunk"], cfg["summary_every"], cfg["summary_model"],
                     cfg["thinking"], cfg["think_directive"], cfg["lang"]))
    log("top_p=%s top_k=%s temperature=%s silence_rms=%s overlap=%s"
        % (cfg["top_p"], cfg["top_k"], cfg["temperature"],
           cfg["silence_rms"], cfg["overlap"]))
    if cfg["preflight"] and not preflight(cfg):
        log("[ERREUR] pretraitement en echec, session non demarree. "
            "(--no-preflight pour forcer.)")
        return 1

    target, label = resolve_source(cfg["source"], cfg["monitor_target"],
                                   cfg["silence_rms"])
    log("source  : %s" % label)
    if cfg["source"] == "monitor" and target:
        log("[source] rappel : si la capture est muette, verifier la coherence "
            "sink/monitor ci-dessus (Bluetooth vs haut-parleurs internes).")
    log("Ctrl-C pour arreter proprement.")

    chunks_dir = tempfile.mkdtemp(prefix="live-listen-")
    log("[chunks] temporaires : %s (supprimes a la fin)" % chunks_dir)
    start = time.time()
    stop_event = threading.Event()
    state = {"prev": "", "count": 0, "summarized_len": 0,
             "first": True, "in_progress": False}
    worker = threading.Thread(
        target=summary_worker_loop,
        args=(cfg, system, transcript_txt, summary_txt, stop_event, state),
        daemon=True, name="live-listen-summary")
    worker.start()

    def append_line(line):
        with open(transcript_txt, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def handle(idx, wav):
        """Transcribe one recorded chunk and append it to the transcript."""
        text = transcribe(wav, cfg["stt_model"], cfg["lang"],
                          cfg["silence_rms"], cfg["stt_timeout"],
                          device=cfg["stt_device"],
                          extra_args=vad_args(cfg))
        try:
            os.remove(wav)
        except OSError:
            pass
        hms = elapsed_hms(start)
        if text is None:
            # Was invisible in the transcript : a whisper failure looked
            # exactly like a silent chunk.
            append_line("[%s] [ERREUR stt chunk_%03d, voir console]\n"
                        % (hms, idx))
            return
        if not text.strip():
            log("[%s] (chunk_%03d : pas de parole)" % (hms, idx))
            return
        log("[%s] %s" % (hms, text))
        append_line("[%s] %s\n" % (hms, text))

    recorder = None
    try:
        if cfg["overlap"]:
            recorder = Recorder(chunks_dir, cfg["chunk"], target, stop_event,
                                backoff=cfg["record_backoff"],
                                max_failures=cfg["record_max_failures"])
            recorder.start()
            log("[enregistrement] thread dedie actif : la capture ne "
                "s'interrompt plus pendant la transcription.")
            while True:
                item = recorder.queue.get()
                recorder.release()
                kind = item[0]
                if kind == "end":
                    break
                if kind == "error":
                    append_line("[%s] [ERREUR enregistrement chunk_%03d, "
                                "voir console]\n" % (elapsed_hms(start), item[1]))
                    if recorder.abandoned:
                        break
                    continue
                handle(item[1], item[2])
        else:
            log("[enregistrement] mode sequentiel (overlap=false).")
            idx = 0
            failures = 0
            while True:
                idx += 1
                wav = os.path.join(chunks_dir, "chunk_%03d.wav" % idx)
                if not record_chunk(wav, cfg["chunk"], target):
                    failures += 1
                    append_line("[%s] [ERREUR enregistrement chunk_%03d, "
                                "voir console]\n" % (elapsed_hms(start), idx))
                    if failures >= cfg["record_max_failures"]:
                        log("[ABANDON] %d echecs d'enregistrement consecutifs : "
                            "session arretee." % failures)
                        break
                    time.sleep(cfg["record_backoff"])
                    continue
                failures = 0
                handle(idx, wav)
    except KeyboardInterrupt:
        log("\nArret demande (Ctrl-C). Fichiers conserves :")
        log("  transcription : %s" % transcript_txt)
        log("  resume        : %s" % summary_txt)
    finally:
        stop_event.set()
        if recorder is not None:
            # The recorder may be mid-record_chunk (chunk seconds + margin),
            # so give it more than a chunk before considering it stuck.
            recorder.join(timeout=cfg["chunk"] + 20)
            if recorder.is_alive():
                log("[arret] enregistreur encore actif apres %ds, abandon. "
                    "Un wav temporaire peut subsister dans %s."
                    % (cfg["chunk"] + 20, chunks_dir))
            if recorder.dropped:
                log("[bilan] %d chunk(s) abandonne(s) car la transcription ne "
                    "suivait pas." % recorder.dropped)
        if state["in_progress"]:
            log("[arret] resume en cours, attente jusqu'a %ds..."
                % (cfg["http_timeout"] + 30))
        # One POST plus margin. A retry in flight is abandoned, which is now
        # safe because the summary file is written atomically.
        worker.join(timeout=cfg["http_timeout"] + 30)
        if worker.is_alive():
            log("[arret] resume abandonne (delai depasse), fichier resume "
                "conserve dans son etat precedent.")
        shutil.rmtree(chunks_dir, ignore_errors=True)
    log("Session terminee.")
    return 0
