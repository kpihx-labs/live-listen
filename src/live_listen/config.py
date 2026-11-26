"""live-listen config : defaults + XDG ensure + precedence + path resolution.

Default config file = ~/.local/share/live-listen/config.json (XDG data dir,
never the project dir). Creation happens two ways : `live-listen init`
writes the defaults file, and ensure-on-startup creates it on first launch
if missing.

Precedence : CLI flags > --config FILE > XDG config file > built-in fallback.
"""

import json
import os
import sys
import threading

APP_DIR_NAME = "live-listen"
CONFIG_FILE_NAME = "config.json"


def xdg_data_dir():
    """XDG data home, ~/.local/share by default."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share")
    return base


def default_config_path():
    return os.path.join(xdg_data_dir(), APP_DIR_NAME, CONFIG_FILE_NAME)


# Built-in fallback, used ONLY if no config file exists yet (it is then
# written to the XDG path so the next launch reads a real file).
BUILTIN_FALLBACK = {
    "lang": "es",
    "prompt": (
        "Tu maintains UN resume progressif unique de la seance a partir de "
        "la transcription fournie (dialogues, cours ou conference). A chaque "
        "mise a jour : conserve l'acquis, ajoute les informations nouvelles, "
        "et corrige si la suite contredit une hypothese anterieure. Structure "
        "le resume en rubriques courtes : deroule et intrigue, personnages ou "
        "intervenants, vocabulaire notable, tournures idiomatiques, references "
        "culturelles. Reponds en francais sauf indication contraire, avec les "
        "citations en langue d'origine entre guillemets."
    ),
    "prompt_file": None,
    "stt_model": os.path.join(
        os.path.expanduser("~"), ".local", "share", "whisper",
        "models", "ggml", "ggml-large-v3-turbo-q5_0.bin"),
    "stt_device": "auto",
    "chunk": 15,
    "summary_every": 90,
    "summary_model": "qwen3-1.7b",
    "thinking": False,
    "source": "monitor",
    "monitor_target": ("alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic"
                       ".HiFi__Speaker__sink.monitor"),
    "server": "http://127.0.0.1:11437",
    "max_tokens": 2048,
    "temperature": 0.0,
    # Sampling cut. OpenVINO's GenerationConfig ships top_p=1.0 and
    # top_k=2**64-1, which is no cut at all : with a non-zero temperature a
    # Gemma-class model then samples the whole vocabulary and answers in
    # random multilingual noise. Measured in k-openvino : 2m20s of noise at
    # temperature 0.7, 1.78s of coherent answer with this cut.
    "top_p": 0.95,
    "top_k": 64,
    # Qwen3 self-emits a <think> block even when enable_thinking=false is sent
    # (measured, documented in k-openvino). "/no_think" prepended to the system
    # prompt is the template-level directive that survives that. Set false for a
    # model family that does not know the token (phi, gemma, llama).
    "think_directive": True,
    # Energy gate before STT. Recalibrated on 2026-09-30 after a live lecture
    # session proved the previous 0.002 unusable : a professor speaking in a
    # room measured 0.00138 to 0.00199 per chunk, so a 0.002 gate rejected
    # essentially the whole talk and let through only the chunks that happened
    # to land a hair above the line (one 15 s chunk yielded the single word
    # "Gracias"). 0.0005 keeps a 2.75x margin below the quietest measured
    # speech, and still sits under a genuinely dead input (0.00066 to 0.00088
    # measured on a silent room mic). Two corpora, both ends, both bracketed.
    #
    # What it deliberately does NOT do : separate speech from a loud station
    # jingle, and it cannot tell a quiet distant speaker from room tone. For
    # real speech/non-speech discrimination the native whisper-cli Silero VAD
    # is the right tool (see vad_model below), not this constant.
    "silence_rms": 0.0005,
    # whisper-cli has a NATIVE neural VAD (Silero, `--vad`). It is a much
    # better speech/non-speech separator than any energy gate. It needs a GGML
    # VAD model in whisper.cpp format, which is NOT installed here, so the
    # default is null : the flags are only emitted when the file is really
    # there, and dropping the model in activates it with no code change.
    "vad_model": None,
    "vad_threshold": 0.5,
    # A summary in flight is abandoned rather than hanging the session, and an
    # STT call cannot stall the loop for 10 minutes unnoticed.
    "http_timeout": 180,
    "stt_timeout": 600,
    # Record the next chunk while the current one is being transcribed, so the
    # capture never pauses for the STT latency. This is what makes it live.
    "overlap": True,
    # Recording failure must not become a tight infinite loop filling the disk.
    "record_backoff": 5,
    "record_max_failures": 10,
    # Bound the new transcript sent per summary round. Without it, a server
    # that stays down makes the prompt grow without limit (quadratic tokens).
    "max_new_chars": 4000,
    # Verify binaries, the STT model, the server and the sink coherence before
    # recording the first chunk.
    "preflight": True,
    "transcript_path": None,
    "summary_path": None,
}

CONFIG_KEYS = tuple(BUILTIN_FALLBACK.keys())


# The summary worker and the recorder run on their own threads : without this
# lock a multi-line error can be split by a line coming from the other thread.
_LOG_LOCK = threading.Lock()


def log(msg):
    with _LOG_LOCK:
        print(msg, flush=True)


def load_json_file(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log("[ERREUR config] fichier %s introuvable : %s" % (what, path))
        sys.exit(2)
    except json.JSONDecodeError as e:
        log("[ERREUR config] %s invalide (%s) : %s" % (what, path, e))
        sys.exit(2)
    if not isinstance(data, dict):
        log("[ERREUR config] %s (%s) n'est pas un objet JSON." % (what, path))
        sys.exit(2)
    return data


def write_defaults_file(path=None):
    """Write the defaults config file. Returns the path written."""
    target = path or default_config_path()
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(BUILTIN_FALLBACK, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return target


def init_config():
    """`live-listen init` : create the XDG config if missing, print path."""
    target = default_config_path()
    if os.path.exists(target):
        log("[config] existe deja, conserve : %s" % target)
    else:
        write_defaults_file(target)
        log("[config] cree : %s" % target)
    print(target, flush=True)
    return target


def ensure_config():
    """Ensure-on-startup : create the XDG config with defaults if missing."""
    target = default_config_path()
    if os.path.exists(target):
        return target, False
    write_defaults_file(target)
    log("[config] cree : %s" % target)
    return target, True


def effective_config(args):
    """Merge builtin fallback < XDG config < --config FILE < CLI flags."""
    cfg = dict(BUILTIN_FALLBACK)
    xdg_path, _created = ensure_config()
    for k, v in load_json_file(xdg_path, "config XDG").items():
        if k in CONFIG_KEYS:
            cfg[k] = v
        else:
            log("[config] cle inconnue ignoree dans %s : %s" % (xdg_path, k))
    if args.config:
        for k, v in load_json_file(args.config, "--config").items():
            if k in CONFIG_KEYS:
                cfg[k] = v
            else:
                log("[config] cle inconnue ignoree dans %s : %s" % (args.config, k))
    for k in CONFIG_KEYS:
        v = getattr(args, k.replace("-", "_"), None)
        if v is None:
            v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    # Loud validations (never a silent default).
    if cfg["source"] not in ("monitor", "mic", "auto"):
        log("[ERREUR config] source invalide : %r (attendu monitor|mic|auto)."
            % cfg["source"])
        sys.exit(2)
    if cfg["stt_device"] not in ("auto", "xpu", "cuda", "cpu"):
        log("[ERREUR config] stt_device invalide : %r "
            "(attendu auto|xpu|cuda|cpu)." % cfg["stt_device"])
        sys.exit(2)
    for k in ("chunk", "summary_every", "max_tokens"):
        if not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] <= 0:
            log("[ERREUR config] %s doit etre un entier > 0, recu : %r."
                % (k, cfg[k]))
            sys.exit(2)
    if not isinstance(cfg["thinking"], bool):
        log("[ERREUR config] thinking doit etre true/false, recu : %r."
            % cfg["thinking"])
        sys.exit(2)
    for k in ("think_directive", "overlap", "preflight"):
        if not isinstance(cfg[k], bool):
            log("[ERREUR config] %s doit etre true/false, recu : %r."
                % (k, cfg[k]))
            sys.exit(2)
    for k in ("http_timeout", "stt_timeout", "top_k", "record_backoff",
              "record_max_failures", "max_new_chars"):
        if not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] <= 0:
            log("[ERREUR config] %s doit etre un entier > 0, recu : %r."
                % (k, cfg[k]))
            sys.exit(2)
    for k in ("top_p", "temperature", "silence_rms", "vad_threshold"):
        v = cfg[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            log("[ERREUR config] %s doit etre un nombre, recu : %r." % (k, v))
            sys.exit(2)
        if not 0.0 <= float(v) <= 1.0:
            log("[ERREUR config] %s doit etre dans [0.0, 1.0], recu : %r."
                % (k, v))
            sys.exit(2)
    if cfg["think_directive"] and cfg["thinking"]:
        log("[config] think_directive et thinking actifs simultanement : "
            "/no_think sera ignore, le raisonnement est demande.")
    if cfg["think_directive"] and "qwen" not in str(cfg["summary_model"]).lower():
        log("[AVERTISSEMENT] think_directive actif sur un modele non-Qwen (%s) : "
            "le jeton /no_think sera interprete comme du texte ordinaire. "
            "Mettre \"think_directive\": false pour cette famille."
            % cfg["summary_model"])
    for k in ("transcript_path", "summary_path", "vad_model"):
        v = cfg.get(k)
        if v is not None and (not isinstance(v, str) or not v.strip()):
            log("[ERREUR config] %s doit etre null ou une chaine non vide, "
                "recu : %r." % (k, v))
            sys.exit(2)
    legacy_models = os.path.join(os.path.expanduser("~"), ".local", "share", "whisper-cpp")
    canonical_models = os.path.join(
        os.path.expanduser("~"), ".local", "share", "whisper", "models", "ggml"
    )
    model = os.path.expanduser(cfg["stt_model"])
    if model.startswith(legacy_models + os.sep):
        model = os.path.join(canonical_models, os.path.basename(model))
    if not model or not os.path.exists(model):
        log("[ERREUR config] modele whisper introuvable : %s" % cfg["stt_model"])
        sys.exit(2)
    cfg["stt_model"] = model
    return cfg


def resolve_prompt(cfg):
    """The file (prompt_file) wins if both are provided."""
    if cfg["prompt_file"]:
        path = os.path.expanduser(cfg["prompt_file"])
        try:
            with open(path, encoding="utf-8") as f:
                prompt = f.read().strip()
        except OSError as e:
            log("[ERREUR prompt] prompt_file illisible (%s) : %s" % (path, e))
            sys.exit(2)
        if not prompt:
            log("[ERREUR prompt] prompt_file vide : %s" % path)
            sys.exit(2)
        log("[prompt] consigne lue depuis : %s" % path)
        return prompt
    if not (cfg["prompt"] or "").strip():
        log("[ERREUR prompt] consigne vide (ni prompt ni prompt_file).")
        sys.exit(2)
    return cfg["prompt"]


def resolve_output_paths(transcript_path, summary_path):
    """Exact resolution rule (4 cases) for the two .txt outputs.

    - neither given : transcription.txt + resume.txt in current directory.
    - only transcript given : resume (resume.txt) created BESIDE it,
      same folder as the given transcript.
    - only summary given : transcription (transcription.txt) created
      BESIDE it, same folder as the given summary.
    - both given : respected verbatim.
    """
    t = (transcript_path or "").strip() if isinstance(transcript_path, str) else None
    s = (summary_path or "").strip() if isinstance(summary_path, str) else None
    t = t or None
    s = s or None
    if t is None and s is None:
        base = os.getcwd()
        return (os.path.join(base, "transcription.txt"),
                os.path.join(base, "resume.txt"))
    if t is not None and s is None:
        t_abs = os.path.abspath(os.path.expanduser(t))
        return t_abs, os.path.join(os.path.dirname(t_abs), "resume.txt")
    if t is None and s is not None:
        s_abs = os.path.abspath(os.path.expanduser(s))
        return os.path.join(os.path.dirname(s_abs), "transcription.txt"), s_abs
    return (os.path.abspath(os.path.expanduser(t)),
            os.path.abspath(os.path.expanduser(s)))
