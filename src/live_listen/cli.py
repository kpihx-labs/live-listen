"""live-listen CLI : ecoute temps reel + transcription + resume progressif.

Pipeline : pw-record (chunks 16 kHz mono, whisper-optimal) -> whisper-cli
-> transcription.txt -> resume progressif via serveur OpenVINO local
(resume.txt, boucle dans le meme processus).

Deux sorties .txt uniquement, voir --transcript-path / --summary-path
(regle de resolution en 4 cas dans le README). Pas de sessions.

Config : CLI > --config FILE > config XDG (~/.local/share/live-listen/)
> repli integre. Chaque cle de config a un flag CLI de meme nom.
Zero dependance pip, stdlib uniquement.
"""

import argparse
import os
import sys

from live_listen import __version__
from live_listen.config import (
    default_config_path,
    effective_config,
    init_config,
    log,
    resolve_output_paths,
    resolve_prompt,
)
from live_listen.pipeline import (
    build_system_prompt,
    do_list_sources,
    run_session,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="live-listen : ecoute temps reel + transcription + "
                    "resume progressif. Precedence : CLI > --config FILE > "
                    "config XDG > repli integre.")
    # Each config key has a same-name CLI flag (default None = not given).
    p.add_argument("--lang", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--stt-model", default=None)
    p.add_argument("--stt-device", choices=["auto", "xpu", "cuda", "cpu"], default=None,
                   help="device STT : auto prefere XPU, puis CUDA, puis GPU, puis CPU")
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--summary-every", type=int, default=None)
    p.add_argument("--summary-model", default=None)
    p.add_argument("--thinking", dest="thinking", action="store_const",
                   const=True, default=None,
                   help="active enable_thinking (defaut config : false)")
    p.add_argument("--no-thinking", dest="thinking", action="store_const",
                   const=False, help="desactive enable_thinking")
    p.add_argument("--source", choices=["monitor", "mic", "auto"], default=None)
    p.add_argument("--monitor-target", default=None)
    p.add_argument("--server", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    # Sampling cut : OpenVINO's own defaults are no cut at all, which turns a
    # non-zero temperature into multilingual noise on a Gemma-class model.
    p.add_argument("--top-p", type=float, default=None,
                   help="coupe nucleus (defaut config : 0.95)")
    p.add_argument("--top-k", type=int, default=None,
                   help="coupe top-k (defaut config : 64)")
    p.add_argument("--think-directive", dest="think_directive",
                   action="store_const", const=True, default=None,
                   help="prefixe /no_think au prompt systeme, directive "
                        "template qui survit a enable_thinking=false (Qwen3)")
    p.add_argument("--no-think-directive", dest="think_directive",
                   action="store_const", const=False,
                   help="desactive /no_think (modeles non-Qwen : phi, gemma)")
    p.add_argument("--silence-rms", type=float, default=None,
                   help="porte d'energie avant STT, RMS normalise 0..1 ; "
                        "0 desactive (defaut config : 0.002)")
    p.add_argument("--vad-model", default=None,
                   help="modele VAD Silero au format GGML pour whisper-cli "
                        "(--vad). N'actif que si le fichier existe.")
    p.add_argument("--vad-threshold", type=float, default=None,
                   help="seuil du VAD natif whisper-cli (defaut config : 0.5)")
    p.add_argument("--http-timeout", type=int, default=None,
                   help="delai max d'un appel resume en secondes")
    p.add_argument("--stt-timeout", type=int, default=None,
                   help="delai max d'un appel whisper en secondes")
    p.add_argument("--overlap", dest="overlap", action="store_const",
                   const=True, default=None,
                   help="enregistre le chunk suivant pendant la transcription "
                        "(defaut config : true)")
    p.add_argument("--no-overlap", dest="overlap", action="store_const",
                   const=False,
                   help="capture strictement sequentielle (la capture attend "
                        "la transcription)")
    p.add_argument("--record-backoff", type=int, default=None,
                   help="pause apres un echec d'enregistrement, secondes")
    p.add_argument("--record-max-failures", type=int, default=None,
                   help="nb d'echecs consecutifs avant abandon de la capture")
    p.add_argument("--max-new-chars", type=int, default=None,
                   help="plafond de transcription nouvelle envoyee par resume")
    p.add_argument("--preflight", dest="preflight", action="store_const",
                   const=True, default=None,
                   help="verifie binaires, modele, serveur et coherence sink "
                        "avant le premier chunk (defaut config : true)")
    p.add_argument("--no-preflight", dest="preflight", action="store_const",
                   const=False,
                   help="lance sans pretraitement (diagnostic uniquement)")
    p.add_argument("--transcript-path", default=None,
                   help="fichier transcription .txt (defaut : transcription.txt, "
                        "voir regle de resolution en 4 cas dans le README)")
    p.add_argument("--summary-path", default=None,
                   help="fichier resume .txt (defaut : resume.txt, "
                        "voir regle de resolution en 4 cas dans le README)")
    # Operational (outside the config file).
    p.add_argument("--config", default=None,
                   help="fichier config alternatif (prioritaire sur la config XDG)")
    p.add_argument("--init", action="store_true",
                   help="cree la config XDG avec les defauts (si absente), "
                        "affiche son chemin, puis quitte")
    p.add_argument("command", nargs="?", choices=["init"],
                   help="positional alias : init = --init")
    p.add_argument("--list-sources", action="store_true",
                   help="sonde l'audio, les modeles et le serveur, puis quitte")
    p.add_argument("--version", action="version",
                   version="live-listen %s" % __version__)
    return p


def print_effective_config(cfg, prompt):
    log("[config] effective :")
    for k in ("lang", "chunk", "summary_every", "summary_model", "thinking",
              "think_directive", "top_p", "top_k", "temperature",
              "silence_rms", "overlap", "preflight", "http_timeout",
              "stt_timeout", "record_backoff", "record_max_failures",
              "max_new_chars", "source", "monitor_target", "server",
              "max_tokens", "stt_model", "stt_device", "transcript_path", "summary_path",
              "prompt_file"):
        log("  %s = %r" % (k, cfg.get(k)))
    short = (prompt[:160] + "...") if len(prompt) > 160 else prompt
    log("  prompt = %r" % short)
    log("  config XDG = %s" % default_config_path())


def main():
    args = build_parser().parse_args()

    if args.init or args.command == "init":
        init_config()
        return 0

    cfg = effective_config(args)

    if args.list_sources:
        do_list_sources(cfg)
        return 0

    prompt = resolve_prompt(cfg)
    system = (prompt + " Langue de transcription : %s. Maintiens UN resume "
              "progressif unique : conserve l'acquis, ajoute le nouveau, "
              "corrige si la suite contredit une hypothese. Ne montre jamais "
              "ton raisonnement : reponds DIRECTEMENT par le resume, en "
              "francais sauf indication contraire. N'invente rien : "
              "uniquement ce qui figure dans la transcription." % cfg["lang"])
    system = build_system_prompt(cfg, system)

    transcript_txt, summary_txt = resolve_output_paths(
        cfg.get("transcript_path"), cfg.get("summary_path"))
    for p in (os.path.dirname(os.path.abspath(transcript_txt)),
              os.path.dirname(os.path.abspath(summary_txt))):
        if p and not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    print_effective_config(cfg, prompt)

    if not os.path.exists(summary_txt):
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("# Resume progressif (en attente du premier resume)\n")
    if not os.path.exists(transcript_txt):
        open(transcript_txt, "a", encoding="utf-8").close()

    return run_session(cfg, system, transcript_txt, summary_txt)


if __name__ == "__main__":
    sys.exit(main())
