"""Basic import + smoke tests."""

from importlib.metadata import version


def test_import():
    from live_listen import __version__

    # `__version__` must mirror the installed distribution metadata (single source of truth),
    # never a second hardcoded copy that drifts on every release.
    assert __version__ == version("k-live-listen")


def test_config_module():
    from live_listen.config import (
        BUILTIN_FALLBACK,
        CONFIG_KEYS,
        default_config_path,
        resolve_output_paths,
    )

    assert default_config_path().endswith("live-listen/config.json")
    assert set(CONFIG_KEYS) == set(BUILTIN_FALLBACK.keys())
    # 4-case rule: neither given -> CWD files.
    import os

    t, s = resolve_output_paths(None, None)
    assert t == os.path.join(os.getcwd(), "transcription.txt")
    assert s == os.path.join(os.getcwd(), "resume.txt")


def test_every_config_key_has_a_cli_flag():
    """The project rule is "chaque cle de config a un flag CLI de meme nom".
    Checked by machine : a key added to BUILTIN_FALLBACK without its flag
    would otherwise only be reachable by hand-editing the XDG file."""
    from live_listen.cli import build_parser
    from live_listen.config import CONFIG_KEYS

    dests = {a.dest for a in build_parser()._actions}
    missing = [k for k in CONFIG_KEYS if k not in dests]
    assert missing == [], "cles sans flag CLI : %s" % missing


def test_cli_flag_types_match_their_config_values():
    """A --chunk declared as str would let "abc" through to a subprocess."""
    from live_listen.cli import build_parser
    from live_listen.config import BUILTIN_FALLBACK

    floats = {"top_p", "temperature", "silence_rms", "vad_threshold"}
    ints = {"chunk", "summary_every", "max_tokens", "top_k", "http_timeout",
            "stt_timeout", "record_backoff", "record_max_failures",
            "max_new_chars"}
    by_dest = {a.dest: a for a in build_parser()._actions}
    for key in floats:
        assert by_dest[key].type is float, key
    for key in ints:
        assert by_dest[key].type is int, key
    for key in ("thinking", "think_directive", "overlap", "preflight"):
        assert isinstance(BUILTIN_FALLBACK[key], bool), key


def test_configs_example_is_loadable_and_known():
    """The README tells the user to run
    `live-listen --config configs/en-small.json`. An example that does not
    load, or that carries an unknown key, is worse than no example."""
    import json
    import os

    from live_listen.config import CONFIG_KEYS

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "en-small.json")
    assert os.path.exists(path), "configs/en-small.json absent"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    unknown = [k for k in data if k not in CONFIG_KEYS]
    assert unknown == [], "cles inconnues dans l'exemple : %s" % unknown
    assert os.path.exists(os.path.expanduser(data["stt_model"])), "stt_model de l'exemple absent"


def test_cli():
    from live_listen.cli import main

    assert main is not None
