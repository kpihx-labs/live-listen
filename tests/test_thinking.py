"""strip_thinking (6 cases) + looks_like_reasoning detector (3 cases)."""

from live_listen.pipeline import looks_like_reasoning, strip_thinking


def test_strip_tagged_block():
    out = strip_thinking("<think>some english reasoning</think>Le vrai resume.")
    assert out == "Le vrai resume."


def test_strip_unclosed_tag():
    out = strip_thinking("Debut utile <think>raisonnement sans fin")
    assert out == "Debut utile"


def test_strip_no_tags_unchanged():
    text = "Un resume francais propre, sans balises."
    assert strip_thinking(text) == text


def test_strip_only_thinking_gives_empty():
    assert strip_thinking("<think>only reasoning here</think>") == ""


def test_strip_case_insensitive():
    out = strip_thinking("<THINK>Reasoning</THINK>Resume final.")
    assert out == "Resume final."


def test_strip_none_gives_empty():
    assert strip_thinking(None) == ""
    assert strip_thinking("") == ""


def test_detector_clean_french_summary():
    text = ("L'intrigue suit un detective a Madrid. Maria dit : "
            "\"Je reviendrai demain.\" Les themes : memoire, exil.")
    assert looks_like_reasoning(text) is False


def test_detector_two_markers_fire():
    text = ("Let's start by noting the user wants a summary. "
            "I need to think step by step about the plot.")
    assert looks_like_reasoning(text) is True


def test_detector_single_marker_no_fire():
    text = "Je resume : step by step, l'intrigue avance vers Madrid."
    assert looks_like_reasoning(text) is False
