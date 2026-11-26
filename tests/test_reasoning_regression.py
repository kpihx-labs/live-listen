"""Regression fixtures : the two reasoning leaks actually recorded on 2026-09-23.

Verbatim from the session artefacts, not retyped from memory:

  sessions/20260923-202607-fr/resume.txt   (leak #1, 4 markers under the old
                                           flat counter)
  sessions/20260923-203051-fr/resume.txt   (leak #2, exactly 2 markers : the
                                           case the old threshold let through
                                           the moment the model reworded)

A synthetic fixture proves the function does what the author thinks. These two
prove it catches what actually happened.
"""

from live_listen.pipeline import (
    REASONING_THRESHOLD,
    looks_like_reasoning,
    matched_markers,
    reasoning_score,
)

LEAK_1 = (
    "Okay, let's start by reviewing the previous summary. The user provided an "
    "empty summary, so I'll begin with that. The new transcript starts with "
    "\"C'est parti !\" under the title \"Société Radio-Canada\". I need to "
    "translate this into French and note it as a new development. Next, I'll "
    "check for any new plot points, characters, or key elements. I should also "
    "ensure that any contradictions with previous assumptions are corrected. "
    "Since there's no previous summary, I'll start with the new information and "
    "then add any new elements. Let me structure the summary step by step, "
    "keeping track of what's been learned and any corrections made."
)

LEAK_2 = (
    "Okay, let's start by looking at the previous summary. The user mentioned "
    "that the previous summary was empty, so I'll begin with that. The new "
    "transcript includes the show's name and the subtitles. I need to add these "
    "to the summary, correct any contradictions, and keep the summary unique. "
    "Since the previous summary was empty, there's nothing to correct. I'll just "
    "add the new information."
)

# The clean summary the pipeline actually produced for the same 2026-09-23
# session, taken from sessions/20260923-201407/summary.md. It must never fire.
GOOD_SUMMARY = (
    "**RESUME PRECEDENT :**\n"
    "- **Intrigue et deroule** : Le film explore un personnage en tension entre "
    "son identite et son environnement.\n"
    "- **Vocabulaire espagnol notable** : *Espiritu*, *echarpe*.\n"
    "- **Tournures idiomatiques** : *Quand on est a l'interieur*, *Trouve une "
    "seule*, *Mais ca depend pas de moi*.\n"
    "- **References culturelles** : L'echarpe symbolisant la solitude.\n"
    "**Corrige :** Aucun contradictoire avec l'hypothese initiale."
)


def test_leak_1_is_detected():
    assert looks_like_reasoning(LEAK_1) is True


def test_leak_2_is_detected():
    """The threshold case. Under the old flat ">= 2 substrings" counter this
    text sat exactly on the boundary, so a single rewording of the model would
    have shipped English reasoning into resume.txt."""
    assert looks_like_reasoning(LEAK_2) is True


def test_leak_2_scores_well_above_threshold():
    """Not marginal : the score must leave room for a reworded variant."""
    score, labels = reasoning_score(LEAK_2)
    assert score >= REASONING_THRESHOLD + 2, (score, labels)


def test_good_summary_not_flagged():
    assert looks_like_reasoning(GOOD_SUMMARY) is False


def test_good_summary_labels_empty():
    assert matched_markers(GOOD_SUMMARY) == []


def test_structural_anchors_fire_on_short_texts():
    """A short leak with no strong substring still trips on structure alone."""
    assert looks_like_reasoning("Okay, let's begin.") is True
    assert looks_like_reasoning("I'll just add the new information.") is True
    assert looks_like_reasoning("I will add it to the summary.") is True


def test_single_medium_marker_still_below_threshold():
    assert looks_like_reasoning("step by step") is False


def test_two_medium_markers_still_below_threshold():
    assert looks_like_reasoning("step by step, i need to be careful") is False
