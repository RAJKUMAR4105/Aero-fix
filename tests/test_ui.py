from __future__ import annotations

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from aerofix.models import Status
from ui import copy as C

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def make() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


def markdown_blob(at: AppTest) -> str:
    """All rendered markdown, with the injected stylesheet stripped out."""
    parts = [str(m.value) for m in at.markdown]
    parts += [str(m.value) for m in at.sidebar.markdown]
    joined = " \n ".join(parts)
    return re.sub(r"<style>.*?</style>", " ", joined, flags=re.S)


def button_labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


def alert_values(at: AppTest) -> list[str]:
    values: list[str] = []
    for kind in ("info", "success", "warning", "error"):
        values.extend(str(item.value) for item in getattr(at, kind))
    return values


def test_header_status_pills_are_spec_copy() -> None:
    at = make().run()
    assert not at.exception
    blob = markdown_blob(at)
    assert C.PILL_READY in blob or C.PILL_NO_MANUALS in blob
    assert C.PHASE_LABELS["idle"] in blob
    assert "pill-index" in blob and "pill-phase" in blob
    assert C.BRAND_NAME in blob and C.BRAND_KICKER in blob
    # The UTC clock mounts as a component iframe in the header column.
    assert any(getattr(f, "srcdoc", "") for f in at.get("iframe"))


def test_hero_headline_and_eyebrow() -> None:
    at = make().run()
    assert not at.exception
    blob = markdown_blob(at)
    assert C.HERO_EYEBROW in blob
    assert C.HERO_TITLE in blob
    assert C.HERO_SUB in blob
    # The non-breaking hyphen keeps "O-320" on one line.
    assert "\u2011" in C.HERO_EYEBROW
    # Live fleet panel and the interactive AOG board fallback.
    assert C.FLEET_TITLE in blob
    assert C.FLEET_IN_AIR in blob
    assert C.FLEET_GROUNDED in blob
    assert C.FLEET_NEXT in blob
    assert C.AOG_BOARD in blob


def test_stepper_labels_and_idle_fill() -> None:
    at = make().run()
    assert not at.exception
    blob = markdown_blob(at)
    for label in C.STEPPER_LABELS:
        assert label in blob
    assert "--fillw:0%" in blob
    assert "class='stepper'" in blob


def test_report_section_chips_and_fault_form() -> None:
    at = make().run()
    assert not at.exception
    blob = markdown_blob(at)
    labels = button_labels(at)
    assert C.SECTION_REPORT in blob
    assert C.SECTION_HELPER in blob
    for chip_label, full_sentence in C.CHIPS:
        assert chip_label in labels
        assert full_sentence  # full sentence goes to the input and tooltip
    assert C.SUBMIT_LABEL in labels
    assert at.text_area[0].label == C.INPUT_LABEL
    assert at.text_area[0].placeholder == C.INPUT_PLACEHOLDER
    assert C.EMPTY_STATE in alert_values(at)


def test_aog_board_chips_are_clickable_fallback() -> None:
    at = make().run()
    assert not at.exception
    labels = button_labels(at)
    for tail, index in C.AOG_CHIPS:
        assert f"{tail} \u00b7 {C.CHIPS[index][0]}" in labels


def test_sidebar_workspace_resume_and_knowledge_base() -> None:
    at = make().run()
    assert not at.exception
    labels = button_labels(at)
    assert at.sidebar.radio[0].label == C.WORKSPACE
    assert at.sidebar.radio[0].options == [C.VIEW_OPS, C.VIEW_AUDIT]
    assert any(t.label == C.TOGGLE_LABEL for t in at.sidebar.toggle)
    assert any(t.label == C.JOB_ID_LABEL for t in at.sidebar.text_input)
    assert C.RESUME_LABEL in labels
    side = " \n ".join(str(m.value) for m in at.sidebar.markdown)
    for text in (C.RESUME_GROUP, C.KB_CARD, C.KB_MANUALS, C.KB_PASSES, C.KB_MODEL):
        assert text in side
    assert "thread" not in side.lower()
    assert "vector store" not in side.lower()


def test_footer_disclaimer_present() -> None:
    at = make().run()
    assert not at.exception
    assert C.FOOTER in markdown_blob(at)


def test_no_implementation_jargon_in_visible_copy() -> None:
    at = make().run()
    assert not at.exception
    surface = " ".join(
        [markdown_blob(at)] + button_labels(at) + alert_values(at)
    ).lower()
    for banned in ("thread", "chunks", "vector store"):
        assert banned not in surface


def test_audit_view_title_button_and_plain_results() -> None:
    at = make().run()
    assert not at.exception
    at.sidebar.radio[0].set_value(C.VIEW_AUDIT)
    at.run()
    assert not at.exception
    blob = markdown_blob(at)
    labels = button_labels(at)
    assert C.AUDIT_TITLE in blob
    assert C.AUDIT_HELPER in blob
    assert C.AUDIT_VERIFY in labels

    at.button[[i for i, b in enumerate(at.button) if b.label == C.AUDIT_VERIFY][0]].click()
    at.run()
    assert not at.exception
    results = alert_values(at)
    plain_results = (
        [re.fullmatch(r"All \d+ records verified", value) for value in results]
        + [re.fullmatch(r"Record #\d+ failed verification", value) for value in results]
        + [value == C.AUDIT_EMPTY_RESULT for value in results]
    )
    assert any(plain_results), results


def test_phase_of_maps_graph_status() -> None:
    from app import PHASE_STEPPER, phase_of

    assert phase_of(None) == "idle"
    assert phase_of(Status.DRAFTING) == "drafting"
    assert phase_of(Status.PENDING_REVIEW) == "pending"
    assert phase_of(Status.APPROVED) == "approved"
    assert phase_of(Status.REJECTED) == "rejected"
    assert phase_of(Status.INSUFFICIENT_DATA) == "insufficient"
    for phase in C.PHASE_LABELS:
        assert phase in PHASE_STEPPER


def test_stepper_html_marks_pending_and_terminal_states() -> None:
    from app import stepper_html

    pending = stepper_html("pending")
    assert "is-pending" in pending and "--fillw:60%" in pending
    approved = stepper_html("approved")
    assert approved.count("is-complete") == len(C.STEPPER_LABELS)
    assert "--fillw:80%" in approved
    rejected = stepper_html("rejected")
    assert "is-terminal-red" in rejected
    insufficient = stepper_html("insufficient")
    assert "is-terminal-grey" in insufficient
    for label in C.STEPPER_LABELS:
        assert label in pending
