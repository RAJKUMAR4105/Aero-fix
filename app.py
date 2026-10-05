"""Streamlit Ops Center UI for the AeroFix AI review workflow.

This module deliberately contains presentation and interaction code only. The
LangGraph workflow, retrieval, review interrupt, and audit chain remain in aerofix/.
All user-facing strings live in ui/copy.py.
"""

from __future__ import annotations

from hashlib import sha1
from html import escape
from json import dumps as json_dumps
from pathlib import Path
import re
import traceback
from typing import NamedTuple
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from aerofix.config import get_settings
from aerofix.graph import AeroFixGraph, WorkflowState
from aerofix.models import ChunkRecord, DraftStep, HumanDecision, ReviewPayload, Status
from ui import copy as C

UI_DIR = Path(__file__).parent / "ui"

# Graph phase -> (active stepper index, terminal style). Approved passes every
# step, so its index overshoots the list and marks all steps complete.
PHASE_STEPPER = {
    "idle": (0, ""),
    "retrieving": (1, ""),
    "drafting": (2, ""),
    "pending": (3, ""),
    "approved": (5, ""),
    "rejected": (3, "red"),
    "insufficient": (1, "grey"),
}

CLOCK_HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html, body {{ margin: 0; background: transparent; overflow: hidden; }}
  .utc {{ font: 700 13px/1 Inter, ui-sans-serif, system-ui, sans-serif; letter-spacing: .08em;
          color: #A9B7BF; font-variant-numeric: tabular-nums; text-align: right;
          padding: 10px 2px; white-space: nowrap; }}
</style></head>
<body>
<div class="utc" aria-label="Current UTC time">{escape(C.CLOCK_PREFIX)} --:--:--</div>
<script>
  try {{ if (window.frameElement) window.frameElement.setAttribute("data-af-bg", "clock"); }} catch (e) {{}}
  (function () {{
    var el = document.querySelector(".utc");
    var prefix = {json_dumps(C.CLOCK_PREFIX)};
    function tick() {{ el.textContent = prefix + " " + new Date().toISOString().slice(11, 19); }}
    tick();
    setInterval(tick, 1000);
  }})();
</script>
</body></html>"""


FOCUS_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"></head><body style="margin:0">
<script>
  try { if (window.frameElement) window.frameElement.setAttribute("data-af-bg", "ctl"); } catch (e) {}
  (function () {
    var tries = 0;
    function focusFaultBox() {
      tries += 1;
      try {
        var box = window.parent.document.querySelector('[data-testid="stTextArea"] textarea');
        if (!box) return false;
        box.focus();
        box.setSelectionRange(box.value.length, box.value.length);
        return true;
      } catch (e) { return false; }
    }
    (function attempt() { if (!focusFaultBox() && tries < 20) setTimeout(attempt, 100); })();
  })();
</script>
</body></html>"""


class StepSlots(NamedTuple):
    """Placeholders that let the header pill and stepper change mid-run."""

    pills: st.delta_generator.DeltaGenerator
    stepper: st.delta_generator.DeltaGenerator


@st.cache_resource
def workflow() -> AeroFixGraph:
    """Keep a single UI-owned graph service per Streamlit process."""
    return AeroFixGraph(get_settings())


def inject_theme() -> None:
    """Load the local, offline CSS asset."""
    css = (UI_DIR / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sky_html(reduce_motion: bool) -> str:
    """Build the animated background with its live-fleet configuration."""
    config = {
        "reduceMotion": bool(reduce_motion),
        "calm": False,
        # Short tags keep the fleet panel tidy; the chips send their full
        # sentence to the fault box instead.
        "faults": [label for label, _full in C.CHIPS],
        "seedAog": [
            {"tail": tail, "fault": C.CHIPS[index][0]} for tail, index in C.AOG_CHIPS
        ],
    }
    raw = (UI_DIR / "sky.html").read_text(encoding="utf-8")
    return raw.replace("__SKY_CONFIG__", json_dumps(config, ensure_ascii=False))


def mount_sky(reduce_motion: bool, phase: str = "idle") -> None:
    """Mount the background first so it sits behind every content block.

    ui/sky.html attaches one canvas to the page body at z-index -1 (never over
    content) and keeps flight state on the parent window, so planes keep
    moving across reruns. The iframe below only hosts that script: it is one
    pixel wide, transparent and pointer-events disabled, so widgets always
    receive clicks; the clickable AOG board in the hero is the interactive
    fallback. The hidden marker div carries the live controls the canvas reads
    every frame - the sidebar toggle and the graph phase (a pending review
    slows the scene to 40% speed instead of stopping it).
    """
    with st.container():
        st.markdown(
            f'<div class="sky-mark" id="af-motion" data-reduce="'
            f'{1 if reduce_motion else 0}" data-phase="{escape(phase)}"></div>',
            unsafe_allow_html=True,
        )
        components.html(sky_html(reduce_motion), height=1, scrolling=False)
    if reduce_motion:
        st.markdown('<div class="motion-reduce-mark"></div>', unsafe_allow_html=True)


def knowledge_counts(graph: AeroFixGraph) -> tuple[str, str]:
    """Read local index counts defensively; this does not touch the workflow."""
    manuals, passages = "\u2014", "\u2014"
    try:
        collection = graph.retriever.backend.collection
        passages = str(collection.count())
        metadata = collection.get(include=["metadatas"]).get("metadatas", [])
        manuals = str(
            len(
                {
                    item.get("document_title")
                    for item in metadata
                    if item and item.get("document_title")
                }
            )
        )
    except Exception:
        pass
    return manuals, passages


def show_error(message: str, key: str) -> bool:
    """Plain sentence plus a retry button; never a raw traceback."""
    st.error(message, icon=":material/error:")
    return st.button(C.ERR_RETRY, key=key)


def phase_of(status: Status | None) -> str:
    if status is None:
        return "idle"
    return {
        Status.DRAFTING: "drafting",
        Status.PENDING_REVIEW: "pending",
        Status.APPROVED: "approved",
        Status.REJECTED: "rejected",
        Status.INSUFFICIENT_DATA: "insufficient",
    }[status]


def brand_html() -> str:
    return f"""
    <div class="brand-lockup">
      <svg class="brand-mark" viewBox="0 0 48 48" role="img" aria-label="{escape(C.BRAND_NAME)} logo">
        <path fill="#1C7293" d="M24 2 43 11v15L24 46 5 26V11z"/>
        <path fill="#D9F4FA" d="m12 27 10-3 4-13 3 1-1 13 9 4-1 3-10-2-6 8-3-1 3-8-8 1z"/>
        <circle cx="35" cy="13" r="3" fill="#E0A030"/>
      </svg>
      <div class="brand-text">
        <div class="brand-title">{escape(C.BRAND_NAME)}</div>
        <div class="brand-kicker">{escape(C.BRAND_KICKER)}</div>
      </div>
    </div>"""


def pills_html(ready: bool, phase: str) -> str:
    index_text = C.PILL_READY if ready else C.PILL_NO_MANUALS
    index_class = "pill-index" if ready else "pill-index is-warn"
    phase_label = C.PHASE_LABELS.get(phase, C.PHASE_LABELS["idle"])
    return (
        '<div class="header-pills">'
        f'<span class="pill {index_class}">{escape(index_text)}</span>'
        f'<span class="pill pill-phase ph-{escape(phase)}">{escape(phase_label)}</span>'
        "</div>"
    )


def render_header(ready: bool, phase: str) -> st.delta_generator.DeltaGenerator:
    """Brand, the ticking UTC clock, and the two status pills."""
    left, right = st.columns([2.3, 1.4])
    with left:
        st.markdown(brand_html(), unsafe_allow_html=True)
    with right:
        components.html(CLOCK_HTML, height=34, scrolling=False)
        pills_slot = st.empty()
        pills_slot.markdown(pills_html(ready, phase), unsafe_allow_html=True)
    return pills_slot


def stepper_html(phase: str) -> str:
    """Full-width stepper with a progress line driven by the real graph state."""
    active, terminal = PHASE_STEPPER.get(phase, (0, ""))
    capped = min(active, len(C.STEPPER_LABELS) - 1)
    fill = round(capped / (len(C.STEPPER_LABELS) - 1) * 80, 2)
    nodes: list[str] = []
    for index, label in enumerate(C.STEPPER_LABELS):
        classes, symbol = ["step"], str(index + 1)
        if index < active:
            classes.append("is-complete")
            symbol = "\u2713"
        elif index == active:
            if terminal == "red":
                classes.append("is-terminal-red")
                symbol = "!"
            elif terminal == "grey":
                classes.append("is-terminal-grey")
                symbol = "\u2013"
            elif phase == "pending":
                classes.append("is-pending")
            else:
                classes.append("is-active")
        nodes.append(
            f"<li class='{' '.join(classes)}'>"
            f"<span class='step-dot'>{symbol}</span>"
            f"<span class='step-label'>{escape(label)}</span></li>"
        )
    return (
        f"<ol class='stepper' style='--fillw:{fill:g}%' aria-label='Progress'>"
        + "".join(nodes)
        + "</ol>"
    )


def fleet_html() -> str:
    return f"""
    <div class="fleet-panel">
      <div class="fleet-title">{escape(C.FLEET_TITLE)}</div>
      <div class="fleet-row"><span>{escape(C.FLEET_IN_AIR)}</span><b id="fleet-air">12</b></div>
      <div class="fleet-row"><span>{escape(C.FLEET_GROUNDED)}</span><b id="fleet-aog">3</b></div>
      <div class="fleet-row fleet-next"><span>{escape(C.FLEET_NEXT)}</span><b id="fleet-next">{escape(C.CHIPS[0][0])}</b></div>
    </div>"""


def render_hero() -> None:
    """Hero card: headline on the left, live fleet status plus the AOG board."""
    with st.container():
        st.markdown('<div class="hero-mark"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown(
                "<section class='hero-copy'>"
                f"<div class='eyebrow'>{escape(C.HERO_EYEBROW)}</div>"
                f"<h1>{escape(C.HERO_TITLE)}</h1>"
                f"<p>{escape(C.HERO_SUB)}</p>"
                "</section>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(fleet_html(), unsafe_allow_html=True)
            st.markdown(
                f"<div class='col-label'>{escape(C.AOG_BOARD)}</div>",
                unsafe_allow_html=True,
            )
            for tail, index in C.AOG_CHIPS:
                label, full = C.CHIPS[index]
                if st.button(
                    f"{tail} \u00b7 {label}",
                    key=f"aog-{tail}",
                    help=full,
                    use_container_width=True,
                ):
                    fill_fault_box(full, label)


def fill_fault_box(sentence: str, label: str | None = None) -> None:
    """Fill the fault box with a whole sentence and remember which chip did it."""
    st.session_state["fault_input"] = sentence
    if label:
        st.session_state["chip_selected"] = label
    st.session_state["focus_fault"] = True


def selected_chip_css() -> None:
    """Teal outline on the example chip that filled the fault box."""
    label = st.session_state.get("chip_selected")
    if not label:
        return
    slug = re.sub(r"[^A-Za-z0-9_]", "-", label)
    st.markdown(
        "<style>"
        f'.stApp div[data-testid="stElementContainer"].st-key-chip-{slug} button {{'
        "border-color: var(--af-teal-bright) !important;"
        "background: rgba(28, 114, 147, 0.42) !important;"
        "box-shadow: inset 0 0 0 1px rgba(111, 210, 239, 0.75) !important;"
        "color: var(--af-text) !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )


def focus_fault_box() -> None:
    """Put the caret in the fault box right after a chip click (one-shot)."""
    if st.session_state.pop("focus_fault", False):
        components.html(FOCUS_HTML, height=0, scrolling=False)


def render_command_bar() -> str:
    """Section heading, wrapping example-fault chips, then the fault form."""
    st.markdown(
        "<div class='section-head'>"
        f"<h2>{escape(C.SECTION_REPORT)}</h2>"
        f"<p>{escape(C.SECTION_HELPER)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="chip-row-mark"></div>', unsafe_allow_html=True)
        for label, full in C.CHIPS:
            if st.button(
                label, key=f"chip-{label}", help=full, use_container_width=False
            ):
                fill_fault_box(full, label)
        # After the loop: a chip click sets `chip_selected` while its button is
        # rendered, so the highlight must be emitted later in the same run.
        selected_chip_css()
    with st.form("fault-form", clear_on_submit=False):
        fault_input = st.text_area(
            C.INPUT_LABEL,
            key="fault_input",
            placeholder=C.INPUT_PLACEHOLDER,
            height=110,
            # The box unmounts on the Audit Log view; keep what was typed (or
            # filled in by a chip) for the whole session so it is still there
            # when the user comes back.
            persist_state="session",
        )
        submitted = st.form_submit_button(
            C.SUBMIT_LABEL, type="primary", use_container_width=True
        )
    focus_fault_box()
    if submitted:
        st.session_state.pop("chip_selected", None)
        return (fault_input or "").strip()
    return ""


def loading_html() -> str:
    return (
        '<div class="loading-run" role="status" aria-live="polite">'
        '<div class="taxi-plane" aria-hidden="true"></div>'
        f"<div class='loading-copy'>{escape(C.LOADING_TITLE)}"
        f"<span>{escape(C.LOADING_STEPS)}</span></div></div>"
    )


def start_job(
    graph: AeroFixGraph, fault_input: str, slots: StepSlots, ready: bool
) -> str | None:
    """Flip the header and stepper to 'retrieving' before blocking on the graph."""
    loading = st.empty()
    try:
        slots.pills.markdown(pills_html(ready, "retrieving"), unsafe_allow_html=True)
        slots.stepper.markdown(stepper_html("retrieving"), unsafe_allow_html=True)
        loading.markdown(loading_html(), unsafe_allow_html=True)
        new_thread_id = str(uuid4())
        graph.start(fault_input, new_thread_id)
    except Exception:
        traceback.print_exc()  # keep the real traceback in the terminal
        loading.empty()
        return C.ERR_START
    st.session_state.thread_id = new_thread_id
    for stale in ("decision_error", "decision_error_msg", "pending_decision"):
        st.session_state.pop(stale, None)
    st.rerun()
    return None


def draft_rows(state: WorkflowState) -> list[dict[str, str | int]]:
    if not state.draft:
        return []
    return [
        {
            "n": step.n,
            "action": step.action,
            "citation_ids": ", ".join(step.citation_ids),
            "warning": step.warning or "",
        }
        for step in state.draft.steps
    ]


def edited_steps(rows: list[dict[str, object]]) -> list[DraftStep]:
    return [
        DraftStep(
            n=int(row["n"]),
            action=str(row["action"]).strip(),
            citation_ids=[
                item.strip()
                for item in str(row["citation_ids"]).split(",")
                if item.strip()
            ],
            warning=str(row.get("warning", "")).strip() or None,
        )
        for row in rows
    ]


def source_anchor(chunk_id: str) -> str:
    return f"source-{sha1(chunk_id.encode('utf-8')).hexdigest()[:12]}"


def render_confidence_gauge(confidence: float) -> None:
    value = min(100, max(0, round(confidence * 100)))
    st.markdown(
        f"""
        <div class="glass-card confidence-wrap" aria-label="Source confidence {value} percent">
          <div class="gauge" style="--value:{value}"><span class="gauge-value">{value}%</span></div>
          <div class="gauge-label"><strong>{escape(C.GAUGE_TITLE)}</strong><br>{escape(C.GAUGE_NOTE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_draft_step(step: DraftStep, chunks: list[ChunkRecord]) -> None:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    citations = []
    for citation_id in step.citation_ids:
        chunk = chunks_by_id.get(citation_id)
        label = f"{chunk.document_title} p.{chunk.page}" if chunk else citation_id
        citations.append(
            f"<a class='citation-chip' href='#{source_anchor(citation_id)}' "
            f"aria-label='{escape(C.CITATION_LABEL)}: {escape(label)}'>"
            f"[{escape(label)}]</a>"
        )
    warning = (
        f"<div class='warning-banner'><strong>{escape(C.CAUTION)}</strong> · "
        f"{escape(step.warning)}</div>"
        if step.warning
        else ""
    )
    st.markdown(
        f"""
        <article class="review-card">
          <h3><span class="step-number">{step.n}</span>{escape(C.STEP_TITLE)}</h3>
          <p>{escape(step.action)}</p>{warning}
          <div aria-label="References for step {step.n}">{''.join(citations)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(chunk: ChunkRecord) -> None:
    score = min(1.0, max(0.0, chunk.score or 0.0))
    text = escape(chunk.text).replace("\n", "<br>")
    st.markdown(
        f"""
        <article id="{source_anchor(chunk.chunk_id)}" class="source-card" tabindex="-1">
          <h3>{escape(chunk.document_title)}</h3>
          <div class="source-meta">{escape(chunk.source)} · Page {chunk.page} · {escape(chunk.section)}</div>
          <p>{text}</p>
          <div class="source-meta">{escape(C.SOURCE_META_RELEVANCE)} {score:.0%}</div>
          <div class="relevance" aria-label="Relevance score {score:.0%}"><span style="width:{score * 100:.1f}%"></span></div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def latest_thread_record(graph: AeroFixGraph, thread_id: str):
    return next(
        (record for record in graph.audit_log.records() if record.thread_id == thread_id),
        None,
    )


def sidebar(graph: AeroFixGraph) -> tuple[str, bool, bool]:
    """Workspace switch, job resume, animation control, and the knowledge card."""
    manuals, passages = knowledge_counts(graph)
    ready = manuals not in ("\u2014", "0")
    with st.sidebar:
        st.markdown(
            f"<div class='side-eyebrow'>{escape(C.WORKSPACE)}</div>",
            unsafe_allow_html=True,
        )
        view = st.radio(
            C.WORKSPACE, [C.VIEW_OPS, C.VIEW_AUDIT], label_visibility="collapsed"
        )
        reduce_motion = st.toggle(
            C.TOGGLE_LABEL,
            key="reduce_motion",
            help=C.TOGGLE_HELP,
        )
        st.divider()
        st.markdown(
            f"<div class='side-eyebrow'>{escape(C.RESUME_GROUP)}</div>",
            unsafe_allow_html=True,
        )
        job_id = st.text_input(
            C.JOB_ID_LABEL, key="resume_job_id", placeholder=C.JOB_ID_PLACEHOLDER
        )
        if st.button(C.RESUME_LABEL, key="resume-job", use_container_width=True):
            restored = graph.state(job_id.strip()) if (job_id or "").strip() else None
            if restored:
                st.session_state.thread_id = job_id.strip()
                st.session_state.notice = C.RESUME_OK
                st.session_state.pop("resume_error", None)
                st.rerun()
            st.session_state["resume_error"] = True
        if st.session_state.get("resume_error"):
            if show_error(C.RESUME_MISS, "retry-resume"):
                st.session_state.pop("resume_error", None)
        current_thread = st.session_state.get("thread_id")
        if current_thread:
            st.caption(f"{C.ACTIVE_JOB}: `{current_thread}`")
        st.markdown(
            f"""
            <aside class="kb-card" aria-label="{escape(C.KB_CARD)}">
              <h3>{escape(C.KB_CARD)}</h3>
              <dl>
                <dt>{escape(C.KB_MANUALS)}</dt><dd>{escape(manuals)}</dd>
                <dt>{escape(C.KB_PASSES)}</dt><dd>{escape(passages)}</dd>
                <dt>{escape(C.KB_MODEL)}</dt><dd>{escape(graph.settings.llm_model)}</dd>
              </dl>
            </aside>
            """,
            unsafe_allow_html=True,
        )
    return view, bool(reduce_motion), ready


def commit_review(
    graph: AeroFixGraph,
    thread_id: str,
    decision: HumanDecision,
    reviewer_id: str,
    reject_reason: str,
    rows: list[dict[str, object]],
) -> str | None:
    """Resume the interrupted workflow; return a plain error sentence on failure."""
    reason = (reject_reason or "").strip()
    if decision == HumanDecision.REJECT and not reason:
        return C.REJECT_REQUIRED
    try:
        payload = ReviewPayload(
            human_decision=decision,
            reviewer_id=reviewer_id,
            edited_steps=edited_steps(rows) if decision == HumanDecision.EDIT else None,
            reject_reason=reason or None,
        )
        graph.resume(thread_id, payload)
    except Exception:
        traceback.print_exc()  # keep the real traceback in the terminal
        return C.ERR_DECISION
    for stale in ("decision_error", "decision_error_msg", "pending_decision"):
        st.session_state.pop(stale, None)
    st.rerun()
    return None


def review_desk(graph: AeroFixGraph, ready: bool, reduce_motion: bool) -> None:
    thread_id = st.session_state.get("thread_id")
    state = graph.state(thread_id) if thread_id else None
    phase = phase_of(state.status if state else None)

    # Mounted first so the canvas stays behind every content block; the marker
    # inside it carries the live toggle and graph phase for the background.
    mount_sky(reduce_motion, phase)

    pills_slot = render_header(ready, phase)
    render_hero()
    stepper_slot = st.empty()
    stepper_slot.markdown(stepper_html(phase), unsafe_allow_html=True)
    slots = StepSlots(pills=pills_slot, stepper=stepper_slot)

    if st.session_state.pop("notice", None):
        st.success(C.RESUME_OK, icon=":material/check_circle:")

    fault_input = render_command_bar()

    errored = bool(st.session_state.get("start_error"))
    if errored:
        if show_error(C.ERR_START, "retry-start"):
            fault_input = st.session_state.pop("pending_fault", "") or fault_input
            st.session_state.pop("start_error", None)
            errored = False

    if fault_input:
        error = start_job(graph, fault_input, slots, ready)
        if error:
            st.session_state["start_error"] = True
            st.session_state["pending_fault"] = fault_input
            st.rerun()

    if not state:
        if not errored:
            st.info(C.EMPTY_STATE, icon=":material/info:")
        return

    if state.status == Status.INSUFFICIENT_DATA:
        st.warning(C.INSUFFICIENT_MSG, icon=":material/warning:")
        return

    if state.status in {Status.APPROVED, Status.REJECTED}:
        record = latest_thread_record(graph, thread_id)
        digest = str(record.record_hash)[:8] if record else ""
        if state.status == Status.APPROVED:
            st.markdown(
                f"<div class='approved-stamp'>{escape(C.APPROVED_STAMP)}</div>",
                unsafe_allow_html=True,
            )
            message = (
                C.approved_message(record.id, digest) if record else C.APPROVED_STAMP
            )
            st.success(message, icon=":material/check_circle:")
        else:
            st.info(C.REJECTED_MSG, icon=":material/info:")
            if record:
                st.caption(C.record_line(record.id, digest))
        return

    if not state.draft:
        st.warning(C.NO_DRAFT_MSG, icon=":material/warning:")
        return

    left, right = st.columns((1.15, 0.85), gap="large")
    rows: list[dict[str, object]] = draft_rows(state)
    with left:
        st.markdown(
            "<div class='col-head'>"
            f"<h2>{escape(C.DRAFT_HEADING)}</h2>"
            f"<p>{escape(C.DRAFT_HELPER)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        for step in state.draft.steps:
            render_draft_step(step, state.retrieved_chunks)
        with st.expander(C.EDIT_EXPANDER, expanded=False):
            st.markdown(
                f"<p class='field-helper'>{escape(C.EDIT_HELPER)}</p>",
                unsafe_allow_html=True,
            )
            rows = st.data_editor(
                draft_rows(state),
                key=f"draft-{thread_id}",
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "n": st.column_config.NumberColumn(
                        C.COL_STEP, min_value=1, required=True
                    ),
                    "action": st.column_config.TextColumn(
                        C.COL_ACTION, required=True, width="large"
                    ),
                    "citation_ids": st.column_config.TextColumn(
                        C.COL_CITATIONS, required=True
                    ),
                    "warning": st.column_config.TextColumn(
                        C.COL_WARNING, width="large"
                    ),
                },
            )
    with right:
        render_confidence_gauge(state.confidence)
        st.markdown(
            "<div class='col-head'>"
            f"<h2>{escape(C.SOURCES_HEADING)}</h2>"
            f"<p>{escape(C.SOURCES_HELPER)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        for chunk in state.retrieved_chunks:
            render_source_card(chunk)

    st.markdown(
        "<section class='signoff-panel'>"
        "<div class='signoff-head'>"
        f"<h2>{escape(C.REVIEW_HEADING)}</h2>"
        f"<span class='draft-badge'>{escape(C.REVIEW_BADGE)}</span>"
        "</div>"
        f"<p class='signoff-help'>{escape(C.REVIEW_HELPER)}</p>"
        "</section>",
        unsafe_allow_html=True,
    )
    reviewer_id = st.text_input(
        C.REVIEWER_LABEL, key=f"reviewer-{thread_id}", placeholder=C.REVIEWER_PLACEHOLDER
    )
    reject_reason = st.text_area(
        C.REJECT_LABEL,
        key=f"reject-{thread_id}",
        placeholder=C.REJECT_PLACEHOLDER,
        height=88,
    )
    disabled = not reviewer_id.strip()
    if disabled:
        st.markdown(
            f"<p class='field-helper'>{escape(C.REVIEWER_HELPER)}</p>",
            unsafe_allow_html=True,
        )
    approve, edit_approve, reject = st.columns(3)
    decision: HumanDecision | None = None
    if approve.button(
        C.BTN_APPROVE, type="primary", use_container_width=True, disabled=disabled
    ):
        decision = HumanDecision.APPROVE
    if edit_approve.button(
        C.BTN_APPROVE_EDITS, use_container_width=True, disabled=disabled
    ):
        decision = HumanDecision.EDIT
    if reject.button(C.BTN_REJECT, use_container_width=True, disabled=disabled):
        decision = HumanDecision.REJECT

    if st.session_state.get("decision_error"):
        if show_error(
            st.session_state.get("decision_error_msg", C.ERR_DECISION),
            "retry-decision",
        ):
            decision = st.session_state.pop("pending_decision", None)
            st.session_state.pop("decision_error", None)
            st.session_state.pop("decision_error_msg", None)

    if decision:
        error = commit_review(
            graph, thread_id, decision, reviewer_id, reject_reason, rows
        )
        if error:
            st.session_state["decision_error"] = True
            st.session_state["decision_error_msg"] = error
            st.session_state["pending_decision"] = decision
            st.rerun()


def run_verify(graph: AeroFixGraph) -> tuple[bool, str, bool]:
    """Map audit chain output to plain language: (ok, message, empty)."""
    try:
        valid, raw = graph.audit_log.verify_chain()
    except Exception:
        return False, C.ERR_VERIFY, False
    if valid:
        match = re.search(r"Verified (\d+)", raw)
        count = int(match.group(1)) if match else 0
        if count == 0:
            return True, C.AUDIT_EMPTY_RESULT, True
        return True, C.verified_message(count), False
    match = re.search(r"Record (\d+)", raw)
    if match:
        return False, C.failed_message(int(match.group(1))), False
    return False, C.ERR_VERIFY, False


def audit_view(graph: AeroFixGraph, ready: bool, reduce_motion: bool) -> None:
    mount_sky(reduce_motion)
    render_header(ready, "idle")
    st.markdown(
        "<section class='hero'>"
        f"<div class='eyebrow'>{escape(C.AUDIT_EYEBROW)}</div>"
        f"<h1>{escape(C.AUDIT_TITLE)}</h1>"
        f"<p>{escape(C.AUDIT_HELPER)}</p>"
        "</section>",
        unsafe_allow_html=True,
    )

    if st.button(C.AUDIT_VERIFY, type="primary", key="verify-chain"):
        st.session_state.audit_verification = run_verify(graph)
        st.session_state.audit_scanning = True
    verification = st.session_state.get("audit_verification")
    if verification:
        ok, message, empty = verification
        if empty:
            st.info(message, icon=":material/info:")
        elif ok:
            st.success(message, icon=":material/check_circle:")
        else:
            st.error(message, icon=":material/error:")
            if st.button(C.ERR_RETRY, key="retry-verify"):
                st.session_state.audit_verification = run_verify(graph)
                st.session_state.audit_scanning = True
                st.rerun()
    else:
        st.info(C.AUDIT_HINT, icon=":material/info:")

    records = list(reversed(graph.audit_log.records()))
    if not records:
        st.info(C.AUDIT_NO_RECORDS, icon=":material/info:")
        return
    broken_match = (
        re.search(r"Record #(\d+)", verification[1])
        if verification and not verification[0]
        else None
    )
    broken_id = int(broken_match.group(1)) if broken_match else None
    scanning = bool(st.session_state.pop("audit_scanning", False))
    cards = []
    for index, record in enumerate(records):
        classes = ["audit-card"]
        if verification and verification[0]:
            classes.append("is-valid")
        if broken_id == record.id:
            classes.append("is-broken")
        decision = escape(record.decision.upper())
        cards.append(
            f"<article class='{' '.join(classes)}' style='--delay:{index * .15:.2f}s'>"
            f"<h3>#{record.id} · {decision}</h3>"
            f"<p>{escape(record.timestamp.strftime('%Y-%m-%d %H:%M UTC'))}</p>"
            f"<p>{escape(C.CARD_TECH)} · {escape(record.reviewer_id)}</p>"
            f"<p>{escape(C.CARD_SIG)} · <code>{escape(str(record.record_hash)[:12])}</code></p>"
            f"<p>{escape(C.CARD_PREV)} · <code>{escape(record.prev_hash[:12])}</code></p>"
            "</article>"
        )
    class_name = "audit-chain is-scanning" if scanning else "audit-chain"
    st.markdown(
        f"<section class='{class_name}' aria-label='Maintenance records'>"
        f"{''.join(cards)}</section>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=C.PAGE_TITLE, page_icon="✈", layout="wide")
    inject_theme()
    graph = workflow()
    view, reduce_motion, ready = sidebar(graph)
    if view == C.VIEW_OPS:
        review_desk(graph, ready, reduce_motion)
    else:
        audit_view(graph, ready, reduce_motion)
    st.markdown(
        f"<div class='page-footer'>{escape(C.FOOTER)}</div>", unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
