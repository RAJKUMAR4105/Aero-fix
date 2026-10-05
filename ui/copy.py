"""Every user-facing string for the AeroFix Ops Center UI.

Presentation copy lives here so wording can be reviewed and tested in one
place. Implementation jargon ("thread", "chunks", "vector store") never
appears in these strings.
"""

from __future__ import annotations

# --- page and brand -------------------------------------------------------
PAGE_TITLE = "AeroFix AI | Ops Center"
BRAND_NAME = "AeroFix AI"
BRAND_KICKER = "Maintenance Ops Center"

# --- header status pills ---------------------------------------------------
PILL_READY = "Manuals indexed \u00b7 Ready"
PILL_NO_MANUALS = "No manuals loaded"

# Phase pill: driven by the real graph state (spec A3).
PHASE_LABELS = {
    "idle": "Idle",
    "retrieving": "Searching manuals",
    "drafting": "Drafting procedure",
    "pending": "Pending review",
    "approved": "Approved",
    "rejected": "Rejected",
    "insufficient": "Insufficient data",
}

# --- hero ------------------------------------------------------------------
# U+2011 NON-BREAKING HYPHEN: "O-320" must never wrap across lines.
HERO_EYEBROW = "CESSNA 172 \u00b7 LYCOMING O\u2011320"
HERO_TITLE = "Find the fault. Cite the manual. You sign off."
HERO_SUB = (
    "Describe a symptom. AeroFix searches the manuals and drafts a cited "
    "troubleshooting procedure. Nothing is logged until a certified "
    "technician approves it."
)

# --- live fleet panel (hero right column) ----------------------------------
FLEET_TITLE = "Fleet status"
FLEET_IN_AIR = "In air"
FLEET_GROUNDED = "Grounded AOG"
FLEET_NEXT = "Next AOG fault"
AOG_BOARD = "AOG board"

# --- stepper ----------------------------------------------------------------
STEPPER_LABELS = (
    "Report fault",
    "Search manuals",
    "Draft procedure",
    "Technician review",
    "Signed off",
)

# --- report a fault ---------------------------------------------------------
SECTION_REPORT = "Report a fault"
SECTION_HELPER = "Describe what you observed, or start from a common squawk."

# (short chip label, full sentence sent to the input and shown as the tooltip).
# Clicking a chip always fills the whole sentence, never just the label.
CHIPS = (
    (
        "Rough idle on run-up",
        "Rough idle on run-up: engine runs rough at idle RPM during "
        "pre-takeoff checks",
    ),
    (
        "High oil temp, low pressure",
        "High oil temp, low pressure: oil temperature runs high while oil "
        "pressure drops in cruise",
    ),
    (
        "Magneto drop over limit",
        "Magneto drop over limit: magneto drop exceeds the limit during the "
        "run-up check",
    ),
    (
        "Carb ice on descent",
        "Carb ice on descent: engine roughness and loss of power while "
        "descending in humid air",
    ),
    (
        "Low cylinder compression",
        "Low cylinder compression: low compression found in a cylinder during "
        "a leak-down check",
    ),
)

# AOG board fallback chips: (tail number, index into CHIPS). The animated
# background is not clickable, so these are the interactive AOG board.
AOG_CHIPS = (("N-AF27", 0), ("N-AF64", 1), ("VT-ARA", 2))

INPUT_LABEL = "What did you observe?"
INPUT_PLACEHOLDER = (
    "e.g. Runs rough at 1,200 RPM; left magneto drop is 200 RPM on run-up"
)
SUBMIT_LABEL = "Search manuals & draft procedure"
EMPTY_STATE = "No active job. Report a fault above to search the manuals."

# --- running state ----------------------------------------------------------
LOADING_TITLE = "Searching the manuals"
LOADING_STEPS = "Finding pages \u00b7 Drafting steps \u00b7 Checking citations"

# --- sidebar ----------------------------------------------------------------
WORKSPACE = "Workspace"
VIEW_OPS = "Ops Center"
VIEW_AUDIT = "Audit Log"
RESUME_GROUP = "Resume a saved job"
JOB_ID_LABEL = "Job ID"
JOB_ID_PLACEHOLDER = "Paste a saved job ID"
RESUME_LABEL = "Resume job"
RESUME_OK = "Saved job resumed and ready for review."
RESUME_MISS = "No saved job was found for that Job ID."
TOGGLE_LABEL = "Reduce animations"
TOGGLE_HELP = "Use a static background and stop non-essential animation."
ACTIVE_JOB = "Active job"
KB_CARD = "Knowledge base"
KB_MANUALS = "Manuals"
KB_PASSES = "Passages"
KB_MODEL = "AI model"
CLOCK_PREFIX = "UTC"

# --- review desk ------------------------------------------------------------
DRAFT_HEADING = "Draft steps"
DRAFT_HELPER = (
    "The drafted procedure. Select a citation to jump to the manual page it "
    "came from."
)
SOURCES_HEADING = "Manual pages"
SOURCES_HELPER = "Pages found in the manuals for this fault."
GAUGE_TITLE = "Source confidence"
GAUGE_NOTE = (
    "Based on the manuals found for this fault. It is not a maintenance "
    "authorization."
)
EDIT_EXPANDER = "Edit draft before approving"
EDIT_HELPER = "Change any step, then choose Approve with edits."
COL_STEP = "Step"
COL_ACTION = "Action"
COL_CITATIONS = "Citations"
COL_WARNING = "Warning / caution"
CITATION_LABEL = "Manual reference"
SOURCE_META_RELEVANCE = "Relevance"

REVIEW_HEADING = "Technician review required"
REVIEW_BADGE = "DRAFT \u00b7 NOT APPROVED"
REVIEW_HELPER = (
    "Review every step and the manual page it cites before you sign off."
)
STEP_TITLE = "Procedure step"
CAUTION = "CAUTION"
REVIEWER_LABEL = "Technician name / certificate no."
REVIEWER_PLACEHOLDER = "e.g. J. Rivera \u00b7 A&P 4810297"
REVIEWER_HELPER = (
    "Enter a technician name and certificate number to enable Approve, "
    "Approve with edits, and Reject."
)
REJECT_LABEL = "Reason for rejection"
REJECT_PLACEHOLDER = "Required when rejecting this draft."
REJECT_REQUIRED = "Add a reason before rejecting this draft."
BTN_APPROVE = "Approve"
BTN_APPROVE_EDITS = "Approve with edits"
BTN_REJECT = "Reject"
APPROVED_STAMP = "APPROVED"


def approved_message(record_id: int, digest: str) -> str:
    """Spec-D success copy for a committed approval."""
    return f"Approved and logged as record #{record_id} \u00b7 hash {digest}"


REJECTED_MSG = (
    "Rejected. Your decision and reason are saved to the maintenance record chain."
)


def record_line(record_id: int, digest: str) -> str:
    return f"Record #{record_id} \u00b7 hash {digest}"


INSUFFICIENT_MSG = "Insufficient verified source information."
NO_DRAFT_MSG = "No procedure draft is available for review."

# --- footer ------------------------------------------------------------------
FOOTER = (
    "Draft only. Verify against current approved maintenance data before "
    "performing work."
)

# --- audit view --------------------------------------------------------------
AUDIT_TITLE = "Maintenance record chain"
AUDIT_EYEBROW = "Record integrity"
AUDIT_HELPER = (
    "Every decision is linked to the record before it, so the list can be checked."
)
AUDIT_VERIFY = "Verify chain"
AUDIT_HINT = "Select Verify chain to check the saved records."
AUDIT_NO_RECORDS = "No maintenance decisions have been saved yet."
AUDIT_EMPTY_RESULT = "There are no records to verify yet."
CARD_TECH = "Technician"
CARD_SIG = "Signature"
CARD_PREV = "Previous"


def verified_message(count: int) -> str:
    """Plain-language success copy (for example, 'All 12 records verified')."""
    return f"All {count} records verified"


def failed_message(record_id: int) -> str:
    """Plain-language failure copy (for example, 'Record #7 failed verification')."""
    return f"Record #{record_id} failed verification"


# --- errors (always a plain sentence plus a retry button) ----------------------
ERR_RETRY = "Try again"
ERR_START = "The procedure draft could not be created. Try again in a moment."
ERR_DECISION = (
    "The review decision was not saved. Check the technician details, then try again."
)
ERR_VERIFY = "The records could not be checked. Try again in a moment."
