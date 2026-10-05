"""Readability + motion acceptance tests for the two UI bugs.

Bug 1 (unreadable fault input) - the first three tests parse ``ui/theme.css``
and always run. The browser test boots a throwaway Streamlit instance on a
free port and asserts the *computed* contrast of the fault box, unfocused and
focused, right after clicking an example chip.

Bug 2 (background planes) - the last test locks down the motion rules in
``ui/sky.html`` (state on the parent window, reduced-motion speed, debug gate).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from ui import copy as C

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
THEME = (ROOT / "ui" / "theme.css").read_text(encoding="utf-8")
SKY = (ROOT / "ui" / "sky.html").read_text(encoding="utf-8")

MIN_RATIO = 4.5


# --------------------------------------------------------------------- utils --
def css_var(name: str) -> str:
    match = re.search(rf"{re.escape(name)}:\s*([^;]+);", THEME)
    assert match, f"{name} missing from ui/theme.css"
    return match.group(1).strip()


def to_rgb(color: str) -> tuple[float, float, float]:
    value = color.strip()
    if value.startswith("#"):
        digits = value[1:]
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    numbers = re.findall(r"[\d.]+", value)
    return float(numbers[0]), float(numbers[1]), float(numbers[2])


def relative_luminance(rgb: tuple[float, float, float]) -> float:
    channels = []
    for raw in rgb:
        value = raw / 255
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    first, second = relative_luminance(to_rgb(fg)), relative_luminance(to_rgb(bg))
    light, dark = max(first, second), min(first, second)
    return (light + 0.05) / (dark + 0.05)


# ------------------------------------------------------------------- Bug 1 ---
def test_field_layers_are_painted_dark_in_every_state() -> None:
    """Every layer of every field carries the dark fill, not just the control."""
    for selector in (
        '[data-testid="stTextAreaRootElement"]',
        '[data-testid="stTextInputRootElement"]',
        '[data-testid="stTextArea"] textarea',
        '[data-testid="stTextInput"] input',
        'div[data-baseweb="textarea"]',
        'div[data-baseweb="input"]',
    ):
        assert selector in THEME, f"{selector} is not styled"
    # background + text must be set on the same rule, in every state
    assert "background-color: var(--af-panel)" in THEME
    assert "-webkit-text-fill-color: var(--af-text)" in THEME
    assert "caret-color: var(--af-teal-bright)" in THEME
    for state in (":hover", ":focus", ":focus-within", ":disabled", ":-webkit-autofill", "::placeholder"):
        assert state in THEME, f"{state} state is not styled"
    # Streamlit's red focus ring on the wrapper must be neutralised
    assert re.search(
        r'\[data-testid="stTextAreaRootElement"\]:focus-within[^}]*border-color:\s*transparent',
        THEME,
    ), "the wrapper must not paint a red focus ring"


def test_theme_colours_meet_wcag_aa_for_text_and_placeholder() -> None:
    text, panel, placeholder = css_var("--af-text"), css_var("--af-panel"), css_var("--af-placeholder")
    hover = "#2A3944"  # fill used by :hover/:focus
    assert contrast_ratio(text, panel) >= MIN_RATIO, "text on field fill"
    assert contrast_ratio(text, hover) >= MIN_RATIO, "text on focused field fill"
    assert contrast_ratio(placeholder, panel) >= MIN_RATIO, "placeholder on field fill"
    assert contrast_ratio(placeholder, hover) >= MIN_RATIO, "placeholder on focused field"


def test_chip_click_fills_the_full_sentence_and_flags_focus() -> None:
    """AppTest: a chip fills the whole sentence and asks for focus."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception

    label, full_sentence = C.CHIPS[0]
    index = [b.label for b in at.button].index(label)
    at.button[index].click()
    at.run()
    assert not at.exception

    assert at.text_area[0].value == full_sentence
    assert full_sentence != label  # never just the short label
    assert at.session_state["chip_selected"] == label
    # focus_fault_box() consumed the flag and mounted the one-shot caret script
    assert any(
        "setSelectionRange" in str(getattr(frame, "srcdoc", ""))
        for frame in at.get("iframe")
    ), "chip click must mount the focus script for the fault box"


@pytest.fixture(scope="module")
def live_app():
    """A throwaway instance of the real app on a free port (temp data paths)."""
    pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="aerofix-test-"))
    (tmp / "data").mkdir()
    env = os.environ.copy()
    env.update(
        {
            "AEROFIX_SQLITE_PATH": str(tmp / "aerofix.sqlite"),
            "AEROFIX_CHROMA_DIR": str(tmp / "chroma"),
            "AEROFIX_CHECKPOINTER_PATH": str(tmp / "checkpoints.sqlite"),
            "AEROFIX_DATA_DIR": str(tmp / "data"),
            "AEROFIX_LLM_PROVIDER": "mock",
            "AEROFIX_EMBEDDING_PROVIDER": "mock",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP),
            "--server.port", str(port), "--server.headless", "true",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}"
    deadline = time.time() + 90
    ready = False
    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)
    if not ready:
        process.terminate()
        pytest.skip("could not start a throwaway Streamlit instance")
    try:
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


MEASURE = r"""(sel) => {
  const parse = (c) => {
    const m = String(c).match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number);
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const over = (fg, bg) => ({r: fg.r*fg.a + bg.r*(1-fg.a), g: fg.g*fg.a + bg.g*(1-fg.a),
                             b: fg.b*fg.a + bg.b*(1-fg.a), a: 1});
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v/12.92
      : Math.pow((v + 0.055)/1.055, 2.4); };
    return 0.2126*f(c.r) + 0.7152*f(c.g) + 0.0722*f(c.b); };
  const ratio = (a, b) => { const l1 = lum(a), l2 = lum(b);
    return (Math.max(l1,l2) + .05)/(Math.min(l1,l2) + .05); };
  const bgOf = (el) => { let n = el, acc = null;
    while (n) { const bg = parse(getComputedStyle(n).backgroundColor);
      if (bg && bg.a > 0) { acc = acc ? over(acc, bg) : bg; if (acc.a >= .999) return acc; }
      n = n.parentElement; }
    return acc || {r:16,g:33,b:44,a:1}; };
  const el = document.querySelector(sel);
  if (!el) return {missing: true};
  const cs = getComputedStyle(el);
  const bg = bgOf(el);
  const fill = parse(cs.webkitTextFillColor) || parse(cs.color);
  const rounded = (n) => Math.round(n * 100) / 100;
  const ph = el.placeholder ? parse(getComputedStyle(el, '::placeholder').color) : null;
  return {
    contrast: rounded(ratio(fill.a < 1 ? over(fill, bg) : fill, bg)),
    placeholderContrast: ph ? rounded(ratio(ph, bg)) : null,
    bg: `rgb(${Math.round(bg.r)}, ${Math.round(bg.g)}, ${Math.round(bg.b)})`,
    border: cs.borderColor,
    caret: cs.caretColor,
    shadow: cs.boxShadow || '',
    focused: document.activeElement === el,
    value: el.value || ''
  };
}"""


def _launch(playwright):
    for kwargs in ({"channel": "msedge"}, {}):
        try:
            return playwright.chromium.launch(headless=True, **kwargs)
        except Exception:
            continue
    pytest.skip("no Chromium/Edge browser available for Playwright")


def test_computed_contrast_of_fault_box_unfocused_and_focused(live_app) -> None:
    """The live page: chip click -> full sentence, readable in both states."""
    from playwright.sync_api import sync_playwright

    textarea = '[data-testid="stTextArea"] textarea'
    with sync_playwright() as playwright:
        browser = _launch(playwright)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        try:
            page.goto(live_app, wait_until="load", timeout=60000)
            page.wait_for_selector(textarea, timeout=60000)
            page.wait_for_timeout(4000)

            # unfocused, empty: text + placeholder must both be readable
            idle = page.evaluate(MEASURE, textarea)
            assert not idle.get("missing"), "fault box not rendered"
            assert idle["contrast"] >= MIN_RATIO, f"unfocused contrast {idle}"
            assert idle["placeholderContrast"] >= MIN_RATIO, f"placeholder {idle}"
            assert "255, 75, 75" not in str(idle["border"]), f"red focus ring {idle}"

            # click an example chip: full sentence, caret in the box, focused
            label, full_sentence = C.CHIPS[0]
            chips = page.locator(
                'div[data-testid="stElementContainer"][class*="st-key-chip-"] button:visible'
            )
            chips.first.click()
            page.wait_for_timeout(2000)

            assert page.input_value(textarea) == full_sentence
            focused = page.evaluate(MEASURE, textarea)
            assert focused["focused"], "chip click must place the caret in the box"
            assert focused["contrast"] >= MIN_RATIO, f"focused contrast {focused}"
            assert focused["caret"] == "rgb(111, 210, 239)", f"caret {focused['caret']}"
            assert "111, 210, 239" in focused["border"], f"focus border {focused['border']}"
            assert "rgba(111, 210, 239" in focused["shadow"], f"focus ring {focused['shadow']}"

            # the selected chip is outlined in teal
            highlight = page.evaluate(
                """() => {
                  const box = document.querySelector(
                    'div[data-testid="stElementContainer"][class*="st-key-chip-"]');
                  const btn = box && box.querySelector('button');
                  return btn ? getComputedStyle(btn).borderColor : null;
                }"""
            )
            assert highlight and "111, 210, 239" in highlight, f"chip highlight {highlight}"
        finally:
            context.close()
            browser.close()


# ------------------------------------------------------------------- Bug 2 ---
def test_sky_persists_flights_and_follows_the_motion_rules() -> None:
    """State lives on the parent window; motion rules are delta-time based."""
    # flights survive reruns / remounts instead of being re-seeded
    assert "host.__AF_SKY__" in SKY, "plane state must live on the parent window"
    assert "getElementById('af-sky-canvas')" in SKY, "canvas must be attached once"
    # z-index -1 keeps it behind #root, .stApp, the sidebar and every portal
    assert "z-index:-1" in SKY
    # OS prefers-reduced-motion slows to 30%, it must not freeze
    assert "prefers-reduced-motion: reduce" in SKY
    assert "0.3" in SKY
    # the sidebar toggle is the only thing that freezes the scene
    assert "toggleOn()" in SKY and "return 0;" in SKY
    # a pending review slows to 40% and never stops the scene
    assert "reviewHold()" in SKY and "base * 0.4" in SKY
    # delta time with a clamp, restartable loop, wall-clock heartbeat
    assert "Math.min(.05, dt)" in SKY
    assert "visibilitychange" in SKY
    assert "requestAnimationFrame" in SKY
    # on-canvas debug counter is gated behind ?debug=1
    assert "get('debug')" in SKY and "debugOverlay" in SKY
    # at least ten aircraft are kept in the air
    assert "MIN_FLYING = 10" in SKY
    # planes dim while they pass behind a content panel
    assert "UNDER_PANEL = 0.35" in SKY
