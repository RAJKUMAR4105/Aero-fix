"""Provider-agnostic structured drafting constrained to retrieved chunks."""

from __future__ import annotations

import os
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from aerofix.config import Settings, get_settings
from aerofix.models import ChunkRecord, DraftResponse, DraftStep

DISCLAIMER = (
    "Draft only. Requires certified technician review and must be verified against "
    "current approved maintenance data."
)


class DraftingLLM(Protocol):
    """Port for a structured maintenance-draft provider."""

    def draft(self, fault_input: str, chunks: list[ChunkRecord]) -> DraftResponse: ...


class ProviderLLM:
    """LangChain provider adapter with Pydantic structured output."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def draft(self, fault_input: str, chunks: list[ChunkRecord]) -> DraftResponse:
        prompt = _prompt(fault_input, chunks)
        provider = self.settings.llm_provider.lower()
        if provider == "openai":
            model = ChatOpenAI(model=self.settings.llm_model, temperature=0)
        elif provider == "anthropic":
            model = ChatAnthropic(model=self.settings.llm_model, temperature=0)
        else:
            raise ValueError("Unsupported LLM provider. Use 'openai', 'anthropic', or 'mock'.")
        response = model.with_structured_output(DraftResponse).invoke(prompt)
        draft = response if isinstance(response, DraftResponse) else DraftResponse.model_validate(response)
        return draft.model_copy(update={"disclaimer": DISCLAIMER})


class MockLLM:
    """Source-only local draft generator for smoke tests; never for real maintenance use."""

    def draft(self, fault_input: str, chunks: list[ChunkRecord]) -> DraftResponse:
        if not chunks:
            return DraftResponse(unresolved_questions=["No retrieved source material is available."])
        source = chunks[0]
        warning = next(
            (line.strip() for line in source.text.splitlines() if line.startswith(("WARNING", "CAUTION"))),
            None,
        )
        action = source.text.split("\n\n")[0].strip()
        return DraftResponse(
            steps=[DraftStep(n=1, action=action, citation_ids=[source.chunk_id], warning=warning)],
            unresolved_questions=["Smoke-test draft only; consult approved current maintenance data."],
        )


def get_llm(settings: Settings | None = None) -> DraftingLLM:
    """Resolve the drafting provider.

    If the configured provider's API credential is missing, fall back to the
    local MockLLM (source-grounded extractive draft) instead of crashing inside
    the OpenAI/Anthropic client constructor with `Missing credentials`.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockLLM()
    api_key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if api_key_env and not os.environ.get(api_key_env):
        print(
            f"[aerofix] {api_key_env} is not set; using the local MockLLM "
            f"(grounded demo draft) instead of provider '{provider}'."
        )
        return MockLLM()
    return ProviderLLM(settings)


def _prompt(fault_input: str, chunks: list[ChunkRecord]) -> str:
    source_text = "\n\n".join(
        f"[chunk_id={chunk.chunk_id}; source={chunk.source}; page={chunk.page}; section={chunk.section}]\n{chunk.text}"
        for chunk in chunks
    )
    return f"""You are drafting a maintenance troubleshooting procedure for a certified
aviation technician. Use only the source chunks below. Do not infer, supplement, or
invent knowledge. Every step must cite one or more provided chunk IDs. Copy WARNING or
CAUTION text verbatim into the warning field of its corresponding step.

Fault input: {fault_input}

Return JSON matching this schema exactly:
{{"steps":[{{"n":1,"action":"...","citation_ids":["chunk id"],"warning":"optional"}}],
"unresolved_questions":["..."]}}

Source chunks:
{source_text}
"""

