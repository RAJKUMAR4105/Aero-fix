"""Measure retrieval grounding and safety refusal behavior against local sources."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from uuid import uuid4

from aerofix.config import get_settings
from aerofix.graph import AeroFixGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AeroFix retrieval and grounding evaluation.")
    parser.add_argument("--scenarios", type=Path, default=Path("eval/scenarios.json"))
    parser.add_argument("--mock", action="store_true", help="Use local mock LLM and embeddings.")
    args = parser.parse_args()
    settings = get_settings()
    if args.mock:
        settings = settings.model_copy(
            update={"llm_provider": "mock", "embedding_provider": "mock"}
        )
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    graph = AeroFixGraph(settings)
    retrieval_hits = 0
    cited_steps = 0
    valid_cited_steps = 0
    refused = 0
    out_of_scope = 0
    latencies: list[float] = []
    try:
        for scenario in scenarios:
            started = time.perf_counter()
            thread_id = f"eval-{uuid4()}"
            graph.start(scenario["fault_input"], thread_id)
            state = graph.state(thread_id)
            latencies.append(time.perf_counter() - started)
            if state is None:
                continue
            expected_document = scenario["expected_document"].lower()
            expected_section = scenario["expected_section"].lower()
            if not scenario["out_of_scope"] and any(
                expected_document in chunk.document_title.lower()
                and expected_section in chunk.section.lower()
                for chunk in state.retrieved_chunks
            ):
                retrieval_hits += 1
            retrieved_ids = {chunk.chunk_id for chunk in state.retrieved_chunks}
            for step in (state.draft.steps if state.draft else []):
                cited_steps += 1
                if step.citation_ids and set(step.citation_ids).issubset(retrieved_ids):
                    valid_cited_steps += 1
            if scenario["out_of_scope"]:
                out_of_scope += 1
                refused += int(state.status.value == "INSUFFICIENT_DATA")
    finally:
        graph.close()
    in_scope = len(scenarios) - out_of_scope
    report = {
        "scenario_count": len(scenarios),
        "retrieval_hit_rate_at_k": round(retrieval_hits / in_scope, 4) if in_scope else None,
        "draft_step_citation_validity": round(valid_cited_steps / cited_steps, 4) if cited_steps else None,
        "out_of_scope_refusal_rate": round(refused / out_of_scope, 4) if out_of_scope else None,
        "median_latency_seconds": round(statistics.median(latencies), 4) if latencies else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

