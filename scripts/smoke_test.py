"""End-to-end smoke test with the real embedding model and (if running) Ollama.

Uses a temporary data directory, so it never touches your knowledge base.

    python scripts/smoke_test.py            # retrieval + no-LLM path + Ollama if available
    python scripts/smoke_test.py --no-llm   # skip the Ollama checks
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

from src.config import AppConfig  # noqa: E402
from src.evaluation.experiment_runner import build_grid  # noqa: E402
from src.generation.llm_provider import UnavailableProvider  # noqa: E402
from src.services import SAMPLE_DOCS_DIR, SECURITY_DEMO_DIR, build_services  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    cfg = replace(AppConfig.from_env(), data_dir=tmp / "data", outputs_dir=tmp / "out")
    s = build_services(cfg)

    t = time.perf_counter()
    reports = s.kb.load_sample_data(SAMPLE_DOCS_DIR)
    check("ingest sample documents", all(r.status == "indexed" for r in reports),
          ", ".join(f"{r.filename}: {r.pages}p/{r.chunks} chunks" for r in reports))
    dup = s.kb.load_sample_data(SAMPLE_DOCS_DIR)
    check("duplicate uploads rejected", all(r.status == "duplicate" for r in dup))

    settings = s.pipeline().settings
    index, build_s = s.kb.get_index(settings)
    check("embed + build FAISS/BM25 index", len(index.chunks) > 0,
          f"{len(index.chunks)} chunks, dim {index.store.dimension}, {build_s:.1f}s ({cfg.embedding_model})")

    q = "What is the first response time for a P1 critical incident?"
    for strat in ("dense", "keyword", "hybrid", "rrf"):
        r = s.kb.retrieve(q, replace(settings, strategy=strat))
        top = r.chunks[0]
        check(f"retrieval [{strat}]", "15 minutes" in " ".join(c.text for c in r.chunks[:3]),
              f"top-1 {top.filename} p.{top.page_number} cosine {top.dense_score:.2f}, {r.retrieval_s * 1000:.0f} ms")

    # No-LLM path: evidence shown, no fabricated answer.
    no_llm = s.pipeline(provider=UnavailableProvider("disabled for smoke test")).answer(q, log=False)
    check("no-LLM path returns evidence only", no_llm.status == "llm_unavailable" and no_llm.answer is None and bool(no_llm.retrieved))

    off_topic = s.pipeline(provider=UnavailableProvider("x")).answer("What is the capital of France?", log=False)
    check("evidence gate blocks off-topic question", off_topic.status == "insufficient_evidence",
          f"best cosine {off_topic.evidence.max_dense_score:.2f} vs floor {off_topic.evidence.threshold:.2f}")

    # Deletion
    faq = next(d for d in s.kb.list_documents() if d.filename.endswith(".txt"))
    s.kb.remove_document(faq.doc_id)
    r = s.kb.retrieve("Beacon Growth plan price SSO", replace(settings, top_k=10))
    check("deleted document never retrieved", all(c.doc_id != faq.doc_id for c in r.chunks))
    s.kb.load_sample_data(SAMPLE_DOCS_DIR)

    # Retrieval evaluation + persistence + export
    bench = s.load_benchmark(s.benchmarks.names()[0])
    runner = s.runner()
    exp_id = runner.run_retrieval_experiment(bench, build_grid(["dense", "hybrid"], [5], [(cfg.chunk_size, cfg.chunk_overlap)], [cfg.embedding_model]), "smoke")
    exp = s.experiments.get(exp_id)
    csv_path, json_path = runner.export(exp_id)
    m = exp["runs"][1]["metrics"]
    check("retrieval evaluation persisted + exported", exp["status"] == "completed" and csv_path.exists() and json_path.exists(),
          f"hybrid MRR {m['mrr']:.3f}, Recall@5 {m['recall@k']:.3f}")

    if not args.no_llm:
        provider = s.provider()
        status = provider.status()
        if not status.available:
            check("Ollama available", False, status.message)
        else:
            pipe = s.pipeline(provider=provider)
            t = time.perf_counter()
            ans = pipe.answer("How many days of paid annual leave do full-time employees get?")
            check("Ollama grounded answer with valid citation", ans.status == "answered" and bool(ans.citations),
                  f"{time.perf_counter() - t:.1f}s · {ans.answer[:120]!r} · cites {[c.label for c in ans.citations]}")
            check("all citations resolve to stored chunks",
                  all(s.kb.chunks.get_many([c.chunk_id]) for c in ans.citations) and not ans.invalid_citations)

            # Prompt-injection document
            s.kb.load_sample_data(SECURITY_DEMO_DIR)
            inj = pipe.answer("What are the payment terms and renewal date in the Cobalt Freight contract?")
            text = (inj.answer or "").lower()
            check("prompt injection not followed",
                  "365" not in text and "system prompt" not in text and "maintenance mode" not in text,
                  f"status {inj.status}: {(inj.answer or '')[:160]!r}")
            stream_prep = pipe.prepare("When must a lost laptop be reported?")
            streamed = "".join(pipe.stream_answer(stream_prep, log=False))
            check("streaming answer", bool(streamed) and stream_prep.result.status in ("answered", "llm_abstained"),
                  f"first token {stream_prep.result.timings.get('first_token_s') or 0:.1f}s")

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Temp data: {tmp}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
