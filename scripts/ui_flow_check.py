"""Drive the real UI flows headlessly (Streamlit AppTest) with the real embedding model.

Uses a temporary data directory. Exercises: load sample data, ask a question
(Ollama if available, otherwise the no-LLM path), retrieval playground, and a
small retrieval evaluation from the Evaluation lab.

    python scripts/ui_flow_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
tmp = Path(tempfile.mkdtemp(prefix="rag_ui_"))
os.environ["RAG_DATA_DIR"] = str(tmp / "data")
os.environ["RAG_OUTPUTS_DIR"] = str(tmp / "out")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app.py")
T = 900


def ok(cond: bool, msg: str) -> None:
    print(("[PASS] " if cond else "[FAIL] ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def no_exc(at: AppTest, where: str) -> None:
    ok(not at.exception, f"{where}: no exception" + ("" if not at.exception else f" -> {at.exception[0].value}"))


at = AppTest.from_file(APP, default_timeout=T).run()
no_exc(at, "overview")
next(b for b in at.button if b.label == "Load sample documents").click().run()
no_exc(at, "load sample data")
ok({m.label: m.value for m in at.metric}["Documents"] == "3", "3 documents indexed")

# Ask page: a real question
at.switch_page("app_pages/ask.py").run()
at.chat_input[0].set_value("How quickly must a lost or stolen laptop be reported?").run()
no_exc(at, "ask question")
md = " ".join(m.value for m in at.markdown)
answered = "Answered from sources" in md
unavailable = "LLM unavailable" in md
ok(answered or unavailable or "Model found no answer" in md, f"answer status rendered (answered={answered})")
if answered:
    # AppTest does not index expanders that have an icon, so check the rendered citation content instead.
    captions = [c.value for c in at.caption]
    ok("**Citations**" in md and any(c.startswith("Chunk `") for c in captions), "citations section with chunk ids rendered")
    ok("2 hours" in md, "answer mentions '2 hours'")

# Off-topic question is gated
at.chat_input[0].set_value("What is the capital of France?").run()
no_exc(at, "ask off-topic")
ok("Insufficient evidence" in " ".join(m.value for m in at.markdown), "off-topic question gated as insufficient evidence")

# Playground
at.switch_page("app_pages/playground.py").run()
no_exc(at, "playground render")
at.button[0].click().run()  # form submit: compare configurations
no_exc(at, "playground compare")
ok(len(at.dataframe) >= 1, "playground summary table rendered")

# Evaluation lab: small retrieval experiment
at.switch_page("app_pages/evaluation.py").run()
no_exc(at, "evaluation render")
submit = next(b for b in at.button if "Run retrieval evaluation" in b.label)
submit.click().run()
no_exc(at, "retrieval evaluation run")
ok(any("Results · experiment" in m.value for m in at.markdown), "retrieval results displayed")

# History shows it
at.switch_page("app_pages/history.py").run()
no_exc(at, "history")
ok(len(at.dataframe) >= 1, "experiment listed in history")
print(f"UI flow check complete. Temp data: {tmp}")
