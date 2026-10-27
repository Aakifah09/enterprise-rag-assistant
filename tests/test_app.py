"""Headless UI tests with Streamlit's AppTest: every page renders without exceptions.

The LLM provider is disabled and no embedding is triggered on these paths, so
the tests need neither Ollama nor a downloaded model.
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT, AppConfig
from src.knowledge_base import KnowledgeBase
from src.persistence.database import Database
from tests.conftest import HashingEmbedder

APP = str(PROJECT_ROOT / "app.py")
PAGES = [
    "app_pages/overview.py",
    "app_pages/knowledge_base.py",
    "app_pages/ask.py",
    "app_pages/playground.py",
    "app_pages/evaluation.py",
    "app_pages/history.py",
    "app_pages/architecture.py",
]


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_OUTPUTS_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("RAG_LLM_PROVIDER", "none")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield tmp_path
    st.cache_resource.clear()


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    if page != PAGES[0]:
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_on_empty_knowledge_base(app_env, page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]


def test_overview_offers_sample_data_and_reports_llm_disabled(app_env):
    at = run_page(PAGES[0])
    assert any("Load sample documents" in b.label for b in at.button)
    assert any("disabled" in w.value.lower() for w in at.sidebar.warning)


def test_pages_render_with_documents_loaded(app_env):
    cfg = AppConfig.from_env()
    kb = KnowledgeBase(cfg, Database(cfg.db_path), embedder_factory=lambda n: HashingEmbedder(n))
    for p in (PROJECT_ROOT / "sample_data" / "documents").iterdir():
        kb.add_document(p.name, p.read_bytes(), is_sample=True)
    at = run_page(PAGES[0])
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Documents"] == "3"
    at = run_page(PAGES[1])
    assert not at.exception
    assert len(at.dataframe) >= 1
    at = run_page(PAGES[4])  # evaluation lab shows the sample benchmark
    assert not at.exception
    assert any("Sample benchmark" in i.value for i in at.info)


@pytest.fixture
def loaded_app(app_env, monkeypatch):
    """App with sample documents and an offline embedder (no model download)."""
    import src.knowledge_base as kb_module

    fake = HashingEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setattr(kb_module, "get_embedder", lambda name, device="auto": fake)
    at = run_page(PAGES[0])
    next(b for b in at.button if b.label == "Load sample documents").click().run()
    assert not at.exception
    return at


def test_ask_flow_without_llm_and_follow_up_question(loaded_app):
    at = loaded_app
    at.switch_page("app_pages/ask.py").run()
    at.pills[0].set_value("Annual leave carry-over").run()  # sample-question pill
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "LLM unavailable" in md  # explicit status, no invented answer
    # A follow-up question after the pills disappear (regression: widget-state KeyError).
    at.chat_input[0].set_value("How quickly must a lost laptop be reported?").run()
    assert not at.exception, at.exception


def test_playground_compare(loaded_app):
    at = loaded_app
    at.switch_page("app_pages/playground.py").run()
    at.button[0].click().run()  # regression: form key collided with a session_state key
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1
