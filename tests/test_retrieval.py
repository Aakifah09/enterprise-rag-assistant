import numpy as np
import pytest

from src.embeddings.embedder import l2_normalize
from src.knowledge_base import RetrievalSettings
from src.retrieval.hybrid_retriever import minmax
from src.retrieval.keyword_retriever import BM25Index, tokenize
from src.retrieval.vector_store import VectorStore


def settings(**kw):
    base = dict(strategy="dense", top_k=3, chunk_size=200, chunk_overlap=40, embedding_model="test/hashing")
    base.update(kw)
    return RetrievalSettings(**base)


# --- vector store ------------------------------------------------------------
def test_vector_store_returns_nearest_by_cosine():
    store = VectorStore(3)
    store.add(["a", "b", "c"], l2_normalize(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)))
    hits = store.search(l2_normalize(np.array([1, 0.1, 0]))[0], 2)
    assert [h[0] for h in hits] == ["a", "c"]
    assert hits[0][1] == pytest.approx(0.995, abs=1e-3)


def test_vector_store_dimension_checks():
    store = VectorStore(4)
    with pytest.raises(ValueError):
        store.add(["a"], np.ones((1, 3), dtype=np.float32))
    store.add(["a"], l2_normalize(np.ones((1, 4))))
    with pytest.raises(ValueError):
        store.search(np.ones(3, dtype=np.float32), 1)


def test_vector_store_rejects_duplicate_ids():
    store = VectorStore(2)
    store.add(["a"], l2_normalize(np.ones((1, 2))))
    with pytest.raises(ValueError):
        store.add(["a"], l2_normalize(np.ones((1, 2))))


def test_empty_vector_store_search_returns_nothing():
    assert VectorStore(8).search(np.ones(8, dtype=np.float32), 5) == []


def test_embedding_dimension_matches_index(loaded_kb, embedder):
    index, _ = loaded_kb.get_index(settings())
    assert index.store.dimension == embedder.dimension
    assert len(index.store) == len(index.chunks) == loaded_kb.chunks.count("cs200_ov40")


# --- keyword ------------------------------------------------------------------
def test_tokenize_drops_stopwords_and_keeps_numbers():
    assert tokenize("What is the P1 response in 4.2 minutes?") == ["p1", "response", "4.2", "minutes"]


def test_bm25_prefers_documents_with_rare_query_terms():
    bm25 = BM25Index(["laptop policy laptop", "expense policy", "holiday schedule"])
    top = bm25.top("laptop", 3)
    assert top[0][0] == 0
    assert len(top) == 1  # zero-score documents are not returned


def test_minmax_handles_constant_and_empty():
    assert list(minmax(np.array([2.0, 2.0]))) == [1.0, 1.0]
    assert list(minmax(np.array([0.0, 0.0]))) == [0.0, 0.0]
    assert list(minmax(np.array([1.0, 3.0]))) == [0.0, 1.0]
    assert minmax(np.array([])).size == 0


# --- strategies ---------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["dense", "keyword", "hybrid", "rrf"])
def test_every_strategy_finds_the_relevant_chunk(loaded_kb, strategy):
    res = loaded_kb.retrieve("medical certificate sick leave consecutive days", settings(strategy=strategy))
    assert res.chunks, strategy
    assert "medical certificate" in res.chunks[0].text
    assert [c.rank for c in res.chunks] == list(range(1, len(res.chunks) + 1))
    assert all(c.strategy == strategy for c in res.chunks)


def test_scores_are_sorted_descending(loaded_kb):
    for strategy in ("dense", "keyword", "hybrid", "rrf"):
        scores = [c.score for c in loaded_kb.retrieve("leave days employees", settings(strategy=strategy, top_k=5)).chunks]
        assert scores == sorted(scores, reverse=True)


def test_hybrid_scores_are_normalised_and_report_both_components(loaded_kb):
    res = loaded_kb.retrieve("P1 first response 15 minutes", settings(strategy="hybrid", top_k=5))
    assert all(0.0 <= c.score <= 1.0 for c in res.chunks)
    assert res.chunks[0].keyword_score > 0
    assert -1.0 <= res.chunks[0].dense_score <= 1.0


def test_hybrid_alpha_extremes_match_single_retrievers(loaded_kb):
    q = "learning budget per year"
    dense = [c.chunk_id for c in loaded_kb.retrieve(q, settings(strategy="dense")).chunks]
    alpha1 = [c.chunk_id for c in loaded_kb.retrieve(q, settings(strategy="hybrid", alpha=1.0)).chunks]
    assert alpha1[0] == dense[0]


def test_metadata_traceability(loaded_kb):
    res = loaded_kb.retrieve("learning budget SGD", settings())
    top = res.chunks[0]
    assert top.filename == "handbook.pdf"
    assert top.page_number == 2
    stored = loaded_kb.chunks.get_many([top.chunk_id])[top.chunk_id]
    assert stored.text == top.text and stored.doc_id == top.doc_id


def test_markdown_chunk_cites_section(loaded_kb):
    res = loaded_kb.retrieve("annual leave carried over", settings())
    assert res.chunks[0].section == "Leave Policy > Annual leave"
    assert res.chunks[0].page_number is None


def test_empty_knowledge_base_returns_no_chunks(kb):
    res = kb.retrieve("anything", settings())
    assert res.chunks == []


def test_empty_query_rejected(loaded_kb):
    with pytest.raises(ValueError):
        loaded_kb.retrieve("   ", settings())


def test_deleted_document_never_retrieved(loaded_kb):
    s = settings(strategy="hybrid", top_k=10)
    before = loaded_kb.retrieve("laptops reported hours passwords", s)
    it_doc = next(c.doc_id for c in before.chunks if c.filename == "it_policy.txt")
    loaded_kb.get_index(s)  # make sure an index is cached
    assert loaded_kb.remove_document(it_doc)
    for strategy in ("dense", "keyword", "hybrid", "rrf"):
        after = loaded_kb.retrieve("laptops reported hours passwords", settings(strategy=strategy, top_k=10))
        assert all(c.doc_id != it_doc for c in after.chunks)
    assert loaded_kb.chunks.get_many([c.chunk_id for c in before.chunks if c.doc_id == it_doc]) == {}
    assert loaded_kb.chunks.embeddings([c.chunk_id for c in before.chunks if c.doc_id == it_doc], "test/hashing") == {}


def test_index_rebuilds_when_corpus_changes(loaded_kb):
    s = settings()
    idx1, _ = loaded_kb.get_index(s)
    idx_same, build_s = loaded_kb.get_index(s)
    assert idx_same is idx1 and build_s == 0.0
    loaded_kb.add_document("new.txt", b"Brand new policy about parking permits for staff vehicles.")
    idx2, _ = loaded_kb.get_index(s)
    assert idx2 is not idx1
    assert len(idx2.chunks) > len(idx1.chunks)


def test_embeddings_are_cached_in_sqlite(loaded_kb, embedder):
    loaded_kb.get_index(settings())
    calls = embedder.calls
    loaded_kb.clear_index_cache()  # simulate restart of the in-memory layer
    loaded_kb.get_index(settings())
    assert embedder.calls == calls  # nothing re-embedded


def test_different_chunking_configs_coexist(loaded_kb):
    a = loaded_kb.retrieve("sick leave", settings(chunk_size=200, chunk_overlap=40))
    b = loaded_kb.retrieve("sick leave", settings(chunk_size=500, chunk_overlap=0))
    assert a.chunks[0].chunk_id != b.chunks[0].chunk_id
    assert "cs500_ov0" in b.chunks[0].chunk_id
