"""Index construction, search sanity, and on-disk round-trip."""

from __future__ import annotations

from pathlib import Path

from rag_eval.config import load_config
from rag_eval.ingest.indexer import build_index, load_index
from rag_eval.retrieve.bm25 import BM25Index
from rag_eval.retrieve.dense import InMemoryVectorStore
from rag_eval.retrieve.embedder import HashingEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "config" / "base.yaml"


def test_bm25_ranks_lexical_match_first() -> None:
    ids = ["a", "b", "c"]
    texts = [
        "structured query language for relational databases",
        "the sun is a star at the center of the solar system",
        "mars is the red planet",
    ]
    index = BM25Index.build(ids, texts)
    top = index.search("relational database query", top_k=1)
    assert top[0][0] == "a"


def test_bm25_empty_query_terms_returns_results_without_error() -> None:
    index = BM25Index.build(["a"], ["hello world"])
    # query with no overlapping terms still returns a ranked list (scores ~0)
    assert index.search("zzz", top_k=1)[0][0] == "a"


def test_vector_store_exact_search() -> None:
    emb = HashingEmbedder(dim=256)
    ids = ["mars", "sql"]
    store = InMemoryVectorStore()
    store.add(ids, emb.embed(["mars red planet two moons", "sql relational database tables"]))
    hit = store.search(emb.embed(["planet mars moons"])[0], top_k=1)
    assert hit[0][0] == "mars"
    assert len(store) == 2


def test_build_and_search_on_sample_corpus() -> None:
    config = load_config(BASE_CONFIG)
    bundle = build_index(config, persist=False)
    assert len(bundle.chunks) >= len(bundle.chunks_by_id)  # ids unique
    assert len(bundle.chunks_by_id) == len(bundle.chunks)

    qvec = bundle.embedder.embed(["red planet with two small moons"])[0]
    top_id = bundle.vector_store.search(qvec, top_k=1)[0][0]
    assert bundle.chunks_by_id[top_id].doc_id == "doc-015"  # Mars

    bm25_top = bundle.bm25.search("structured query language relational", top_k=1)[0][0]
    assert bundle.chunks_by_id[bm25_top].doc_id == "doc-004"  # SQL


def test_index_round_trip(tmp_path: Path) -> None:
    config = load_config(BASE_CONFIG)
    built = build_index(config, persist=True, root=tmp_path)
    loaded = load_index(config, root=tmp_path)

    assert [c.id for c in loaded.chunks] == [c.id for c in built.chunks]
    q = loaded.embedder.embed(["jupiter the largest gas giant planet"])[0]
    assert (
        loaded.vector_store.search(q, top_k=1)[0][0] == built.vector_store.search(q, top_k=1)[0][0]
    )
