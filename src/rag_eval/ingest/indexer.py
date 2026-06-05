"""Build and persist the dense + sparse indexes for a config.

Index artifacts are keyed by a signature over the parts of the config that change the index
(corpus + chunking + embedder), so ablation variants that differ only in, say, ``rerank`` reuse
the same index, while a chunk-size ablation builds its own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rag_eval.config import Config
from rag_eval.ingest.chunking import chunk_documents
from rag_eval.ingest.loaders import load_corpus
from rag_eval.retrieve.bm25 import BM25Index
from rag_eval.retrieve.dense import InMemoryVectorStore, VectorStore, get_vector_store
from rag_eval.retrieve.embedder import Embedder, get_embedder
from rag_eval.types import Chunk

DEFAULT_INDEX_ROOT = Path("data/index")


@dataclass(slots=True)
class IndexBundle:
    """Everything the query path needs: chunks, both indexes, and the embedder."""

    chunks: list[Chunk]
    chunks_by_id: dict[str, Chunk]
    vector_store: VectorStore
    bm25: BM25Index
    embedder: Embedder


def index_signature(config: Config) -> str:
    """Stable hash of the config parts that determine the index contents."""
    payload = json.dumps(
        {
            "corpus": config.corpus.model_dump(),
            "chunking": config.chunking.model_dump(),
            "embedder": config.embedder.model_dump(),
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def index_dir(config: Config, root: Path | None = None) -> Path:
    return (root or DEFAULT_INDEX_ROOT) / index_signature(config)


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    lines = [
        json.dumps(
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "text": c.text,
                "ordinal": c.ordinal,
                "title": c.title,
                "metadata": c.metadata,
            }
        )
        for c in chunks
    ]
    path.write_text("\n".join(lines))


def _read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        chunks.append(
            Chunk(
                id=r["id"],
                doc_id=r["doc_id"],
                text=r["text"],
                ordinal=int(r["ordinal"]),
                title=r.get("title", ""),
                metadata=dict(r.get("metadata", {})),
            )
        )
    return chunks


def build_index(config: Config, persist: bool = True, root: Path | None = None) -> IndexBundle:
    """Build dense + BM25 indexes from the corpus; optionally persist them to disk."""
    docs = load_corpus(config.corpus.path)
    chunks = chunk_documents(docs, config.chunking)
    if not chunks:
        raise ValueError("corpus produced no chunks; check chunking config")

    ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]

    embedder = get_embedder(config.embedder)
    store = get_vector_store(config.vector_store)
    store.add(ids, embedder.embed(texts))
    bm25 = BM25Index.build(ids, texts)

    bundle = IndexBundle(chunks, {c.id: c for c in chunks}, store, bm25, embedder)
    if persist:
        save_index(config, bundle, root=root)
    return bundle


def save_index(config: Config, bundle: IndexBundle, root: Path | None = None) -> Path:
    """Persist the index artifacts; returns the directory written to."""
    out = index_dir(config, root)
    out.mkdir(parents=True, exist_ok=True)
    _write_chunks(out / "chunks.jsonl", bundle.chunks)
    bundle.bm25.save(out)
    if isinstance(bundle.vector_store, InMemoryVectorStore):
        bundle.vector_store.save(out)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "signature": index_signature(config),
                "config_name": config.name,
                "num_chunks": len(bundle.chunks),
                "embedder": config.embedder.backend,
                "vector_store": config.vector_store.backend,
            },
            indent=2,
        )
    )
    return out


def load_index(config: Config, root: Path | None = None) -> IndexBundle:
    """Load a previously built index for this config."""
    src = index_dir(config, root)
    if not (src / "meta.json").exists():
        raise FileNotFoundError(f"no index at {src}; run `make ingest` first")

    chunks = _read_chunks(src / "chunks.jsonl")
    bm25 = BM25Index.load(src)
    embedder = get_embedder(config.embedder)
    if config.vector_store.backend != "memory":
        raise NotImplementedError("loading is only implemented for the in-memory vector store")
    store: VectorStore = InMemoryVectorStore.load(src)
    return IndexBundle(chunks, {c.id: c for c in chunks}, store, bm25, embedder)
