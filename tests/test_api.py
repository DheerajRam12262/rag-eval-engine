"""FastAPI /health and /query, exercised offline via TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_eval.api.app import create_app
from rag_eval.ingest.indexer import IndexBundle


@pytest.fixture(scope="module")
def client(base_index: IndexBundle) -> TestClient:
    return TestClient(create_app(bundle=base_index))


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["num_chunks"] > 0


def test_query_answers_with_citations_and_telemetry(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "what are the two moons of mars called"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert body["citations"]
    assert body["contexts"]
    assert body["contexts"][0]["doc_id"]  # context surfaced
    assert body["latencies"]["total_ms"] >= 0.0
    assert "prompt_tokens" in body["usage"]


def test_query_abstains_on_no_answer(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "what is the price of bitcoin today"})
    assert resp.status_code == 200
    assert resp.json()["abstained"] is True


def test_query_top_k_override(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "how does photosynthesis work", "top_k": 2})
    assert resp.status_code == 200
    assert len(resp.json()["contexts"]) <= 2


def test_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}).status_code == 422
