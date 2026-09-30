import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test_unused")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


api = load("rag_api", "api/app.py")
evaluation = load("retrieval_eval", "eval/evaluate.py")
ocr_eval = load("ocr_eval", "eval/evaluate_ocr.py")


def test_evidence_must_be_in_correct_source_and_page():
    case = {"expect_source": "circular.pdf", "expect_page": 2, "expect_text": "तीस दिन"}
    assert evaluation.hit({"source": "/data/circular.pdf", "page": 2, "text": "तीस\nदिन"}, case)
    assert not evaluation.hit({"source": "/data/circular.pdf", "page": 2, "text": "अन्य नियम"}, case)
    assert not evaluation.hit({"source": "/data/other.pdf", "page": 2, "text": "तीस दिन"}, case)
    assert not evaluation.hit({"source": "/data/circular.pdf", "page": 3, "text": "तीस दिन"}, case)


def test_alternative_phrases_and_legacy_source_only_labels():
    assert evaluation.hit({"text": "Tier 1 capital", "source": "a"}, {"expect_any": ["Tier I", "Tier 1"]})
    assert evaluation.hit({"text": "", "source": "/data/a.pdf"}, {"expect_source": "a.pdf"})


@pytest.mark.parametrize("cases", [[], [{"template": True}], [{"question": "q"}],
    [{"question": "q", "expect_any": [""]}], [{"question": "q", "expect_text": "x", "expect_page": 0}]])
def test_invalid_or_placeholder_benchmarks_are_rejected(cases):
    with pytest.raises(ValueError):
        evaluation.validate_cases(cases)


def test_ocr_unicode_and_real_errors():
    # Canonically equivalent nukta forms must not produce false OCR errors.
    assert ocr_eval.error_counts("क़र्ज़\n सीमा", "क़र्ज़ सीमा")["cer"] == 0
    assert ocr_eval.error_counts("कि", "की")["char_errors"] == 1
    assert ocr_eval.error_counts("तीस दिन", "तीस ऋण")["wer"] == 0.5
    assert ocr_eval.error_counts("तीस दिन", "")["wer"] == 1
    assert ocr_eval.error_counts("दिन", "दिन दिन दिन")["wer"] == 2
    with pytest.raises(ValueError):
        ocr_eval.error_counts(" ", "text")


def test_language_metrics_use_each_questions_first_relevant_rank(monkeypatch):
    def handler(request):
        query = json.loads(request.content)["query"]
        texts = ["wrong", "तीस दिन"] if query == "हिन्दी" else ["wrong"]
        return httpx.Response(200, json={"results": [{"text": t, "source": "a.pdf"} for t in texts]})
    real_client = httpx.Client
    monkeypatch.setattr(evaluation.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    result = evaluation.score("http://test", [
        {"question": "हिन्दी", "language": "hi", "expect_text": "तीस दिन"},
        {"question": "kitne din", "language": "hinglish", "expect_text": "तीस दिन"},
    ], 6, True)
    assert result["recall"] == 0.5
    assert result["mrr"] == 0.25
    assert result["by_language"]["hi"]["mrr"] == 0.5
    assert result["by_language"]["hinglish"]["recall"] == 0
    assert result["p95_ms"] >= 0


@pytest.mark.parametrize("language, query, rule", [
    ("hi", "समय सीमा क्या है?", "Devanagari"),
    ("hinglish", "Samay seema kya hai?", "Latin script"),
    ("en", "What is the deadline?", "English"),
])
def test_chat_preserves_query_citations_and_response_language(monkeypatch, language, query, rule):
    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            assert q == query
            return [{"source": "/data/परिपत्र.pdf", "page": 1, "text": "तीस दिन"}]

        def ollama(request):
            payload = json.loads(request.content)
            assert rule in payload["messages"][0]["content"]
            assert query in payload["messages"][1]["content"]
            assert "[1] (परिपत्र.pdf p.1)" in payload["messages"][1]["content"]
            return httpx.Response(200, text=json.dumps({"message": {"content": "तीस दिन [1]"}, "done": True}) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                response = await client.post("/api/chat", json={"query": query, "language": language})
                assert response.status_code == 200
                assert "event: sources" in response.text
                tokens = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                assert "तीस दिन [1]" in tokens
    asyncio.run(run())


def test_empty_retrieval_does_not_call_model_and_language_is_validated(monkeypatch):
    async def run():
        async def retrieve(*args):
            return []
        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        monkeypatch.setattr(api, "state", {})  # no model client available
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "kya hai?"}], "language": "hinglish"})
            assert response.status_code == 200
            assert "jaankari nahi mili" in response.json()["choices"][0]["message"]["content"]
            for lang in ("invalid", []):
                response = await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "q"}], "language": lang})
                assert response.status_code == 422
            response = await client.post("/api/chat", json={"query": "q", "language": "invalid"})
            assert response.status_code == 422
    asyncio.run(run())


def test_hindi_filenames_remain_distinct():
    assert api.SAFE_NAME.sub("_", "परिपत्र.pdf") == "परिपत्र.pdf"
    assert api.SAFE_NAME.sub("_", "नियम.pdf") != api.SAFE_NAME.sub("_", "ऋण.pdf")


# ---------------------------------------------------------------- areas

def test_area_names_are_cleaned_and_cannot_escape_the_corpus():
    assert api.clean_area(None) is None
    assert api.clean_area("") is None
    assert api.clean_area("all") is None
    assert api.clean_area("Income Tax") == "income_tax"
    assert api.clean_area("companies-act") == "companies_act"
    assert api.clean_area("../../etc") == "etc"
    assert api.clean_area("../") is None


def test_search_and_chat_pass_the_area_through(monkeypatch):
    seen = []

    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            seen.append(area)
            return []

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            await client.post("/api/search", json={"query": "q", "area": "gst"})
            await client.post("/api/search", json={"query": "q"})
            await client.post("/api/chat", json={"query": "q", "area": "income_tax"})
    asyncio.run(run())
    assert seen == ["gst", None, "income_tax"]


def test_only_finished_or_failed_jobs_can_be_dismissed(monkeypatch):
    jobs = {1: "error", 2: "parsing"}

    class Cur:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def execute(self, sql, args):
            self.row = None
            jid = args[0]
            if jobs.get(jid) in ("error", "done"):
                del jobs[jid]
                self.row = (jid,)
        async def fetchone(self): return self.row

    class Conn:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        def cursor(self): return Cur()

    class Pool:
        def connection(self): return Conn()

    async def run():
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        monkeypatch.setitem(api.state, "pool", Pool())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            assert (await client.delete("/api/jobs/1")).status_code == 200
            assert (await client.delete("/api/jobs/2")).status_code == 409
    asyncio.run(run())
    assert jobs == {2: "parsing"}


# ---------------------------------------------------------------- prompt versions

def test_v2_replaces_only_the_concise_rule():
    v1, v2 = api.answer_prompt("auto", "v1"), api.answer_prompt("auto", "v2")
    assert "Be concise" in v1 and "Be concise" not in v2
    assert "Never drop one" in v2
    for shared in ("Cite them inline", "do not guess", "Preserve the original numbers"):
        assert shared in v1 and shared in v2


def test_chat_uses_the_requested_prompt_version(monkeypatch):
    seen = []

    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            seen.append(json.loads(request.content)["messages"][0]["content"])
            return httpx.Response(200, text=json.dumps({"message": {"content": "ok"}, "done": True}) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                await client.post("/api/chat", json={"query": "q", "prompt_version": "v2"})
                await client.post("/api/chat", json={"query": "q"})
                bad = await client.post("/api/chat", json={"query": "q", "prompt_version": "v9"})
                assert bad.status_code == 422
    asyncio.run(run())
    assert "Never drop one" in seen[0]
    assert "Be concise" in seen[1]


def test_v2_does_not_open_unanswerable_questions_with_yes_or_no():
    v2 = api.answer_prompt("auto", "v2")
    assert "say so in the first sentence" in v2
    assert 'Never\n  open with "yes" or "no"' in v2


# ---------------------------------------------------------------- thinking and length cap

def test_think_setting_and_length_cap_reach_the_model(monkeypatch):
    payloads = []

    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, text=json.dumps({"message": {"content": "ok"}, "done": True}) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                await client.post("/api/chat", json={"query": "q", "think": False})
                await client.post("/api/chat", json={"query": "q", "think": True})
                await client.post("/api/chat", json={"query": "q"})
    asyncio.run(run())
    assert [p["think"] for p in payloads] == [False, True, api.LLM_THINK]
    assert all(p["options"]["num_predict"] == api.LLM_MAX_TOKENS for p in payloads)


def test_answer_cut_by_length_limit_says_so(monkeypatch):
    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            lines = [{"message": {"content": "partial answer"}, "done": False},
                     {"message": {"content": ""}, "done": True, "done_reason": "length"}]
            return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                r = await client.post("/api/chat", json={"query": "q"})
                tokens = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
                assert "partial answer" in tokens
                assert any("cut short" in str(t) for t in tokens)
    asyncio.run(run())
