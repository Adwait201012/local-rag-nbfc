import re
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


# ---------------------------------------------------------------- thinking status

def test_stream_announces_thinking_once_before_the_answer(monkeypatch):
    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            lines = [{"message": {"thinking": "Let me check", "content": ""}, "done": False},
                     {"message": {"thinking": " the passages", "content": ""}, "done": False},
                     {"message": {"content": "The answer"}, "done": False},
                     {"message": {"content": " is 15%."}, "done": True, "done_reason": "stop"}]
            return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                r = await client.post("/api/chat", json={"query": "q"})
        events = [l[7:] for l in r.text.splitlines() if l.startswith("event: ")]
        return events, r.text
    events, text = asyncio.run(run())
    assert events.count("status") == 1
    assert events.index("status") < events.index("token")
    assert events[0] == "sources" and events[-1] == "done"
    assert "Let me check" not in text          # the reasoning itself is never shown


# ---------------------------------------------------------------- language detection

def test_hinglish_is_detected_without_mistaking_english():
    assert api.detect_language("Public deposit ka period kitna hona chahiye?") == "hinglish"
    assert api.detect_language("Kya depositor teen mahine se pehle paisa nikal sakta hai?") == "hinglish"
    assert api.detect_language("सार्वजनिक जमा की अवधि कितनी होनी चाहिए?") == "hi"
    assert api.detect_language("What is the minimum CRAR for a Middle Layer NBFC?") == "en"
    assert api.detect_language("Can a director's relative get a ₹6 lakh loan at par?") == "en"
    assert api.resolve_language("Public deposit ka period kitna hai?", "en") == "en"   # explicit choice wins


def test_auto_answers_detected_hinglish_in_english(monkeypatch):
    prompts = []

    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            req = json.loads(request.content)
            if req["stream"] is False:          # the question-translation step
                return httpx.Response(200, json={"message": {"content": "What is the deposit period?"}})
            prompts.append(req["messages"][0]["content"])
            return httpx.Response(200, text=json.dumps({"message": {"content": "ok"}, "done": True}) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                await client.post("/api/chat", json={"query": "Public deposit ka period kitna hona chahiye?"})
                await client.post("/api/chat", json={"query": "What period must a public deposit have?"})
                await client.post("/api/chat", json={"query": "Public deposit ka period kitna hona chahiye?",
                                                     "language": "hinglish"})
                await client.post("/api/chat", json={"query": "सार्वजनिक जमा की अवधि कितनी होनी चाहिए?"})
    asyncio.run(run())
    assert prompts[0].endswith(api.LANGUAGE_RULES["en"])        # detected Hinglish -> English answer
    assert prompts[1].endswith(api.LANGUAGE_RULES["en"])
    assert prompts[2].endswith(api.LANGUAGE_RULES["hinglish"])  # explicit choice still honoured
    assert prompts[3].endswith(api.LANGUAGE_RULES["en"])        # detected Hindi -> English answer


def test_no_match_message_stays_in_hinglish():
    assert "jaankari nahi mili" in api.no_match("Public deposit ka period kitna hona chahiye?", "auto")



def _stream(monkeypatch, body):
    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            lines = [{"message": {"thinking": "checking", "content": ""}, "done": False},
                     {"message": {"content": "12 to 60 months [1]."}, "done": True, "done_reason": "stop"}]
            return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                r = await client.post("/api/chat", json=body)
        events, data, cur = [], [], None
        for line in r.text.splitlines():
            if line.startswith("event: "):
                cur = line[7:]
            elif line.startswith("data: "):
                events.append(cur); data.append(json.loads(line[6:]))
        return events, data
    return asyncio.run(run())


def test_hindi_question_gets_fixed_note_then_english_answer(monkeypatch):
    events, data = _stream(monkeypatch, {"query": "सार्वजनिक जमा की अवधि कितनी होनी चाहिए?"})
    tokens = [d for e, d in zip(events, data) if e == "token"]
    assert tokens[0] == api.ENGLISH_ON_PURPOSE["hi"]
    assert tokens[1] == "12 to 60 months [1]."
    # the thinking status arrives before the note, so the page can still show it
    assert events.index("status") < events.index("token")


def test_hinglish_question_gets_hinglish_note(monkeypatch):
    events, data = _stream(monkeypatch, {"query": "Public deposit ka period kitna hona chahiye?"})
    tokens = [d for e, d in zip(events, data) if e == "token"]
    assert tokens[0] == api.ENGLISH_ON_PURPOSE["hinglish"]


def test_english_question_gets_no_note(monkeypatch):
    events, data = _stream(monkeypatch, {"query": "What period must a public deposit have?"})
    tokens = [d for e, d in zip(events, data) if e == "token"]
    assert tokens == ["12 to 60 months [1]."]


def test_explicit_hindi_gets_translation_warning(monkeypatch):
    events, data = _stream(monkeypatch, {"query": "What period must a public deposit have?", "language": "hi"})
    tokens = [d for e, d in zip(events, data) if e == "token"]
    assert tokens[-1] == api.VERIFY_TRANSLATION["hi"]
    assert api.ENGLISH_ON_PURPOSE["hi"] not in tokens


def _stream_pieces(monkeypatch, body, pieces):
    """Run /api/chat with a fake model that streams the given answer pieces."""
    seen = {}

    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            seen["messages"] = json.loads(request.content)["messages"]
            lines = [{"message": {"thinking": "checking", "content": ""}, "done": False}]
            lines += [{"message": {"content": p}, "done": False} for p in pieces]
            lines += [{"message": {"content": ""}, "done": True, "done_reason": "stop"}]
            return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                r = await client.post("/api/chat", json=body)
        cur, tokens = None, []
        for line in r.text.splitlines():
            if line.startswith("event: "):
                cur = line[7:]
            elif line.startswith("data: ") and cur == "token":
                tokens.append(json.loads(line[6:]))
        return tokens
    return asyncio.run(run()), seen


HINDI_Q = "सार्वजनिक जमा की अवधि कितनी होनी चाहिए?"


def test_english_instruction_is_repeated_beside_a_hindi_question(monkeypatch):
    _, seen = _stream_pieces(monkeypatch, {"query": HINDI_Q}, ["12 to 60 months [1]."])
    assert seen["messages"][-1]["content"].endswith("even though the question is in Hindi.")
    _, seen = _stream_pieces(monkeypatch, {"query": "What period must a public deposit have?"}, ["ok"])
    assert "English only" not in seen["messages"][-1]["content"]


def test_note_is_dropped_when_the_model_answers_in_hindi_anyway(monkeypatch):
    hindi = ["सार्वजनिक जमा की अवधि ", "कम से कम 12 महीने ", "और अधिकतम 60 महीने होनी चाहिए [1]।"]
    tokens, _ = _stream_pieces(monkeypatch, {"query": HINDI_Q}, hindi)
    assert api.ENGLISH_ON_PURPOSE["hi"] not in tokens          # never claims English falsely
    assert tokens[-1] == api.VERIFY_TRANSLATION["hi"]          # warns instead
    assert "".join(tokens[:-1]) == "".join(hindi)              # answer text intact


def test_long_english_answer_gets_note_first_and_text_intact(monkeypatch):
    pieces = ["A public deposit ", "must be repayable ", "after twelve months ", "but not later than ",
              "sixty months from ", "acceptance [1]."]
    tokens, _ = _stream_pieces(monkeypatch, {"query": HINDI_Q}, pieces)
    assert tokens[0] == api.ENGLISH_ON_PURPOSE["hi"]
    assert "".join(tokens[1:]) == "".join(pieces)
    assert api.VERIFY_TRANSLATION["hi"] not in tokens


def _stream_translated(monkeypatch, body, translation, answer="A public deposit must run 12 to 60 months [1]."):
    """Fake model: answers the translation call with `translation`, streams `answer` otherwise."""
    calls = []

    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            req = json.loads(request.content)
            calls.append(req)
            if req["stream"] is False:
                if translation is None:
                    return httpx.Response(500, text="boom")
                return httpx.Response(200, json={"message": {"content": translation}})
            lines = [{"message": {"thinking": "x", "content": ""}, "done": False},
                     {"message": {"content": answer}, "done": True, "done_reason": "stop"}]
            return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                r = await client.post("/api/chat", json=body)
        cur, tokens = None, []
        for line in r.text.splitlines():
            if line.startswith("event: "):
                cur = line[7:]
            elif line.startswith("data: ") and cur == "token":
                tokens.append(json.loads(line[6:]))
        return tokens
    return asyncio.run(run()), calls


def test_hindi_question_is_translated_and_model_never_sees_hindi(monkeypatch):
    tokens, calls = _stream_translated(monkeypatch, {"query": HINDI_Q},
                                       "What should the period of a public deposit be?")
    translate, answer = calls
    assert translate["think"] is False and "Translate" in translate["messages"][0]["content"]
    user_msg = answer["messages"][-1]["content"]
    assert "Question: What should the period of a public deposit be?" in user_msg
    assert not re.search(r"[\u0900-\u097f]", user_msg)          # no Hindi reaches the answering model
    assert tokens[0].startswith(api.ENGLISH_ON_PURPOSE["hi"])
    assert "Question understood as: What should the period of a public deposit be?" in tokens[0]


def test_hinglish_question_is_translated_too(monkeypatch):
    tokens, calls = _stream_translated(monkeypatch, {"query": "Public deposit ka period kitna hona chahiye?"},
                                       "What should the period of a public deposit be?")
    assert "Question: What should the period" in calls[1]["messages"][-1]["content"]
    assert "Question understood as:" in tokens[0]


def test_failed_translation_falls_back_to_the_instruction(monkeypatch):
    tokens, calls = _stream_translated(monkeypatch, {"query": HINDI_Q}, None)
    assert calls[-1]["messages"][-1]["content"].endswith("even though the question is in Hindi.")
    assert "understood as" not in tokens[0]


def test_translation_still_in_hindi_is_rejected(monkeypatch):
    tokens, calls = _stream_translated(monkeypatch, {"query": HINDI_Q}, "सार्वजनिक जमा की अवधि?")
    assert calls[-1]["messages"][-1]["content"].endswith("even though the question is in Hindi.")


def test_english_question_is_not_translated(monkeypatch):
    tokens, calls = _stream_translated(monkeypatch, {"query": "What period must a public deposit have?"}, "x")
    assert len(calls) == 1                                      # no translation call at all



def test_v3_is_v2_plus_the_source_law_rule():
    v2, v3 = api.answer_prompt("auto", "v2"), api.answer_prompt("auto", "v3")
    assert "Never apply a rule, rate, limit or figure from one regulation" in v3
    assert "Never apply a rule" not in v2
    assert v3.replace(api.SOURCE_LAW_RULE + "\n", "") == v2      # nothing else differs


# ---------------------------------------------------------------- token budget

def _length_stop(monkeypatch, query, pieces):
    async def run():
        async def retrieve(q, top_k, use_rerank=True, area=None):
            return [{"source": "/data/x.pdf", "page": 1, "text": "rule"}]

        def ollama(request):
            req = json.loads(request.content)
            if req["stream"] is False:
                return httpx.Response(200, json={"message": {"content": "What is the rule?"}})
            lines = [{"message": {"thinking": "long reasoning", "content": ""}, "done": False}]
            lines += [{"message": {"content": p}, "done": False} for p in pieces]
            lines += [{"message": {"content": ""}, "done": True, "done_reason": "length"}]
            return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines) + "\n")

        monkeypatch.setattr(api, "retrieve", retrieve)
        monkeypatch.setattr(api, "APP_PASSWORD", "")
        async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as llm:
            monkeypatch.setitem(api.state, "http", llm)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                r = await client.post("/api/chat", json={"query": query})
        return [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    return asyncio.run(run())


def test_budget_spent_on_reasoning_says_no_answer_was_written(monkeypatch):
    for q in ("What period must a public deposit have?", "सार्वजनिक जमा की अवधि कितनी होनी चाहिए?"):
        data = _length_stop(monkeypatch, q, [])
        assert api.NO_ANSWER in data and api.CUT_SHORT not in data


def test_budget_hit_mid_answer_says_cut_short(monkeypatch):
    data = _length_stop(monkeypatch, "What period must a public deposit have?", ["Twelve to sixty months"])
    assert api.CUT_SHORT in data and api.NO_ANSWER not in data


def test_default_budget_is_6144():
    assert api.LLM_MAX_TOKENS == 6144
