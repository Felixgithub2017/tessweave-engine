import json
import unittest
from fastapi.testclient import TestClient
from flow_engine.server import create_app, TextStream
from test_engine import fixture, engine_for


class Tokenizer:
    # Protocol fixture, not a model implementation. Real tokenizer tested by
    # tools/validate_model.py using a user's local checkpoint.
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [3, 4, 5, 6, 7]

    def decode(self, ids, **kwargs):
        return "".join(f"t{i} " for i in ids)


class ServerTests(unittest.TestCase):
    def setUp(self):
        _, model = fixture()
        self.engine = engine_for(model)
        self.client = TestClient(create_app(self.engine, Tokenizer()))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def payload(self, **kw):
        return {"model": "flow-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4, **kw}

    def test_readiness_models_metrics(self):
        self.assertTrue(self.client.get("/health").json()["ready"])
        self.assertEqual(self.client.get("/v1/models").json()["data"][0]["id"], "flow-model")
        self.assertIn("flow_generated_tokens", self.client.get("/metrics").text)

    def test_nonstream_and_sse_equal_with_usage(self):
        base = self.client.post("/v1/chat/completions", json=self.payload()).json()
        streamed = self.client.post("/v1/chat/completions", json=self.payload(stream=True))
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: {")]
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        self.assertEqual(text, base["choices"][0]["message"]["content"])
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 4)
        self.assertIn("data: [DONE]", streamed.text)

    def test_unknown_options_and_multimodal_rejected(self):
        self.assertEqual(self.client.post("/v1/chat/completions", json=self.payload(top_p=.9)).status_code, 422)
        self.assertEqual(self.client.post("/v1/chat/completions", json=self.payload(model="wrong")).status_code, 404)
        p = self.payload(messages=[{"role": "user", "content": [{"type": "image_url"}]}])
        self.assertEqual(self.client.post("/v1/chat/completions", json=p).status_code, 422)

    def test_context_and_body_caps(self):
        self.assertEqual(self.client.post("/v1/chat/completions", json=self.payload(max_tokens=64)).status_code, 400)
        self.assertEqual(self.client.post("/v1/chat/completions", content=b"x" * (1048576 + 1)).status_code, 413)

    def test_final_flush(self):
        class Bytes:
            def decode(self, ids, **kwargs):
                return "hello" if len(ids) == 1 else "hello world"
        s = TextStream(Bytes())
        self.assertEqual(s.push(1), "")
        self.assertEqual(s.push(2), "hello ")
        self.assertEqual(s.push(final=True), "world")


if __name__ == "__main__":
    unittest.main()
