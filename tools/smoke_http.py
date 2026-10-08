"""Real running-service smoke check, including SSE/JSON equality and usage."""
import argparse
import json
from pathlib import Path
from urllib.request import build_opener, ProxyHandler, Request
from benchmark_http import request_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8010")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("Use a fresh output path")
    opener = build_opener(ProxyHandler({}))
    def get(path):
        with opener.open(args.base_url + path, timeout=60) as r:
            return json.load(r)
    before = get("/health")
    assert before["ready"]
    name = get("/v1/models")["data"][0]["id"]
    messages = [{"role": "user", "content": "用一句话解释 KV Cache。"}]
    data = json.dumps({"model": name, "messages": messages, "max_tokens": 16, "temperature": 0}).encode()
    with opener.open(Request(args.base_url + "/v1/chat/completions", data=data,
                             headers={"Content-Type": "application/json"}), timeout=60) as r:
        normal = json.load(r)
    streamed = request_one(args.base_url + "/v1", name, messages, 16, 60)
    assert normal["choices"][0]["message"]["content"] == streamed["text"]
    assert normal["usage"] == streamed["usage"]
    after = get("/health")
    assert after["active"] == after["waiting"] == 0
    report = {"ready": True, "model": name, "json_equals_sse": True, "usage_equal": True,
              "response": streamed["text"], "usage": streamed["usage"],
              "first_text_s": streamed["first_text_s"], "request_elapsed_s": streamed["elapsed_s"],
              "stats_after": after, "scope": "Local HTTP smoke, not a cross-engine benchmark"}
    with Path(args.output).open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
