import asyncio
from contextlib import asynccontextmanager
import json
import queue
import time
from typing import Literal
from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from .cache import CapacityError


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=262144)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    stream: bool = False
    stream_options: dict | None = None


class TextStream:
    """Word/Unicode-safe incremental detokenization; terminal flush is mandatory."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.emitted = ""

    def push(self, token=None, final=False):
        if token is not None:
            self.ids.append(token)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not text.startswith(self.emitted):
            raise RuntimeError("Tokenizer revised previously emitted text; unsupported incremental decoding")
        if final:
            end = len(text)
        elif text.endswith("\n") or (text and "\u4e00" <= text[-1] <= "\u9fff"):
            end = len(text)
        else:
            # Do not emit an incomplete byte token or an unstable word suffix.
            end = text.rfind(" ") + 1
            bad = text.find("\ufffd", len(self.emitted))
            if bad >= 0:
                end = min(end, bad)
        end = max(len(self.emitted), end)
        delta = text[len(self.emitted):end]
        self.emitted = text[:end]
        return delta


def create_app(engine, tokenizer, served_name="flow-model"):
    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        # A successfully loaded weight file alone is not a readiness probe.
        warm = await asyncio.to_thread(engine.submit, [0], 1)
        while True:
            event = await asyncio.to_thread(warm.events.get)
            if event["type"] == "done":
                if event["error"]:
                    engine.close()
                    raise RuntimeError(event["error"])
                break
        engine.trace.emit("ready", model=served_name)
        try:
            yield
        finally:
            await asyncio.to_thread(engine.close)

    app = FastAPI(title="Flow Inference", version="0.1.0a1", lifespan=lifespan)

    @app.middleware("http")
    async def limit_body(request, call_next):
        if request.method == "POST":
            size = 0
            parts = []
            async for piece in request.stream():
                size += len(piece)
                if size > 1024 * 1024:
                    return JSONResponse({"error": "Request body exceeds 1 MiB"}, status_code=413)
                parts.append(piece)
            request._body = b"".join(parts)
        return await call_next(request)

    @app.get("/health")
    async def health():
        stats = await asyncio.to_thread(engine.stats)
        return JSONResponse(stats, status_code=200 if stats["ready"] else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.get("/metrics")
    async def metrics():
        stats = await asyncio.to_thread(engine.stats)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(f"flow_{k} {float(v)}" for k, v in stats.items()
                                           if isinstance(v, (int, float, bool))) + "\n")

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: HTTPRequest):
        if body.model != served_name:
            raise HTTPException(404, "Unknown model; inspect /v1/models")
        if body.stream_options is not None and (set(body.stream_options) - {"include_usage"}):
            raise HTTPException(422, "Only stream_options.include_usage is supported")
        try:
            ids = await asyncio.to_thread(tokenizer.apply_chat_template, [m.model_dump() for m in body.messages],
                                          tokenize=True, add_generation_prompt=True)
            r = await asyncio.to_thread(engine.submit, ids, body.max_tokens, body.temperature, body.seed)
        except CapacityError as exc:
            raise HTTPException(429, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        created = int(time.time())
        base = {"id": f"chatcmpl-{r.id}", "created": created, "model": served_name}

        async def events():
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        e = await asyncio.to_thread(r.events.get, True, .5)
                    except queue.Empty:
                        continue
                    yield e
                    if e["type"] == "done":
                        break
            finally:
                await asyncio.to_thread(engine.cancel, r.id)

        if body.stream:
            async def stream():
                decoder = TextStream(tokenizer)
                def chunk(delta, finish=None, usage=None):
                    result = {**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage is not None:
                        result["usage"] = usage
                    return "data: " + json.dumps(result, ensure_ascii=False) + "\n\n"
                try:
                    yield chunk({"role": "assistant", "content": ""})
                    async for e in events():
                        if e["type"] == "token":
                            text = decoder.push(e["token_id"])
                            if text:
                                yield chunk({"content": text})
                        elif e["error"]:
                            yield "data: " + json.dumps({"error": e["error"]}) + "\n\n"
                        else:
                            tail = decoder.push(final=True)
                            if tail:
                                yield chunk({"content": tail})
                            yield chunk({}, e["finish_reason"], e["usage"])
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    await asyncio.to_thread(engine.cancel, r.id)
            return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        output = []
        async for e in events():
            if e["type"] == "token":
                output.append(e["token_id"])
            else:
                if e["error"]:
                    raise HTTPException(500, e["error"])
                text = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return {**base, "object": "chat.completion", "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": text}, "finish_reason": e["finish_reason"]}], "usage": e["usage"]}
        raise HTTPException(499, "Client disconnected")

    return app
