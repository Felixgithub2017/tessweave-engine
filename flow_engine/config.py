from dataclasses import dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_dict(cls, d):
        # Unsupported architecture semantics must fail BEFORE loading weights.
        if d.get("model_type") not in {"llama", "qwen2"}:
            raise ValueError("Supported: dense llama/qwen2 with full attention and default RoPE only")
        if d.get("quantization_config"):
            raise ValueError("Quantized checkpoints are not supported by the dense loader")
        if d.get("rope_scaling") and d["rope_scaling"].get("rope_type", d["rope_scaling"].get("type")) != "default":
            raise ValueError("Scaled RoPE requires a dedicated implementation; refusing silent approximation")
        if d.get("use_sliding_window") or (d.get("model_type") == "llama" and d.get("sliding_window")):
            raise ValueError("Sliding-window attention is not implemented")
        if d.get("hidden_act", "silu") != "silu" or d.get("mlp_bias"):
            raise ValueError("Only unbiased SwiGLU MLP is supported")
        if d.get("num_local_experts") or d.get("num_experts") or d.get("auto_map"):
            raise ValueError("MoE and custom remote-code checkpoints are not supported")
        h, nh = int(d["hidden_size"]), int(d["num_attention_heads"])
        nk = int(d.get("num_key_value_heads", nh))
        hd = int(d.get("head_dim", h // nh))
        if min(h, nh, nk, hd) <= 0 or nh % nk or h != nh * hd or hd % 2:
            raise ValueError("Invalid or unsupported attention dimensions")
        eos = d.get("eos_token_id", [])
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else eos)
        cfg = cls(d["model_type"], int(d["vocab_size"]), h, int(d["intermediate_size"]),
                  int(d["num_hidden_layers"]), nh, nk, hd, float(d.get("rms_norm_eps", 1e-6)),
                  float(d.get("rope_theta", 10000)), int(d["max_position_embeddings"]),
                  bool(d.get("tie_word_embeddings", False)),
                  True if d["model_type"] == "qwen2" else bool(d.get("attention_bias", False)), tuple(eos))
        if min(cfg.vocab_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.max_position_embeddings) <= 0:
            raise ValueError("Invalid model dimensions")
        return cfg

    @classmethod
    def read(cls, path):
        config = cls.from_dict(json.loads((Path(path) / "config.json").read_text()))
        generation = Path(path) / "generation_config.json"
        if generation.exists():
            eos = json.loads(generation.read_text()).get("eos_token_id")
            if eos is not None:
                config = replace(config, eos_token_ids=tuple([eos] if isinstance(eos, int) else eos))
        return config

    def kv_bytes_per_token(self, element_bytes):
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * element_bytes


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    kv_blocks: int = 256
    max_context: int = 2048
    max_active: int = 4
    max_queue: int = 32
    prefill_chunk: int = 256
    min_prefill_chunk: int = 16
    target_step_ms: float = 50.0
    prefix_cache: bool = True
    attention: str = "sdpa"
    draft_tokens: int = 0
    draft_window: int = 512

    def __post_init__(self):
        for key in ("block_size", "kv_blocks", "max_context", "max_active", "max_queue", "prefill_chunk", "min_prefill_chunk"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.prefill_chunk < self.min_prefill_chunk or self.target_step_ms <= 0:
            raise ValueError("Invalid prefill control bounds")
        if self.attention not in {"sdpa", "triton"}:
            raise ValueError("attention must be sdpa or triton")
        if not 0 <= self.draft_tokens <= 8 or self.draft_window < 8:
            raise ValueError("draft_tokens must be 0..8; draft_window must be at least 8")
