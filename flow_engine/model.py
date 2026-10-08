"""Independent Llama/Qwen2 forward path; no AutoModel.generate or serving engine.

Only tokenizer/config formats come from Hugging Face. Real weights are loaded
from safetensors without pickle or execution of repository Python code.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        f = x.float()
        return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rotary(x, positions, theta):
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    freq = positions.float()[..., None] * inv
    freq = torch.cat((freq, freq), -1)[:, :, None, :]
    half = d // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * freq.cos().to(x.dtype) + rotated * freq.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=c.attention_bias)
        # Qwen2 biases Q/K/V but not O.
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attention_bias and c.model_type == "llama")

    def pack(self):
        if hasattr(self, "qkv_proj"):
            return
        self.qkv_sizes = [self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features]
        self.qkv_proj = pack_linears([self.q_proj, self.k_proj, self.v_proj])
        del self.q_proj, self.k_proj, self.v_proj

    def project(self, h):
        if hasattr(self, "qkv_proj"):
            return self.qkv_proj(h).split(self.qkv_sizes, dim=-1)
        return self.q_proj(h), self.k_proj(h), self.v_proj(h)


def pack_linears(layers):
    """Exact row concatenation, not quantization or arithmetic approximation."""
    with torch.device("meta"):
        fused = nn.Linear(layers[0].in_features, sum(x.out_features for x in layers), bias=layers[0].bias is not None)
    fused.weight = nn.Parameter(torch.cat([x.weight.detach() for x in layers], 0), requires_grad=False)
    if layers[0].bias is not None:
        fused.bias = nn.Parameter(torch.cat([x.bias.detach() for x in layers], 0), requires_grad=False)
    return fused


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        if hasattr(self, "gate_up_proj"):
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def pack(self):
        if not hasattr(self, "gate_up_proj"):
            self.gate_up_proj = pack_linears([self.gate_proj, self.up_proj])
            del self.gate_proj, self.up_proj


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)


class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32, fuse_linears=True):
        path = Path(path).resolve(strict=True)
        cfg = ModelConfig.read(path)
        # Meta construction avoids a second randomly initialized model in RAM.
        with torch.device("meta"):
            model = cls(cfg)
        model.to(dtype=dtype)
        model.to_empty(device=device)
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        expected = dict(model.named_parameters())
        seen = set()
        index = path / "model.safetensors.index.json"
        if index.exists():
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            names = ["model.safetensors"]
        for name in names:
            # Index entries cannot escape the model directory. HF cache symlinks
            # for individual files are allowed; no code in them is executed.
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError("Unsafe or unsupported shard filename")
            with safe_open(str(path / name), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key == "lm_head.weight" and cfg.tie_word_embeddings:
                        continue
                    if key.endswith("rotary_emb.inv_freq"):
                        continue
                    if key not in expected or key in seen:
                        raise ValueError(f"Unexpected/duplicate weight: {key}")
                    tensor = shard.get_tensor(key)
                    if tensor.shape != expected[key].shape:
                        raise ValueError(f"Shape mismatch for {key}: {tensor.shape} != {expected[key].shape}")
                    with torch.no_grad():
                        expected[key].copy_(tensor)
                    seen.add(key)
        missing = set(expected) - seen
        if missing:
            raise ValueError(f"Missing required weights: {sorted(missing)[:12]}")
        # Release the old parameter mapping before packing so obsolete tensors
        # can be freed a layer at a time rather than retaining a full copy.
        del expected
        if fuse_linears:
            model.pack_projections()
        return model.eval()

    def pack_projections(self):
        for layer in self.model.layers:
            layer.self_attn.pack()
            layer.mlp.pack()
        return self

    @torch.inference_mode()
    def forward(self, ids, positions, tables, lengths, pool, attention="sdpa", all_logits=False):
        """Uniform chunks: [B,T]; decode batches have T=1, prefill B=1.

        positions are absolute; lengths include this step's tokens. Updating KV
        happens layer by layer. Explicit offset masks are essential: is_causal
        alone is wrong when query length differs from the accumulated KV length.
        """
        c = self.config
        x = self.model.embed_tokens(ids)
        b, t, _ = x.shape
        slots = [s for table, n in zip(tables, lengths) for s in pool.slots(table, n - t, t)]
        for li, layer in enumerate(self.model.layers):
            residual = x
            h = layer.input_layernorm(x)
            a = layer.self_attn
            q_raw, k_raw, v_raw = a.project(h)
            q = rotary(q_raw.reshape(b, t, c.num_attention_heads, c.head_dim), positions, c.rope_theta)
            k = rotary(k_raw.reshape(b, t, c.num_key_value_heads, c.head_dim), positions, c.rope_theta)
            v = v_raw.reshape(b, t, c.num_key_value_heads, c.head_dim)
            pool.write(li, slots, k.flatten(0, 1), v.flatten(0, 1))
            if attention == "triton" and t == 1:
                from .kernels.paged_decode import paged_decode
                out = paged_decode(q[:, 0], pool.k[li], pool.v[li], tables, lengths).unsqueeze(1)
            else:
                keys, values = pool.gather(li, tables, lengths)
                repeat = c.num_attention_heads // c.num_key_value_heads
                keys = keys.repeat_interleave(repeat, dim=2).transpose(1, 2)
                values = values.repeat_interleave(repeat, dim=2).transpose(1, 2)
                p = torch.arange(keys.shape[2], device=x.device)
                valid = (p[None, None, :] <= positions[:, :, None])
                valid &= p[None, None, :] < torch.tensor(lengths, device=x.device)[:, None, None]
                out = F.scaled_dot_product_attention(q.transpose(1, 2), keys, values,
                                                     attn_mask=valid[:, None], dropout_p=0).transpose(1, 2)
            x = residual + a.o_proj(out.reshape(b, t, c.hidden_size))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # Only the last position is needed for autoregressive sampling.
        return self.lm_head(self.model.norm(x if all_logits else x[:, -1])).float()
