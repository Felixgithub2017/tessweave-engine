"""Weight-free *qualification audit*, never a substitute for model inference.

Uses only the standard library and the native config gate. No remote code,
tokenizer, model object, weight file, proxy, credential or pip installer is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request

from .config import ModelConfig

CATALOG = "arena-text-2026-10-02.json"
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
LIMIT = 2 * 1024 * 1024
DIMENSIONS = {
    "model_type", "architectures", "hidden_size", "vocab_size", "intermediate_size",
    "moe_intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "qk_nope_head_dim", "qk_rope_head_dim",
    "v_head_dim", "kv_lora_rank", "q_lora_rank", "n_routed_experts",
    "n_shared_experts", "num_local_experts", "num_experts", "num_experts_per_tok",
    "num_experts_per_token", "num_selected_experts", "moe_topk", "top_k",
    "first_k_dense_replace", "moe_layer_freq", "layer_types", "sliding_window",
    "max_position_embeddings", "rope_theta", "rope_scaling", "rope_parameters",
    "torch_dtype", "dtype", "tie_word_embeddings", "hidden_act", "rms_norm_eps",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "full_attention_interval", "attention_type",
    "attention_types", "hybrid_layer_pattern", "partial_rotary_factor",
    "global_head_dim", "num_global_key_value_heads", "attention_k_eq_v",
    "num_kv_shared_layers", "hidden_activation", "final_logit_softcapping",
    "swa_head_dim", "swa_num_attention_heads", "swa_num_key_value_heads",
    "sliding_window_size", "local_layer_ids", "model_max_length", "use_sconv",
    "sconv_kernel_size", "hc_mult", "hc_sinkhorn_iters", "compress_ratios",
    "compress_rope_theta", "index_topk", "index_head_dim", "index_n_heads",
    "indexer_types", "expert_dtype", "num_nextn_predict_layers", "qk_norm",
    "scoring_func", "topk_method", "routed_scaling_factor", "num_shared_experts",
    "moe_router_use_sigmoid", "moe_router_activation_func", "route_scale",
    "moe_router_enable_expert_bias", "attn_res_block_size", "routed_expert_hidden_size",
    "full_attn_layers", "kda_layers", "num_heads", "short_conv_kernel_size",
    "hybrid_attention_ratio", "swa_num_attention_heads", "swa_num_key_value_heads",
    "attention_projection_size", "moe_renormalize", "enable_attention_fp32_softmax",
    "enable_lm_head_fp32", "enable_moe_fp32_combine", "route_norm",
}


def load_catalog():
    data = json.loads(files("flow_engine").joinpath("data", CATALOG).read_text())
    models = data["models"]
    if len(models) != 20 or len({m["repo_id"] for m in models}) != 20:
        raise ValueError("Expected 20 distinct checkpoint candidates")
    ranks = [m["overall_rank"] for m in models]
    if ranks != sorted(set(ranks)) or any(not REPO.fullmatch(m["repo_id"]) for m in models):
        raise ValueError("Invalid catalog rank or repository")
    return data


def _permitted_url(url):
    p = urlparse(url)
    # Metadata redirects may use HF's resolve-cache API, never a CDN/Xet blob.
    return (p.scheme == "https" and p.hostname == "huggingface.co" and
            p.port in (None, 443) and not p.username and not p.password and
            (p.path.startswith("/api/models/") or
             (p.path.startswith("/api/resolve-cache/models/") and p.path.endswith("/config.json")) or
             ("/resolve/" in p.path and p.path.endswith("/config.json"))))


class MetadataRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _permitted_url(newurl):
            raise ValueError("Refusing non-metadata redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class MetadataClient:
    def __init__(self, timeout=12):
        if not 0 < timeout <= 60:
            raise ValueError("timeout must be in (0, 60]")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), MetadataRedirect())
        self.events = []

    def get_json(self, url):
        if not _permitted_url(url):
            raise ValueError("Only HF repository metadata and config.json are allowed")
        start = time.monotonic()
        event = {"method": "GET", "url": url, "proxy": "disabled", "bytes": 0}
        try:
            req = Request(url, headers={"User-Agent": "Flow-Inference-Metadata-Audit/0.1",
                                        "Accept": "application/json", "Accept-Encoding": "identity"})
            with self.opener.open(req, timeout=self.timeout) as response:
                if not _permitted_url(response.geturl()):
                    raise ValueError("Unexpected response destination")
                if int(response.headers.get("Content-Length", "0")) > LIMIT:
                    raise ValueError("Metadata exceeds 2 MiB limit")
                raw = response.read(LIMIT + 1)
                event["bytes"] = len(raw)
                if len(raw) > LIMIT:
                    raise ValueError("Metadata exceeds 2 MiB limit")
                result = json.loads(raw.decode("utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object")
                event.update(status=response.status, sha256=hashlib.sha256(raw).hexdigest())
                return result, event["sha256"]
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["elapsed_seconds"] = round(time.monotonic() - start, 4)
            self.events.append(event)


def _quantization_summary(value):
    if not isinstance(value, dict):
        return {"malformed": True}
    result = {k: value[k] for k in ("quant_method", "format", "fmt", "expert_dtype",
              "activation_scheme", "scale_fmt", "weight_block_size", "bits", "group_size") if k in value}
    for key in ("ignore", "ignored_layers", "modules_to_not_convert"):
        if key in value:
            result[key + "_count"] = len(value[key]) if isinstance(value[key], (list, dict)) else None
    groups = value.get("config_groups", {})
    if isinstance(groups, dict):
        result["config_groups"] = {name: {k: group[k] for k in ("format", "weights", "input_activations", "output_activations") if k in group}
                                   for name, group in groups.items() if isinstance(group, dict)}
    result["note"] = "Summary only; exact layer exclusions/scales must be read from pinned config and weights before implementing a loader."
    return result


def audit_config(config):
    """Extract architecture evidence without instantiating third-party code."""
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    sections, features = {}, set()

    def visit(d, path, depth=0):
        if depth > 12:
            raise ValueError("Config nesting exceeds audit limit")
        selected = {k: v for k, v in d.items() if k in DIMENSIONS}
        if selected:
            sections[path] = selected
        for key, value in d.items():
            if key == "auto_map" and value:
                features.add("custom_remote_code_declared_not_executed")
            if key == "quantization_config" and value:
                features.add("quantized_storage_requires_loader_and_kernel")
                sections[path + ".quantization_config"] = _quantization_summary(value)
            if key in {"n_routed_experts", "num_local_experts", "num_experts"} and isinstance(value, int) and value > 1:
                features.add("moe_routing_and_expert_weights")
            if key == "kv_lora_rank" and isinstance(value, int) and value > 0:
                features.add("latent_kv_attention_requires_architecture_specific_cache")
            if key in {"sliding_window", "sliding_window_size", "local_layer_ids", "layer_types", "attention_types", "hybrid_layer_pattern"} and value:
                features.add("window_or_layer_specific_attention_requires_review")
            if key in {"compress_ratios", "index_topk", "indexer_types"} and value:
                features.add("compressed_or_indexed_attention_requires_review")
            if key == "hc_mult" and isinstance(value, int) and value > 1:
                features.add("multi_stream_residual_requires_review")
            if key == "num_nextn_predict_layers" and isinstance(value, int) and value > 0:
                features.add("mtp_weights_present_not_automatic_speculative_support")
            if key == "expert_dtype" and value:
                features.add("expert_precision_can_differ_from_attention_precision")
            if key.startswith("linear_") and value:
                features.add("linear_attention_state_requires_review")
            if key in {"rope_scaling", "rope_parameters"} and value:
                features.add("position_encoding_requires_review")
            if key in {"vision_config", "audio_config", "video_config"} and isinstance(value, dict):
                features.add("multimodal_wrapper_text_backbone_must_be_verified")
            if isinstance(value, dict) and key.endswith("config") and key != "quantization_config":
                visit(value, path + "." + key, depth + 1)

    visit(config, "config")
    try:
        ModelConfig.from_dict(config)
        gate = {"status": "config_accepted_only", "blocker": None}
    except (ValueError, KeyError, TypeError, ZeroDivisionError, AttributeError, OverflowError) as exc:
        gate = {"status": "blocked", "blocker": str(exc)}
    return {"sections": sections, "features": sorted(features), "native_config_gate": gate,
            "synthetic_forward": "not_run", "full_shape_meta_forward": "not_run",
            "real_weights_inference": "not_run", "gpu_performance": "not_run",
            "supported": False,
            "warning": "Metadata audit only. Accepted config is not a numeric forward pass or inference support."}


def audit_remote(entry, timeout=12):
    client = MetadataClient(timeout)
    result = dict(entry, repository_url="https://huggingface.co/" + entry["repo_id"],
                  supported=False, arena_endpoint_checkpoint_equivalence="unverified")
    repo = entry["repo_id"]
    if not REPO.fullmatch(repo):
        raise ValueError("Invalid repository id")
    try:
        info, _ = client.get_json("https://huggingface.co/api/models/" + repo)
        revision = info.get("sha", "")
        if not isinstance(revision, str) or not SHA.fullmatch(revision):
            raise ValueError("Repository did not supply a commit SHA; refusing mutable config fetch")
        card = info.get("cardData") or {}
        siblings = info.get("siblings") or []
        names = [x.get("rfilename", "") for x in siblings if isinstance(x, dict)]
        result.update(revision=revision, repository_license=card.get("license"), gated=info.get("gated"),
                      weight_files_listed=sum(n.endswith(".safetensors") for n in names),
                      hub_safetensors_summary=info.get("safetensors"),
                      parameter_count_caveat="Hub storage metadata, not independently counted logical parameters; packed quantization and auxiliary tensors can change the count.")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
        config, digest = client.get_json(url)
        result.update(metadata_status="fetched", config_url=url, config_sha256=digest,
                      architecture=audit_config(config))
    except Exception as exc:
        result.update(metadata_status="unavailable", error=str(exc))
    result["requests"] = client.events
    return result


def audit_captured(bundle):
    """Re-audit captured JSON; preserve weaker provenance instead of inventing pins."""
    catalog = load_catalog()
    targets = {m["repo_id"]: m for m in catalog["models"]}
    records, seen = [], set()
    for incoming in bundle["models"]:
        repo = incoming["repo_id"]
        if repo not in targets or repo in seen:
            raise ValueError("Unknown or repeated captured repository")
        seen.add(repo)
        record = dict(targets[repo], supported=False)
        for key in ("config_url", "revision", "revision_verified_for_config", "source_representation", "transport"):
            record[key] = incoming.get(key)
        config = incoming.get("config")
        if config is None:
            record.update(metadata_status="unavailable", error=incoming.get("error", "Missing config"))
        else:
            canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if len(canonical) > LIMIT:
                raise ValueError("Captured config exceeds 2 MiB")
            record.update(metadata_status="captured_config_audited", architecture=audit_config(config),
                          canonical_config_sha256=hashlib.sha256(canonical).hexdigest(),
                          hash_scope="Reconstructed canonical JSON, not upstream raw bytes")
        records.append(record)
    if seen != set(targets):
        raise ValueError("Captured bundle must account for all 20 targets, including unavailable entries")
    records.sort(key=lambda r: r["overall_rank"])
    return {**{k: v for k, v in catalog.items() if k != "models"},
            "capture_date": bundle.get("captured_at"),
            "source_note": "Local audit of externally captured configs; claimed revision provenance retained, not independently network verified by this run.",
            "network_requests_by_this_run": 0, "weights_loaded": False, "remote_code_executed": False,
            "models": records, "summary": {"targets": 20,
            "configs_audited": sum(r["metadata_status"] == "captured_config_audited" for r in records),
            "inference_qualified": 0}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch", action="store_true", help="Explicitly fetch metadata/config only, direct connection without proxies")
    p.add_argument("--repo", action="append", help="Limit to an exact repo id from the frozen list; repeatable")
    p.add_argument("--config", type=Path, help="Audit one local config.json without any network")
    p.add_argument("--captured-configs", type=Path, help="Offline audit of a 20-model captured JSON bundle; does not verify provenance remotely")
    p.add_argument("--timeout", type=float, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 4 or not 0 < args.timeout <= 60:
        p.error("workers must be 1..4 and timeout must be in (0,60]")
    if args.output and args.output.exists():
        p.error("Output already exists; choose a new filename")
    if (args.config or args.captured_configs) and (args.fetch or args.repo):
        p.error("Local config audit cannot be combined with --fetch or --repo")
    if args.config and args.captured_configs:
        p.error("Choose one local config input mode")
    if args.captured_configs:
        if args.captured_configs.stat().st_size > 32 * 1024 * 1024:
            p.error("Captured bundle exceeds 32 MiB")
        report = audit_captured(json.loads(args.captured_configs.read_text(encoding="utf-8")))
    elif args.config:
        if args.config.stat().st_size > LIMIT:
            p.error("Local config exceeds 2 MiB")
        report = audit_config(json.loads(args.config.read_text(encoding="utf-8")))
    else:
        catalog = load_catalog()
        models = catalog["models"]
        if args.repo:
            unknown = set(args.repo) - {m["repo_id"] for m in models}
            if unknown:
                p.error("Repo not in frozen target list: " + ", ".join(sorted(unknown)))
            models = [m for m in models if m["repo_id"] in args.repo]
        if args.fetch:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(lambda e: audit_remote(e, args.timeout), models))
        else:
            results = [dict(m, metadata_status="not_fetched", supported=False) for m in models]
        report = {k: v for k, v in catalog.items() if k != "models"}
        report.update(created_at=datetime.now(timezone.utc).isoformat(), weights_downloaded=False,
                      remote_code_executed=False, proxy="disabled", models=results,
                      summary={"targets": len(results), "metadata_fetched": sum(r["metadata_status"] == "fetched" for r in results),
                               "inference_qualified": 0})
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(json.dumps({"output": str(args.output), "summary": report.get("summary"),
                      "warning": "Audit completed, not inference qualification."}, ensure_ascii=False)
          if args.output else payload)
    # Exit success reports a successful audit, not qualification. An attempted
    # network fetch with failures has a separate nonzero status for automation.
    if args.fetch and any(r["metadata_status"] != "fetched" for r in report["models"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
