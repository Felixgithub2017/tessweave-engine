# Security boundaries

This is a local research server, not a hardened multitenant service. It binds only to 127.0.0.1. Use an SSH tunnel to a trusted remote host; do not expose the port through a public proxy without authentication, TLS, rate limits and a security review.

The loader accepts local safetensors only and never executes model-repository Python. It rejects custom-code configurations. Tokenizer parsing and dependencies still consume untrusted data, so only use trusted checkpoints. Shard names cannot use path traversal; individual HF-cache symlinks are allowed. Weight size/resource demands can still exhaust memory.

HTTP requests have a 1 MiB body cap, finite message/token limits and bounded request/output queues. These are not a defense against all denial-of-service attacks, slow clients or OS resource exhaustion. Shared cache namespaces exist in the internal API; the public local endpoint uses a single namespace and has no tenant isolation or authenticated user model.

Default rotating traces omit raw prompts, token IDs and response text, but contain model paths, configuration and diagnostics. Benchmark reports intentionally contain public workload outputs. Protect logs and do not post private paths, credentials or sensitive model responses in issues.

Before public repository publication, configure a private vulnerability-reporting channel. Until that channel exists, do not post exploit details publicly. This project currently offers no security-response SLA.
