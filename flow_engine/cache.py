"""Physical KV pages, reference ownership and a full-block prefix trie by hash.

Each cached block key includes its parent hash and token IDs. Published pages are
immutable; partial tails are never shared. The cache owns references independent
of live requests, so LRU eviction cannot free a page still used by a request.
"""
from collections import OrderedDict
import hashlib
import struct
import torch


class CapacityError(RuntimeError):
    pass


class KVPool:
    def __init__(self, model, blocks, block_size, dtype, device, prefix_cache=True):
        self.block_size = block_size
        self.capacity = blocks
        shape = (model.num_hidden_layers, blocks, block_size, model.num_key_value_heads, model.head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty_like(self.k)
        self.refs = [0] * blocks
        self.free = list(reversed(range(blocks)))
        self.prefixes = OrderedDict()
        self.prefix_cache = prefix_cache
        self.hits = 0
        self.evictions = 0

    @property
    def allocated_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    def _release_one(self, block):
        if self.refs[block] <= 0:
            raise RuntimeError("KV reference underflow")
        self.refs[block] -= 1
        if not self.refs[block]:
            self.free.append(block)

    def release(self, table):
        for block in table:
            self._release_one(block)

    def allocate(self, count):
        if count < 0:
            raise ValueError("Negative allocation")
        while len(self.free) < count and self.prefixes:
            _, block = self.prefixes.popitem(last=False)
            self._release_one(block)
            self.evictions += 1
        if len(self.free) < count:
            raise CapacityError("KV capacity exhausted; request must wait or reduce its token budget")
        result = [self.free.pop() for _ in range(count)]
        for block in result:
            self.refs[block] = 1
        return result

    def _keys(self, tokens, namespace):
        parent = hashlib.sha256(namespace.encode()).digest()
        for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
            chunk = tokens[start:start + self.block_size]
            parent = hashlib.sha256(parent + struct.pack(f"<{len(chunk)}q", *chunk)).digest()
            yield parent

    def acquire_prefix(self, tokens, namespace="local"):
        table = []
        if self.prefix_cache:
            # At least the last prompt token must run to recover its logits.
            for key in self._keys(tokens[:-1], namespace):
                if key not in self.prefixes:
                    break
                block = self.prefixes[key]
                self.prefixes.move_to_end(key)
                self.refs[block] += 1
                table.append(block)
        return table

    def publish(self, tokens, computed, table, namespace="local"):
        if not self.prefix_cache:
            return
        for i, key in enumerate(self._keys(tokens[:computed], namespace)):
            if key not in self.prefixes:
                self.prefixes[key] = table[i]
                self.refs[table[i]] += 1

    def clear_prefixes(self):
        for block in self.prefixes.values():
            self._release_one(block)
        self.prefixes.clear()

    def slots(self, table, start, count):
        return [table[p // self.block_size] * self.block_size + p % self.block_size
                for p in range(start, start + count)]

    def write(self, layer, slots, k, v):
        ids = torch.as_tensor(slots, dtype=torch.long, device=self.k.device)
        self.k[layer].flatten(0, 1).index_copy_(0, ids, k)
        self.v[layer].flatten(0, 1).index_copy_(0, ids, v)

    def gather(self, layer, tables, lengths):
        # Portable reference path; copies pages to a dense padded batch. The
        # optional Triton decode kernel bypasses this gather altogether.
        width = max(lengths)
        indices = [self.slots(table, 0, n) + [0] * (width - n) for table, n in zip(tables, lengths)]
        ids = torch.tensor(indices, dtype=torch.long, device=self.k.device)
        k, v = self.k[layer].flatten(0, 1)[ids], self.v[layer].flatten(0, 1)[ids]
        valid = torch.arange(width, device=ids.device)[None] < torch.tensor(lengths, device=ids.device)[:, None]
        # A padding slot may contain uninitialized values; masking attention
        # scores alone does not prevent NaNs in the V multiplication.
        return k.masked_fill(~valid[..., None, None], 0), v.masked_fill(~valid[..., None, None], 0)

    def stats(self):
        return {"physical_blocks": self.capacity, "free_blocks": len(self.free),
                "cached_prefix_blocks": len(self.prefixes), "prefix_hit_tokens": self.hits,
                "evictions": self.evictions, "kv_allocated_bytes": self.allocated_bytes}
