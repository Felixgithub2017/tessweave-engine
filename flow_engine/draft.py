"""Bounded prompt lookup. Draft tokens are proposals, NEVER emitted directly.

No learned draft model or external framework. Most-recent longest suffix match
in already committed context; useful for copying/repetition, often useless for
novel answers. O(window * maximum_match) CPU work, with both bounds explicit.
"""


def propose(tokens, maximum, window=512, min_match=3, max_match=6):
    if maximum < 1 or len(tokens) < min_match * 2:
        return []
    history = tokens[-window:]
    for n in range(min(max_match, len(history) // 2), min_match - 1, -1):
        needle = history[-n:]
        for i in range(len(history) - n - 1, -1, -1):
            if history[i:i + n] == needle:
                return history[i + n:min(i + n + maximum, len(history))]
    return []
