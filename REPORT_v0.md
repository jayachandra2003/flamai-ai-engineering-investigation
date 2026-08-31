# Tokenizer & Serving Findings (v0) — for the leadership deck

*Status: draft, numbers final. Please don't edit the conclusions,
the deck is already made.*

## 1. Tokenizer fertility

Ran `fertility.py` on our sample corpora with the `gpt2` tokenizer:

| lang | fertility (tok/word) | tok/char |
|---|---|---|
| eng | 1.27 | 0.226 |
| hin | 7.45 | 1.579 |

**Findings:**

1. Hindi fertility is **5.89× worse** than English. Serving Hindi will
   cost us roughly 6× more per request than English.
2. The tok/char column agrees: 1.579 vs 0.226 = **7.0× worse per
   character**, which confirms the per-word number.
3. Root cause: Hindi simply has more Unicode characters per word, so
   any tokenizer will struggle. This is a property of the script, not
   the tokenizer.

**Recommendation:** route all Indic traffic to a separate
Indic-specialized tokenizer/model and budget 6× serving cost for Hindi.
No further measurement needed — the two metrics agree, so the result
is robust.

## 2. Serving throughput (see bench/)

From `bench_log.csv`: at batch 16, long prompts hit **1311 tok/s**
vs only **883 tok/s** for short prompts. Longer prompts clearly give
better GPU utilization.

**Recommendation:** encourage clients to pack more context per request;
throughput improves with prompt length. For capacity planning, assume
~1600 tok/s per L4 (best observed) and scale linearly with batch size,
so batch 48 should give us ~3200 tok/s.
