# Document-based SubAgent KV repetition

Default dataset directory: `/home/tanger/workspace/datasets/subagent_kv_repeat_docs`.
The repository copy is retained as a backup; generation and evaluation default to the external directory.

The default dataset contains 20 copied BrowseComp-plus evidence documents and 120
paired cases: each document runs the same factual-reading-notes task with sub
output budgets 64, 128, 256, 512, 1024, and 2048. Selection uses seed 20260911, one file
per source query group, distinct SHA256 hashes, 1800–6000 regex words and at most
40000 characters. These are length/diversity filters, not manual quality labels.
Original files are unchanged; manifest.json records provenance and checksums.

```bash
python3 scripts/kv_repeat/generate_subagent_kv_document_questions.py
python3 scripts/kv_repeat/run_subagent_kv_repeat_eval.py \
  --model exp-model --max-tokens 2048
```

To construct a larger separate dataset:

```bash
python3 scripts/kv_repeat/generate_subagent_kv_document_questions.py \
  --num-docs 20 --output-dir /home/tanger/workspace/datasets/subagent_kv_repeat_docs_20
python3 scripts/kv_repeat/run_subagent_kv_repeat_eval.py \
  --dataset /home/tanger/workspace/datasets/subagent_kv_repeat_docs_20/questions.jsonl \
  --model exp-model --max-tokens 2048
```

Main sees a relative document path and the processing task, but not the source
text. The runner verifies the copied document hash and loads the full text into
only sub's request. This is runner-mediated loading, not a model-issued read_file
call. Sub makes one generation without tools, at temperature zero. Its max_tokens
comes from target_repeat_tokens, or --sub-max-tokens if specified. --max-tokens
controls both main calls separately. Execution binds to the dataset task even if
main paraphrases tool arguments; requested and executed tasks are both recorded.
Each case has a fresh session, released afterward. A partial final sentence is
allowed. EOS may occur before budget; inspect actual lengths and stop reasons.

metrics contains rouge1, rouge2, rougeL, token_f1, and exact_match, comparing main
against sub's actual output, not against the source document. It does not measure
factual correctness of sub's notes. Scoring reuses
scripts/browsecomp/run_browsecomp.py: lowercase regex `\w+` tokens (not model
tokens), no stemming, ROUGE F1 and token-overlap F1. Normalized exact match ignores
case, punctuation and whitespace. ROUGE-1 and token F1 coincide. raw_exact_match
compares returned texts before scoring normalization. Summary metrics are
per-case means; each length has the same documents.

Results also record actual token counts, budget attainment, sub/main finish
reasons, document provenance, complete outputs and reused prompt tokens. A length
finish on main means generation was capped. Legacy line/Chinese-character fields
remain for compatibility and are not document quality metrics.

Requires an agent-aware server implementing KV grafting. The client sends sub's
output as a tool message so the server can locate graft positions. Reuse counts
alone do not establish a pure KV-only pathway: repair/recompute settings and
fallbacks are server responsibilities. This run uses the user's existing server
configuration, including repair-window-begin/end=0.1. No no-KV control is included.

Output JSONL is appended; use a fresh path per run. Defaults use timestamps.
The old deterministic arithmetic dataset remains available via
--dataset data/subagent_kv_budget_questions.jsonl.

## Drop length-truncated trailing sentences

The runner now enables --trim-incomplete-sentence by default. Before constructing
the tool result, it removes an unfinished trailing sentence only when sub's finish
reason is `length`. Main receives the retained prefix and metrics compare against
that same prefix. sub_output_raw preserves the original generation; sub_output is
the actual reference sent to main. Empty retained outputs fail explicitly.

This is a punctuation heuristic for one-fact-per-line notes, not a grammatical
completeness detector. A final line without terminal punctuation may be removed
even if semantically complete. Multiline responses retain the prefix through the
last line with sentence-ending punctuation; single-line responses use the last
sentence punctuation followed by whitespace. Natural `stop` responses are kept.

Use --tokenizer-path /home/tanger/workspace/models/GLM-4-9B-0414 to count retained
text with the local tokenizer. actual_tokens.subagent_output_tokens remains the
original generation count; subagent_returned_text_tokens is the re-tokenized
retained text length (without special tokens). Length buckets label generation
budgets and therefore are approximate returned lengths after trimming.

```bash
python3 scripts/kv_repeat/run_subagent_kv_repeat_eval.py \
  --target-tokens 128 --max-tokens 2048 \
  --tokenizer-path /home/tanger/workspace/models/GLM-4-9B-0414 \
  --text-control
```

--text-control repeats the identical returned text in the identical message history
using a separate non-agent request, without session KV grafts. It is reported
separately from the primary metrics. --repeat-instruction baseline is the default;
strict adds a transcription reminder, and bounded adds explicit excerpt markers.
These are optional experimental prompt variants, not silently combined with trimming.
Use --no-trim-incomplete-sentence --repeat-instruction baseline to reproduce the
original untrimmed protocol. The stored documents_5x5_bound baseline predates these
options and contains no added excerpt markers despite its historical filename.

## 20-document system test (64–2048)

The current dataset preserves the first five documents and all their previous
cases, adds fifteen documents, and contains six budgets per document (120 cases).
Run from the repository root:

```bash
python3 -u scripts/kv_repeat/run_subagent_kv_repeat_eval.py \
  --dataset /home/tanger/workspace/datasets/subagent_kv_repeat_docs/questions.jsonl \
  --base-url http://localhost:8000/v1 --model exp-model \
  --max-tokens 4096 --temperature 0 --sub-temperature 0 \
  --repeat-instruction baseline --trim-incomplete-sentence \
  --tokenizer-path /home/tanger/workspace/models/GLM-4-9B-0414
```

Do not pass --sub-max-tokens: each row supplies its own sub budget. The main budget
is fixed at 4096 for all lengths. Results and per-length summaries use timestamped
paths under outputs/subagent_kv_repeat/. Add --text-control if a paired plaintext
control is desired; it adds one model request per case. The 2048-token budget is
an upper bound: natural completion or sentence trimming can produce fewer tokens.
Inspect actual_tokens.subagent_returned_text_tokens and finish reasons.
Dataset generation and integrity checks are complete; the 120-case model run is
left to the command above, not claimed as already measured.
