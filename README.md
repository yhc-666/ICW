# ICW

Python 3.12. Install dependencies with `pip install -r requirements.txt`.

## Data

Place the prepared benchmark JSONL files in:

```text
data/rubricbench.jsonl
data/researchqa.jsonl
data/judgebench.jsonl
data/rmbench-chat.jsonl
data/rewardbench-chat-hard.jsonl
data/rewardbench2-precise-if.jsonl
```

Each row needs `sample_id`, `prompt`, `response_a`, `response_b`,
`preferred_response` (`A` or `B`), and `reference_rubrics`.
## Run

From the repository root:

```bash
export API_BASE_URL="https://your-endpoint/v1"
export API_KEY="your-api-key"
export WEIGHT_MODEL="your-weighting-model"
export VERIFIER_MODEL="your-verifier-model"
export VERIFIER_EXTRA_BODY='{"reasoning_effort":"high"}'
python scripts/run_formal.py --data-dir data --output outputs/formal --concurrency 8
```

`API_BASE_URL` is the prefix before `/chat/completions`.
`WEIGHT_EXTRA_BODY` and `VERIFIER_EXTRA_BODY` accept optional JSON provider
settings; both default to `{}`. Set them to match the selected models.
All six benchmarks run by default; `--benchmarks rubricbench researchqa`
selects specific prepared files.

After successful completion, the output directory contains only
`weights.jsonl` (final rubric weights), `scores.jsonl` (A/B scores and
predictions), and `metrics.json` (benchmark accuracy and macro accuracy).
