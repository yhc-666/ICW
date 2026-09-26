"""Run the formal ICW benchmark suite and save final results only."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from ahp_weight.agents.pair_compare_agent import PairCompareAgent
from ahp_weight.methods.fixed_ratio_single import compute_weights
from bench_eval.datasets.pairwise_loader import PairwiseExample, load_examples
from bench_eval.scoring import preference_credit, score_responses
from bench_eval.verifiers.per_criterion_verifier import PerCriterionVerifier
from infra.llm.client import LLMClient, parse_extra_body


BENCHMARK_SETTINGS = {
    "rubricbench": (1.3, 0.005),
    "researchqa": (1.3, 0.02),
    "judgebench": (5.0, 0.25),
    "rmbench-chat": (4.0, 0.25),
    "rewardbench-chat-hard": (2.5, 0.15),
    "rewardbench2-precise-if": (5.0, 0.25),
}


async def evaluate_benchmark(
    *,
    benchmark: str,
    examples: list[PairwiseExample],
    agent: PairCompareAgent,
    verifier: PerCriterionVerifier,
    concurrency: int,
) -> tuple[list[dict], list[dict], dict]:
    """Evaluate every input row with prompt-only importance weighting."""
    alpha, tau = BENCHMARK_SETTINGS[benchmark]
    semaphore = asyncio.Semaphore(concurrency)
    weight_jobs = {}

    async def evaluate(example: PairwiseExample) -> tuple[dict, dict, float]:
        async with semaphore:
            weight_key = (
                example.prompt,
                tuple((rubric.rubric_id, rubric.criterion) for rubric in example.rubrics),
            )
            if weight_key not in weight_jobs:
                weight_jobs[weight_key] = asyncio.create_task(compute_weights(
                    prompt=example.prompt,
                    rubrics=example.rubrics,
                    agent=agent,
                    alpha=alpha,
                    tau=tau,
                ))
            weights = await weight_jobs[weight_key]
            vector_a, vector_b = await asyncio.gather(
                verifier.verify(prompt=example.prompt, response=example.response_a, rubrics=example.rubrics),
                verifier.verify(prompt=example.prompt, response=example.response_b, rubrics=example.rubrics),
            )
            identity = {"benchmark": benchmark, "sample_id": example.sample_id}
            scores = score_responses(weights, vector_a, vector_b)
            credit = preference_credit(scores["prediction"], example.preferred_response)
            return {**identity, "weights": weights}, {**identity, **scores}, credit

    results = await asyncio.gather(*(evaluate(example) for example in examples))
    metrics = {
        "num_examples": len(examples),
        "accuracy": sum(credit for _, _, credit in results) / len(examples),
    }
    return [weights for weights, _, _ in results], [scores for _, scores, _ in results], metrics


def required_environment(name: str) -> str:
    """Require runtime connection and model settings."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"set {name}")
    return value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write final records without intermediates or model responses."""
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


async def run(args: argparse.Namespace) -> dict:
    """Run the selected benchmarks, then publish their complete final outputs."""
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if len(args.benchmarks) != len(set(args.benchmarks)):
        raise ValueError("benchmarks must be unique")
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    datasets = {
        name: load_examples(args.data_dir / f"{name}.jsonl", name)
        for name in args.benchmarks
    }
    base_url = required_environment("API_BASE_URL")
    api_key = required_environment("API_KEY")
    weight_model = required_environment("WEIGHT_MODEL")
    verifier_model = required_environment("VERIFIER_MODEL")
    weight_extra = parse_extra_body(os.environ.get("WEIGHT_EXTRA_BODY", "{}"))
    verifier_extra = parse_extra_body(os.environ.get("VERIFIER_EXTRA_BODY", "{}"))

    all_weights, all_scores, benchmark_metrics = [], [], {}
    async with LLMClient(base_url=base_url, api_key=api_key, concurrency=args.concurrency) as client:
        verifier = PerCriterionVerifier(client=client, model=verifier_model, extra_body=verifier_extra)
        for benchmark, examples in datasets.items():
            agent = PairCompareAgent(
                client=client,
                model=weight_model,
                benchmark=benchmark,
                extra_body=weight_extra,
                use_task_priors=args.task_priors,
            )
            weights, scores, metrics = await evaluate_benchmark(
                benchmark=benchmark,
                examples=examples,
                agent=agent,
                verifier=verifier,
                concurrency=args.concurrency,
            )
            all_weights.extend(weights)
            all_scores.extend(scores)
            benchmark_metrics[benchmark] = metrics
    metrics = {
        "task_priors": args.task_priors,
        "benchmarks": benchmark_metrics,
        "macro_accuracy": sum(item["accuracy"] for item in benchmark_metrics.values()) / len(benchmark_metrics),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "weights.jsonl", all_weights)
    write_jsonl(output / "scores.jsonl", all_scores)
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return metrics


def main() -> None:
    """Command-line entry for the formal benchmark experiment."""
    parser = argparse.ArgumentParser(description="Run ICW with one full pairwise review.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(BENCHMARK_SETTINGS), default=list(BENCHMARK_SETTINGS))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--task-priors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the domain priority block in the comparison prompt",
    )
    args = parser.parse_args()
    try:
        metrics = asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(metrics, indent=2))
