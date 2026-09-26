"""Load prepared pairwise benchmarks and their fixed rubric sets."""

import json
from dataclasses import dataclass
from pathlib import Path

from schemas.rubric import Rubric, parse_rubrics


DATASET_NAMES = {
    "rmbench": "rmbench-chat",
    "rewardbench": "rewardbench-chat-hard",
    "rewardbench2": "rewardbench2-precise-if",
}


@dataclass(frozen=True)
class PairwiseExample:
    sample_id: str
    prompt: str
    response_a: str
    response_b: str
    preferred_response: str
    rubrics: list[Rubric]


def load_examples(path: Path, benchmark: str) -> list[PairwiseExample]:
    """Validate all rows before issuing model requests."""
    examples = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each row must be an object")
                for field in ("sample_id", "prompt", "response_a", "response_b"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"{field} must be a nonempty string")
                preferred = row.get("preferred_response")
                if preferred not in {"A", "B"}:
                    raise ValueError("preferred_response must be A or B")
                if "label" in row:
                    expected = 0 if preferred == "A" else 1
                    if type(row["label"]) is not int or row["label"] != expected:
                        raise ValueError("label conflicts with preferred_response")
                dataset_name = row.get("benchmark", benchmark)
                if not isinstance(dataset_name, str):
                    raise ValueError("benchmark must be a string")
                if DATASET_NAMES.get(dataset_name, dataset_name) != benchmark:
                    raise ValueError("benchmark does not match its input file")
                if row["sample_id"] in seen:
                    raise ValueError(f"duplicate sample_id: {row['sample_id']}")
                rubrics = parse_rubrics(row.get("reference_rubrics"))
                seen.add(row["sample_id"])
                examples.append(PairwiseExample(
                    sample_id=row["sample_id"],
                    prompt=row["prompt"],
                    response_a=row["response_a"],
                    response_b=row["response_b"],
                    preferred_response=preferred,
                    rubrics=rubrics,
                ))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"empty benchmark: {path}")
    return examples
