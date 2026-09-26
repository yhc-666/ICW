"""Rubric criteria without method-specific weights."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rubric:
    rubric_id: str
    criterion: str

    def to_json(self) -> dict[str, str]:
        return {"rubric_id": self.rubric_id, "criterion": self.criterion}


def parse_rubrics(value: object) -> list[Rubric]:
    """Validate a nonempty rubric set with unique identifiers."""
    if not isinstance(value, list) or not value:
        raise ValueError("reference_rubrics must be a nonempty list")
    rubrics = []
    seen = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"rubric_id", "criterion"}:
            raise ValueError("each rubric must contain only rubric_id and criterion")
        if any(not isinstance(text, str) or not text.strip() for text in item.values()):
            raise ValueError("rubric_id and criterion must be nonempty strings")
        rubric = Rubric(item["rubric_id"].strip(), item["criterion"].strip())
        if rubric.rubric_id in seen:
            raise ValueError(f"duplicate rubric_id: {rubric.rubric_id}")
        seen.add(rubric.rubric_id)
        rubrics.append(rubric)
    return rubrics
