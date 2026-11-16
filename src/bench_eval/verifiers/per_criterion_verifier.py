"""Independent binary verification of each response criterion."""

import asyncio
import re

from infra.llm.client import LLMClient
from schemas.rubric import Rubric


PER_CRITERION_SYSTEM_PROMPT = (
    "You are a strict rubric verifier. Given one user prompt, one response, "
    "and ONE evaluation criterion, decide whether the response satisfies that "
    "criterion. Judge only this local criterion, not overall response quality, "
    "and never compare against another response. Mark satisfied=1 only when "
    "the response clearly and completely satisfies the criterion; partial "
    "satisfaction is 0. Return strict JSON only."
)


PER_CRITERION_USER_TEMPLATE = """Original user prompt:
{prompt}

Response to verify:
{response}

Single criterion:
{criterion}

Verification rules:
- Judge only the criterion's stated local boundary. Do not fail it merely
  because another part of the overall task is weak or missing.
- Preserve exact quantifiers: exactly, at least, at most, only, every, and each
  are not interchangeable.
- If the criterion forbids or says to avoid something, satisfied=1 only when
  the prohibited condition is absent.
- Judge meaning and response evidence, not keyword overlap.
- The reason must be one concise final justification. The satisfied bit and
  reason must agree.

Return exactly:
{{"satisfied": 1, "reason": "short evidence-based reason"}}
"""


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


EXACT_NUMERIC_COUNT_PATTERN = re.compile(
    r"^\s*the\s+response\s+"
    r"(?:(?:must\s+)?(?:contain|include|use|have)|contains|includes|uses|has)\s+"
    r"exactly\s+(?P<count>\d+|"
    + "|".join(NUMBER_WORDS)
    + r")\s+"
    r"(?:numbers?|numerals?|numeric(?:al)?\s+occurrences?)\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)


NUMERIC_OCCURRENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)


def parse_satisfaction(value: dict) -> int:
    """Require an explicit binary decision and a supporting reason."""
    satisfied = value.get("satisfied")
    if type(satisfied) is not int or satisfied not in {0, 1}:
        raise ValueError("satisfied must be the integer zero or one")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("verifier reason must be nonempty")
    return satisfied


def check_numeric_count(criterion: str, response: str, satisfied: int) -> int:
    """Check an explicitly stated, response-wide digit-occurrence count."""
    match = EXACT_NUMERIC_COUNT_PATTERN.fullmatch(criterion)
    if match is None:
        return satisfied
    count = match.group("count").lower()
    expected = int(count) if count.isdigit() else NUMBER_WORDS[count]
    return int(len(NUMERIC_OCCURRENCE_PATTERN.findall(response)) == expected)


class PerCriterionVerifier:
    def __init__(self, *, client: LLMClient, model: str, extra_body: dict):
        self.client = client
        self.model = model
        self.extra_body = extra_body

    async def verify(
        self, *, prompt: str, response: str, rubrics: list[Rubric]
    ) -> dict[str, int]:
        """Verify each criterion independently for one candidate response."""
        async def verify_criterion(rubric: Rubric) -> int:
            satisfied = await self.client.complete(
                model=self.model,
                system_prompt=PER_CRITERION_SYSTEM_PROMPT,
                user_prompt=PER_CRITERION_USER_TEMPLATE.format(
                    prompt=prompt, response=response, criterion=rubric.criterion
                ),
                parse=parse_satisfaction,
                extra_body=self.extra_body,
            )
            return check_numeric_count(rubric.criterion, response, satisfied)

        bits = await asyncio.gather(*(verify_criterion(rubric) for rubric in rubrics))
        return {rubric.rubric_id: bit for rubric, bit in zip(rubrics, bits, strict=True)}
