"""Response-independent pairwise importance judgments."""

import json
from dataclasses import dataclass

from infra.llm.client import LLMClient
from schemas.rubric import Rubric


PAIR_COMPARE_SYSTEM_PROMPT = (
    "You help set scoring weights for grading. Given one user prompt and the "
    "rubrics used to grade answers to it, you compare two rubrics and say which "
    "one should carry more weight: failing the more important rubric must cost "
    "an answer more credit. Use only the prompt and the rubrics. Do not guess "
    "what any particular answer will say."
)


PAIR_COMPARE_USER_TEMPLATE = """User prompt:
{prompt}

All rubrics used to grade answers to this prompt:
{full_rubric_set}

Compare these two rubrics:
first  ({first_rubric_id}): {first_criterion}
second ({second_rubric_id}): {second_criterion}

Which of the two should carry more weight when grading answers to this
prompt?

Each rubric has two properties:
- Source: requested (the prompt asks for it in words) or added (the rubric
  adds it).
- Subject: form (how the response is written) or content (what the
  response says).

Priority, from highest to lowest:
1. Requested form. The response is written the way the prompt says:
   length, count, language, casing, layout, sentence patterns, and words or
   characters to use or avoid.
2. Correct result. The final answer or result is right and safe.
3. Requested content. The response says what the prompt asks for.
4. Added content. Supporting facts, reasoning, examples, and depth.
5. Added form. Clarity, organization, tone, and conciseness.

How to decide:
1. Give each rubric its priority from its source and subject.
2. Priority decides: the rubric with the higher priority carries more
   weight, even when the other rubric looks more central to the task.
3. At equal priority, the rubric that checks one concrete thing carries
   more weight than the rubric that states a general wish.
4. At equal priority and equal concreteness, answer "neutral".
5. Give the same answer whichever rubric is shown first.
{revisit_feedback}
Return JSON only:
{{"more_important": "first" | "second" | "neutral", "reason": "<= 25 words"}}
"""


PAIR_REVISIT_FEEDBACK_TEMPLATE = """
Consistency review: the initial judgments for every rubric pair follow.
Each forward/reverse entry preserves its display order, winner_rubric_id,
and original reason. A null winner means that call answered neutral.
The status distinguishes:
- agreed_winner: both display orders support the same rubric_id;
- both_neutral: both calls report no supported importance advantage;
- one_order_neutral: only one call is neutral;
- opposite_winners: the two calls support different rubric_ids.
The last two statuses indicate order instability. Their null collapsed winner
does not establish that the rubrics are equally important. Even both_neutral
can reflect insufficient evidence rather than an established equality.

Initial comparison evidence (fallible prior judgments, not ground truth):
{previous_round_history}

Re-judge the highlighted pair using the user prompt and complete rubric set,
checking both previous reasons against the prompt and the priorities above.
Use the other pairs as context, not as votes or authoritative evidence.
Keep a supported judgment; revise it when its reason is unsupported,
misreads a requirement, or overlooks a relevant requirement. Explain the
concrete prompt or rubric evidence for your judgment. A request for review
does not imply that the highlighted pair's initial judgment was wrong.
Seek coherent importance judgments, but do not invent a preference merely
to remove a conflict. Answer "neutral" when neither rubric is supported as
more important. Return a judgment only for the highlighted pair.
"""


@dataclass(frozen=True)
class Judgment:
    first_id: str
    second_id: str
    winner_id: str | None
    reason: str

    def to_history(self) -> dict:
        return {
            "display_order": [self.first_id, self.second_id],
            "winner_rubric_id": self.winner_id,
            "reason": self.reason,
        }


class PairCompareAgent:
    def __init__(self, *, client: LLMClient, model: str, extra_body: dict):
        self.client = client
        self.model = model
        self.extra_body = extra_body

    def review_feedback(self, previous_round_history: str) -> str:
        """Render the consistency-review context."""
        return PAIR_REVISIT_FEEDBACK_TEMPLATE.format(previous_round_history=previous_round_history)

    async def compare(
        self,
        *,
        prompt: str,
        rubrics: list[Rubric],
        first: Rubric,
        second: Rubric,
        feedback: str = "",
    ) -> Judgment:
        """Compare one ordered pair using only the prompt and rubric context."""
        if first == second or first not in rubrics or second not in rubrics:
            raise ValueError("highlighted pair must contain two distinct rubrics from the set")
        full_set = json.dumps(
            [rubric.to_json() for rubric in sorted(rubrics, key=lambda item: item.rubric_id)],
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = PAIR_COMPARE_USER_TEMPLATE.format(
            prompt=prompt,
            full_rubric_set=full_set,
            first_rubric_id=first.rubric_id,
            first_criterion=first.criterion,
            second_rubric_id=second.rubric_id,
            second_criterion=second.criterion,
            revisit_feedback=feedback,
        )

        def parse(value: dict) -> Judgment:
            decision = value.get("more_important")
            if decision not in {"first", "second", "neutral"}:
                raise ValueError("more_important must be first, second, or neutral")
            reason = value.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("comparison reason must be nonempty")
            winner = None
            if decision == "first":
                winner = first.rubric_id
            elif decision == "second":
                winner = second.rubric_id
            return Judgment(first.rubric_id, second.rubric_id, winner, reason.strip())

        return await self.client.complete(
            model=self.model,
            system_prompt=PAIR_COMPARE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            parse=parse,
            extra_body=self.extra_body,
        )
