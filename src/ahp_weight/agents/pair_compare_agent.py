"""Response-independent pairwise importance judgments."""

import json
from dataclasses import dataclass

from infra.llm.client import LLMClient
from schemas.rubric import Rubric


PAIR_COMPARE_SYSTEM_PROMPT = (
    "You compare the importance of two rubric criteria for one user prompt. "
    "Importance means scoring weight: when grading candidate answers to this "
    "prompt, failing a more important criterion must cost more credit. Use "
    "only the user prompt and the fixed rubric set. Never guess at candidate "
    "answers, human preferences, or verifier outcomes."
)


RUBRICBENCH_PRIORITIES = """Priorities:
Graders first check that an answer honors every explicit requirement of the
prompt, then that it is correct and useful.
1. Requirements the prompt states explicitly: an exact count, length,
   language, wording, ordering, format, platform, rule set, named method, or
   required output form. An answer that breaks one fails the request.
2. Correctness and safety of the requested result.
3. Content the user needs to actually use the result.
4. Optional explanation, alternatives, examples, and polish the prompt never
   asked for.
Two rubrics that each check something the prompt explicitly asks for are
equally important. Restating the task, or asking for the answer to be
complete, functional, or runnable, does not make a rubric an explicit
requirement; a specific check of correctness, compatibility, or user-visible
behavior matters more than such a broad one."""


RESEARCHQA_PRIORITIES = """Priorities:
Graders look first for the content that actually answers the question.
1. Content that directly answers the question: the facts for a what-question,
   the mechanism for a how-question, the cause for a why-question, and any
   explicitly requested comparison or implication.
2. Specific evidence, limitations, measurable factors, and downstream effects
   behind the answer.
3. Background, breadth, organization, and citations, unless the prompt
   requests them.
Judge what a rubric asks the answer to contain, not words such as detailed or
in-depth. A rubric naming a concrete strategy, mechanism, cause, or factor
that answers the question matters more than a broad rubric asking the answer
to explore the topic in depth or to be effective."""


JUDGEBENCH_PRIORITIES = """Priorities:
Graders care most about whether the final answer is right, then whether the
reasoning behind it holds up.
1. The final answer, conclusion, choice, or computed value being correct.
2. The overall reasoning being valid.
3. Conditions and constraints stated explicitly in the prompt.
4. Presentation and formatting.
A rubric that checks the final answer as delivered, including an answer
format the prompt requires, belongs to the first priority. A rubric that only
limits the answer to the given options does not check that the answer is
right, so it matters less. When the problem requires a calculation, doing it
correctly matters more than the exact form of the answer. Premises and setup
facts of the same problem are equally important, even when one seems broader."""


RMBENCH_CHAT_PRIORITIES = """Priorities:
Graders check first that the answer is accurate and fits the situation the
user described, then that it is complete and usable.
1. Exactly correct and current content, with realistic limitations, necessary
   caveats, and safety. Restating the task is not a correctness check.
2. Conditions the user stated explicitly, such as their location,
   environment, skill level, scope, or only/no restrictions, even when a
   condition looks minor.
3. A complete main deliverable or requested recommendation.
4. Steps and reasoning the user needs to apply the answer.
5. Style, organization, tone, and formatting the user never requested.
Between rubrics of equal priority, the one more specifically tied to the
prompt matters more."""


CHAT_HARD_PRIORITIES = """Priorities:
Graders check first that the answer does exactly what the instruction asks.
1. Correct and safe completion of exactly the requested task, including any
   rule the prompt sets on how to respond, such as "only X" or a required
   format.
2. Other constraints stated explicitly in the prompt.
3. Substance the user needs to apply the answer.
4. Length, tone, confidence, and surface polish.
A rubric about the exact subject and action of the prompt matters more than a
broad quality claim. Sophistication, length, or confident wording never make
up for missing the task, a factual error, a safety problem, or a broken rule."""


PRECISE_IF_PRIORITIES = """Priorities:
This task type tests whether the answer follows the exact rules the prompt
sets.
1. Rules the prompt sets on the form or procedure of the answer: casing,
   count, wording, language, ordering, length, punctuation, code only or no
   explanation, or an exact output format.
2. Correct and complete main content of the requested operation.
3. Supporting detail the user needs.
4. Style and polish the prompt never asked for.
A stated rule matters more than the main content, even when the content looks
more central. A restriction the prompt never states is not a rule."""


BENCHMARK_PRIORITY_BLOCKS: dict[str, str] = {
    "rubricbench": RUBRICBENCH_PRIORITIES,
    "researchqa": RESEARCHQA_PRIORITIES,
    "judgebench": JUDGEBENCH_PRIORITIES,
    "rmbench-chat": RMBENCH_CHAT_PRIORITIES,
    "rewardbench-chat-hard": CHAT_HARD_PRIORITIES,
    "rewardbench2-precise-if": PRECISE_IF_PRIORITIES,
}


_USER_TEMPLATE_SKELETON = """User prompt:
{prompt}

Full rubric set (fixed context for this sample):
{full_rubric_set}

Compare these two rubrics:
first  ({first_rubric_id}): {first_criterion}
second ({second_rubric_id}): {second_criterion}

Which highlighted rubric should carry more weight when grading answers to
this prompt?

<benchmark_priorities>

General rules:
- First list what the prompt itself fixes: the requested result and every
  explicit must, must-not, count, format, or procedural rule. A rubric tied
  to any of them outranks generic quality.
- Rank two rubrics only when the prompt or the priorities above give a
  reason; never build a preference from plausible reasoning alone. If
  neither rubric is favored by them, answer "neutral".
- The priorities above run from most to least important. A rubric under a
  higher priority is more important, even when the other sounds more
  central. Two rubrics under the same priority are equally important, so
  answer "neutral", unless the prompt or the notes above favor one.
- Judge the concrete behavior each rubric uniquely requires, not wording such
  as detailed, complete, robust, or professional, and not a category label
  attached to the rubric.
- Use the full rubric set: a rubric that repeats another, vaguely sums up
  several others, or restates the whole task is less important than a
  specific requirement that only it covers.
- For a negatively phrased rubric, judge the value of avoiding that defect;
  do not reverse its meaning.
- First decide which rubric_id wins; your answer must not change if the two
  rubrics were displayed in the opposite order.
{revisit_feedback}
Return JSON only:
{{"more_important": "first" | "second" | "neutral", "reason": "<= 25 words"}}
"""


PAIR_COMPARE_USER_TEMPLATES: dict[str, str] = {
    benchmark: _USER_TEMPLATE_SKELETON.replace(
        "<benchmark_priorities>",
        priorities,
    )
    for benchmark, priorities in BENCHMARK_PRIORITY_BLOCKS.items()
}


# Without task priors: the same comparison and general rules, minus the
# benchmark priority block and the two rules that refer to it.
USER_TEMPLATE_WITHOUT_PRIORS = """User prompt:
{prompt}

Full rubric set (fixed context for this sample):
{full_rubric_set}

Compare these two rubrics:
first  ({first_rubric_id}): {first_criterion}
second ({second_rubric_id}): {second_criterion}

Which highlighted rubric should carry more weight when grading answers to
this prompt?

General rules:
- First list what the prompt itself fixes: the requested result and every
  explicit must, must-not, count, format, or procedural rule. A rubric tied
  to any of them outranks generic quality.
- Rank two rubrics only when the prompt gives a reason; never build a
  preference from plausible reasoning alone. If neither rubric is favored
  by the prompt, answer "neutral".
- Judge the concrete behavior each rubric uniquely requires, not wording such
  as detailed, complete, robust, or professional, and not a category label
  attached to the rubric.
- Use the full rubric set: a rubric that repeats another, vaguely sums up
  several others, or restates the whole task is less important than a
  specific requirement that only it covers.
- For a negatively phrased rubric, judge the value of avoiding that defect;
  do not reverse its meaning.
- First decide which rubric_id wins; your answer must not change if the two
  rubrics were displayed in the opposite order.
{revisit_feedback}
Return JSON only:
{{"more_important": "first" | "second" | "neutral", "reason": "<= 25 words"}}
"""


# What a review checks the earlier reasons against, with and without priors.
REVIEW_BASIS_WITH_PRIORS = "the prompt and the priorities above"
REVIEW_BASIS_WITHOUT_PRIORS = "the prompt"


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
checking both previous reasons against {review_basis}.
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
    def __init__(
        self, *, client: LLMClient, model: str, benchmark: str, extra_body: dict, use_task_priors: bool
    ):
        if benchmark not in PAIR_COMPARE_USER_TEMPLATES:
            raise ValueError(f"unsupported benchmark: {benchmark}")
        self.client = client
        self.model = model
        self.extra_body = extra_body
        if use_task_priors:
            self.template = PAIR_COMPARE_USER_TEMPLATES[benchmark]
            self.review_basis = REVIEW_BASIS_WITH_PRIORS
        else:
            self.template = USER_TEMPLATE_WITHOUT_PRIORS
            self.review_basis = REVIEW_BASIS_WITHOUT_PRIORS

    def review_feedback(self, previous_round_history: str) -> str:
        """Render the consistency-review context for this prompt variant."""
        return PAIR_REVISIT_FEEDBACK_TEMPLATE.format(
            previous_round_history=previous_round_history,
            review_basis=self.review_basis,
        )

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
        user_prompt = self.template.format(
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
