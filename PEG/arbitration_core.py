from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from peg_core import (
    GENERATION_CONFIG,
    call_agent,
)


PANEL_SIZE = 5


# ============================================================
# Judge prompt
# ============================================================

def build_judge_prompt(
    original_prompt: str,
    agent_outputs: List[Dict[str, str]],
) -> str:
    """
    Construct the complete arbitration prompt for the judge.

    Important:
        agent_outputs must contain the five COMPLETE raw responses.
        Nothing is truncated or pre-parsed before judging.

    The manuscript specifies the arbitration behavior but does not
    provide the historical judge prompt verbatim. This prompt is the
    explicit reproducible implementation used for the revised
    experiment.
    """

    if len(agent_outputs) != PANEL_SIZE:
        raise ValueError(
            f"Expected {PANEL_SIZE} agent responses, "
            f"received {len(agent_outputs)}."
        )

    blocks = []

    for position, item in enumerate(
        agent_outputs,
        start=1,
    ):
        blocks.append(
            f"================ AGENT {position} ================\n"
            f"Agent ID: {item['agent_id']}\n\n"
            f"{item['response']}\n"
            f"==============================================="
        )

    responses_text = "\n\n".join(blocks)

    return (
        "You are the designated judge in a multi-agent arbitration "
        "procedure.\n\n"

        "Five independent agents were given the same original prompt. "
        "Each agent produced its own classification and supporting "
        "reasoning without seeing the other agents' responses.\n\n"

        "Your task is to synthesize the five responses into one final "
        "answer to the original prompt. Evaluate the reasoning provided "
        "by all agents, resolve disagreements by weighing the quality "
        "and plausibility of their reasoning, and return a complete final "
        "response that answers the original prompt. Do not decide merely "
        "by majority vote.\n\n"

        "================ ORIGINAL PROMPT ================\n"
        f"{original_prompt.strip()}\n"
        "=================================================\n\n"

        "The five independent responses are provided below "
        "simultaneously:\n\n"

        f"{responses_text}\n"
    )


# ============================================================
# Independent panel generation
# ============================================================

def _run_single_panel_agent(
    agent_id: str,
    agent: Any,
    original_prompt: str,
) -> Dict[str, str]:
    """
    Execute one independent arbitration-panel agent.

    Every panel agent receives exactly the same original prompt.
    """

    response = call_agent(
        agent,
        original_prompt,
    )

    return {
        "agent_id": agent_id,
        "input_text": original_prompt,
        "response": response,
    }


def run_panel_in_parallel(
    original_prompt: str,
    agents: Dict[str, Any],
    agent_order: List[str],
) -> List[Dict[str, str]]:
    """
    Execute all five panel agents independently and concurrently.

    Returned results are reordered into agent_order regardless of
    completion order.
    """

    if len(agent_order) != PANEL_SIZE:
        raise ValueError(
            f"Arbitration requires exactly {PANEL_SIZE} panel agents."
        )

    if len(set(agent_order)) != PANEL_SIZE:
        raise ValueError(
            "All arbitration panel agent IDs must be unique."
        )

    missing = [
        agent_id
        for agent_id in agent_order
        if agent_id not in agents
    ]

    if missing:
        raise KeyError(
            f"Missing panel agents: {missing}"
        )

    results_by_id = {}

    with ThreadPoolExecutor(
        max_workers=PANEL_SIZE
    ) as executor:

        futures = {
            executor.submit(
                _run_single_panel_agent,
                agent_id,
                agents[agent_id],
                original_prompt,
            ): agent_id
            for agent_id in agent_order
        }

        for future in as_completed(futures):
            agent_id = futures[future]

            result = future.result()

            results_by_id[agent_id] = result

    return [
        results_by_id[agent_id]
        for agent_id in agent_order
    ]


# ============================================================
# Full arbitration
# ============================================================

def run_arbitration(
    original_prompt: str,
    agents: Dict[str, Any],
    agent_order: List[str],
    judge: Any,
    judge_id: str,
    run_id: int,
) -> Dict:
    """
    Execute one complete arbitration run.

    Stage 1:
        Five agents independently answer the same prompt.

    Stage 2:
        Claude judge receives all five complete responses
        simultaneously and produces the final response.
    """

    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError(
            "run_id must be a positive integer."
        )

    # --------------------------------------------------------
    # Five independent responses.
    # --------------------------------------------------------

    panel_outputs = run_panel_in_parallel(
        original_prompt=original_prompt,
        agents=agents,
        agent_order=agent_order,
    )

    # Verify independence of input semantics.
    for item in panel_outputs:
        if item["input_text"] != original_prompt:
            raise RuntimeError(
                "Panel-agent input mismatch detected."
            )

    # --------------------------------------------------------
    # Claude arbitration judge.
    # --------------------------------------------------------

    judge_prompt = build_judge_prompt(
        original_prompt=original_prompt,
        agent_outputs=panel_outputs,
    )

    judge_response = call_agent(
        judge,
        judge_prompt,
    )

    judge_usage = getattr(
        judge,
        "last_usage",
        None,
    )

    judge_metadata = getattr(
        judge,
        "last_response_metadata",
        None,
    )

    return {
        "architecture":
            "arbitration",

        "run_id":
            run_id,

        "panel_size":
            PANEL_SIZE,

        "agent_order":
            list(agent_order),

        "panel_outputs":
            panel_outputs,

        "judge_id":
            judge_id,

        "judge_input":
            judge_prompt,

        "judge_response":
            judge_response,

        "judge_usage":
            judge_usage,

        "judge_metadata":
            judge_metadata,

        "final_response":
            judge_response,

        "generation_config":
            dict(GENERATION_CONFIG),
    }


def run_arbitration_repeated(
    original_prompt: str,
    agents: Dict[str, Any],
    agent_order: List[str],
    judge: Any,
    judge_id: str,
    num_runs: int = 10,
) -> List[Dict]:
    """
    Execute multiple independent complete arbitration runs.

    One run means:

        five new independent panel responses
            +
        one new judge response

    This is not equivalent to requesting multiple return sequences
    from a single generation call.
    """

    if num_runs <= 0:
        raise ValueError(
            "num_runs must be positive."
        )

    results = []

    for run_id in range(
        1,
        num_runs + 1,
    ):
        results.append(
            run_arbitration(
                original_prompt=original_prompt,
                agents=agents,
                agent_order=agent_order,
                judge=judge,
                judge_id=judge_id,
                run_id=run_id,
            )
        )

    return results
