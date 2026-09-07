from typing import Any, Callable, Dict, List


# ============================================================
# Paper-level configuration
# ============================================================

PANEL_SIZE = 5

GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
}


# ============================================================
# Output normalization
# ============================================================

def _content_to_text(value: Any) -> str:
    """
    Convert common model/API response objects into plain text.

    This function does NOT classify the output as yes/no.
    Raw model content is preserved.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        # Chat-style generated_text may look like:
        #
        # [
        #   {"role": "user", ...},
        #   {"role": "assistant", "content": "..."}
        # ]
        #
        # Prefer the final message containing content.
        for item in reversed(value):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])

        return "\n".join(str(item) for item in value)

    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])

        if "text" in value:
            return str(value["text"])

        return str(value)

    return str(value)


def extract_response_text(raw: Any) -> str:
    """
    Normalize one model call into a single complete response string.

    Supported formats include:
      - plain strings
      - Hugging Face list[dict]
      - Hugging Face chat generated_text
      - OpenAI-style choices
      - generic dictionaries

    No yes/no parsing or fallback classification occurs here.
    """
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw.strip()

    if isinstance(raw, dict):
        if "generated_text" in raw:
            return _content_to_text(
                raw["generated_text"]
            ).strip()

        if "text" in raw:
            return str(raw["text"]).strip()

        if "choices" in raw:
            choices = raw.get("choices") or []

            if choices:
                first = choices[0]

                if isinstance(first, dict):
                    if "message" in first:
                        return _content_to_text(
                            first["message"]
                        ).strip()

                    if "text" in first:
                        return str(
                            first["text"]
                        ).strip()

        return str(raw).strip()

    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return ""

        # Each agent call must produce exactly one response.
        if len(raw) != 1:
            raise RuntimeError(
                "Expected exactly one model response per agent call, "
                f"but received {len(raw)} responses."
            )

        return extract_response_text(raw[0])

    return str(raw).strip()


# ============================================================
# Model execution
# ============================================================

def call_agent(
    agent: Callable,
    prompt: str,
) -> str:
    """
    Execute exactly one stochastic model generation using the
    inference settings reported in the manuscript.
    """
    raw = agent(
        prompt,
        **GENERATION_CONFIG,
    )

    response = extract_response_text(raw)

    if not response:
        raise RuntimeError(
            "Model returned an empty response."
        )

    return response


# ============================================================
# Rolling review
# ============================================================

def build_review_prompt(
    original_prompt: str,
    previous_response: str,
) -> str:
    """
    Build the input for Agents 2-5.

    The COMPLETE output of the immediately preceding agent is
    passed without truncation or answer-only extraction.
    """
    return (
        "Original prompt:\n"
        "================ ORIGINAL PROMPT ================\n"
        f"{original_prompt.strip()}\n"
        "=================================================\n\n"

        "Complete response from the previous agent:\n"
        "============= PREVIOUS AGENT RESPONSE ============\n"
        f"{previous_response}\n"
        "=================================================\n\n"

        "Critique, revise, or affirm the previous agent's response. "
        "Preserve reasoning that is correct and correct any errors "
        "or unsupported conclusions you identify. Return a complete "
        "updated response that answers the original prompt."
    )


def run_rolling_review(
    original_prompt: str,
    agents: Dict[str, Callable],
    agent_order: List[str],
    run_id: int,
) -> Dict[str, Any]:
    """
    Execute one independent five-agent rolling-review run.

    Parameters
    ----------
    original_prompt:
        Exact benchmark prompt for one target question.

    agents:
        Mapping from agent ID to callable model interface.

    agent_order:
        Ordered list of exactly five agent IDs.

    run_id:
        Independent run number.

    Returns
    -------
    dict
        Full reproducibility trace containing every agent's input
        and complete output, plus the final Agent-5 response.
    """
    if len(agent_order) != PANEL_SIZE:
        raise ValueError(
            f"Rolling review requires exactly {PANEL_SIZE} agents; "
            f"received {len(agent_order)}."
        )

    if len(set(agent_order)) != PANEL_SIZE:
        # Repeated model instances are allowed, but their agent IDs
        # must remain unique (e.g. claude_1 ... claude_5).
        raise ValueError(
            "Each panel position must have a unique agent ID. "
            "Repeated model instances should use distinct IDs."
        )

    missing = [
        agent_id
        for agent_id in agent_order
        if agent_id not in agents
    ]

    if missing:
        raise KeyError(
            f"Missing agent implementations: {missing}"
        )

    trace: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Agent 1
    #
    # Receives the original benchmark prompt only.
    # --------------------------------------------------------

    first_agent_id = agent_order[0]

    first_response = call_agent(
        agents[first_agent_id],
        original_prompt,
    )

    trace.append({
        "run_id": run_id,
        "agent_position": 1,
        "agent_id": first_agent_id,
        "input_type": "original_prompt",
        "input_text": original_prompt,
        "previous_response": None,
        "response": first_response,
    })

    previous_response = first_response

    # --------------------------------------------------------
    # Agents 2-5
    #
    # Each receives the complete response from the immediately
    # preceding agent.
    # --------------------------------------------------------

    for position, agent_id in enumerate(
        agent_order[1:],
        start=2,
    ):
        review_prompt = build_review_prompt(
            original_prompt=original_prompt,
            previous_response=previous_response,
        )

        response = call_agent(
            agents[agent_id],
            review_prompt,
        )

        trace.append({
            "run_id": run_id,
            "agent_position": position,
            "agent_id": agent_id,
            "input_type": "review_previous_response",
            "input_text": review_prompt,
            "previous_response": previous_response,
            "response": response,
        })

        previous_response = response

    # --------------------------------------------------------
    # Manuscript definition:
    # the final agent's output is the panel answer.
    # --------------------------------------------------------

    final_response = previous_response

    return {
        "architecture": "rolling_review",
        "run_id": run_id,
        "panel_size": PANEL_SIZE,
        "agent_order": list(agent_order),
        "trace": trace,
        "final_agent_id": agent_order[-1],
        "final_response": final_response,
    }


def run_rolling_review_repeated(
    original_prompt: str,
    agents: Dict[str, Callable],
    agent_order: List[str],
    num_runs: int = 10,
) -> List[Dict[str, Any]]:
    """
    Execute independent repetitions of the complete five-agent chain.

    One run is:

        Agent1 -> Agent2 -> Agent3 -> Agent4 -> Agent5

    It is NOT ten sampled completions from a single agent call.
    """
    if num_runs <= 0:
        raise ValueError(
            "num_runs must be a positive integer."
        )

    results = []

    for run_id in range(1, num_runs + 1):
        result = run_rolling_review(
            original_prompt=original_prompt,
            agents=agents,
            agent_order=agent_order,
            run_id=run_id,
        )

        results.append(result)

    return results
