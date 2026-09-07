import copy
import os
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
)


# ============================================================
# Shared helpers
# ============================================================

def _device_map_for_single_gpu(device: int):
    return {
        "": f"cuda:{device}"
    }


def _prepare_hf_input(
    tokenizer,
    prompt: str,
) -> str:
    """
    Preserve prompt semantics exactly.

    Chat templates are allowed only as model-interface formatting.
    No additional system or task instruction is inserted.
    """
    prompt = prompt.strip()

    chat_template = getattr(
        tokenizer,
        "chat_template",
        None,
    )

    if chat_template:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


# ============================================================
# Hugging Face
# ============================================================

class HFAgent:
    """
    Local Hugging Face causal-language-model agent.
    """

    provider = "huggingface"

    def __init__(
        self,
        model_name: str,
        device: int,
        agent_id: Optional[str] = None,
    ):
        self.model_name = model_name
        self.agent_id = (
            agent_id or model_name
        )
        self.device_id = int(device)

        print(
            f"[INIT] HF "
            f"agent={self.agent_id} "
            f"model={self.model_name} "
            f"device=cuda:{self.device_id}",
            flush=True,
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        )

        if (
            self.tokenizer.pad_token_id is None
            and
            self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=(
                    _device_map_for_single_gpu(
                        self.device_id
                    )
                ),
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
        )

        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                "Unexpected Hugging Face generation "
                f"arguments: {sorted(kwargs)}"
            )

        formatted_prompt = (
            _prepare_hf_input(
                self.tokenizer,
                prompt,
            )
        )

        encoded = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        model_device = next(
            self.model.parameters()
        ).device

        encoded = {
            key: value.to(model_device)
            for key, value
            in encoded.items()
        }

        input_length = (
            encoded["input_ids"]
            .shape[1]
        )

        generated_ids = (
            self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=(
                    self.tokenizer
                    .eos_token_id
                ),
                pad_token_id=(
                    self.tokenizer
                    .pad_token_id
                ),
            )
        )

        if generated_ids.shape[0] != 1:
            raise RuntimeError(
                "Expected exactly one generated "
                "sequence; received "
                f"{generated_ids.shape[0]}."
            )

        new_tokens = generated_ids[
            0,
            input_length:
        ]

        response = (
            self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        if not response:
            raise RuntimeError(
                f"{self.agent_id} returned "
                "an empty response."
            )

        return [
            {
                "generated_text": response,
                "prompt_tokens": int(input_length),
                "completion_tokens": int(new_tokens.numel()),
                "total_tokens": int(
                    input_length + new_tokens.numel()
                ),
            }
        ]


class HFMultimodalAgent:
    """
    Hugging Face multimodal model used in text-only mode.

    Used for checkpoints such as:
        Qwen/Qwen3-VL-8B-Instruct
        google/gemma-3-4b-it

    The benchmark itself is text-only. The multimodal processor is
    required because these checkpoints are multimodal architectures.

    No image content and no additional semantic instruction are added.
    """

    provider = "hf_multimodal"

    def __init__(
        self,
        model_name: str,
        device: int,
        agent_id: Optional[str] = None,
    ):
        self.model_name = model_name
        self.agent_id = agent_id or model_name
        self.device_id = int(device)

        print(
            f"[INIT] HF multimodal "
            f"agent={self.agent_id} "
            f"model={self.model_name} "
            f"device=cuda:{self.device_id}",
            flush=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=_device_map_for_single_gpu(
                self.device_id
            ),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                "Unexpected HF multimodal generation arguments: "
                f"{sorted(kwargs)}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.strip(),
                    }
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(model_device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        if generated_ids.shape[0] != 1:
            raise RuntimeError(
                "Expected exactly one generated sequence; "
                f"received {generated_ids.shape[0]}."
            )

        new_tokens = generated_ids[
            0,
            input_length:
        ]

        response = self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if not response:
            raise RuntimeError(
                f"{self.agent_id} returned an empty response."
            )

        return [
            {
                "generated_text": response,
                "prompt_tokens": int(input_length),
                "completion_tokens": int(new_tokens.numel()),
                "total_tokens": int(
                    input_length + new_tokens.numel()
                ),
            }
        ]

# ============================================================
# Anthropic Claude
# ============================================================

class AnthropicAgent:
    """
    Anthropic Messages API agent.

    API key:
        ANTHROPIC_API_KEY

    The exact Claude model ID is supplied by experiment
    configuration rather than hard-coded here.
    """

    provider = "anthropic"

    def __init__(
        self,
        model_name: str,
        agent_id: Optional[str] = None,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the "
                "'anthropic' package. Install with:\n"
                "    pip install anthropic"
            ) from exc

        api_key = os.getenv(
            "ANTHROPIC_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set."
            )

        self.model_name = model_name
        self.agent_id = (
            agent_id or model_name
        )

        self.client = Anthropic(
            api_key=api_key,
        )
        
        self.last_usage = None
        self.last_response_metadata = None

        print(
            f"[INIT] Anthropic "
            f"agent={self.agent_id} "
            f"model={self.model_name}",
            flush=True,
        )

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                "Unexpected Anthropic generation "
                f"arguments: {sorted(kwargs)}"
            )

        if not do_sample:
            raise ValueError(
                "Paper protocol requires stochastic "
                "sampling; do_sample=False is not "
                "supported for this experiment."
            )

        # Deliberately pass the paper-reported settings.
        #
        # If the selected Claude model/API version cannot
        # honor them, allow the API to fail loudly rather
        # than silently changing the protocol.
        message = (
            self.client.messages.create(
                model=self.model_name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
            )
        )

        usage = getattr(
            message,
            "usage",
            None,
        )

        self.last_usage = {
            "input_tokens": getattr(
                usage,
                "input_tokens",
                None,
            ),
            "output_tokens": getattr(
                usage,
                "output_tokens",
                None,
            ),
            "cache_creation_input_tokens": getattr(
                usage,
                "cache_creation_input_tokens",
                None,
            ),
            "cache_read_input_tokens": getattr(
                usage,
                "cache_read_input_tokens",
                None,
            ),
        }

        self.last_response_metadata = {
            "model": getattr(
                message,
                "model",
                self.model_name,
            ),
            "stop_reason": getattr(
                message,
                "stop_reason",
                None,
            ),
        }

        text_blocks = []

        for block in message.content:
            if getattr(
                block,
                "type",
                None,
            ) == "text":
                text_blocks.append(
                    block.text
                )

        response = (
            "\n".join(text_blocks)
            .strip()
        )

        if not response:
            raise RuntimeError(
                f"{self.agent_id} returned "
                "an empty response."
            )

        return [
            {
                "generated_text": response
            }
        ]


# ============================================================
# Google Gemini
# ============================================================

class GeminiAgent:
    """
    Google Gemini API agent.

    API key:
        GEMINI_API_KEY

    The exact Gemini model ID is supplied by experiment
    configuration rather than hard-coded here.
    """

    provider = "gemini"

    def __init__(
        self,
        model_name: str,
        agent_id: Optional[str] = None,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Gemini provider requires the "
                "'google-genai' package. Install with:\n"
                "    pip install google-genai"
            ) from exc

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.model_name = model_name
        self.agent_id = (
            agent_id or model_name
        )

        self._types = types

        self.client = genai.Client(
            api_key=api_key,
        )

        print(
            f"[INIT] Gemini "
            f"agent={self.agent_id} "
            f"model={self.model_name}",
            flush=True,
        )

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                "Unexpected Gemini generation "
                f"arguments: {sorted(kwargs)}"
            )

        if not do_sample:
            raise ValueError(
                "Paper protocol requires stochastic "
                "sampling; do_sample=False is not "
                "supported for this experiment."
            )

        config = (
            self._types.GenerateContentConfig(
                candidate_count=1,
                max_output_tokens=(
                    max_new_tokens
                ),
                temperature=temperature,
                top_p=top_p,
            )
        )

        result = (
            self.client.models.generate_content(
                model=self.model_name,
                contents=prompt.strip(),
                config=config,
            )
        )

        response = (
            (result.text or "")
            .strip()
        )

        if not response:
            raise RuntimeError(
                f"{self.agent_id} returned "
                "an empty response."
            )

        return [
            {
                "generated_text": response
            }
        ]


# ============================================================
# Provider-aware loader
# ============================================================

def load_agents(
    agent_specs: List[dict],
):
    """
    Load a heterogeneous five-agent panel.

    Each specification must contain:

        {
            "agent_id": "...",
            "provider": "hf|anthropic|gemini",
            "model_id": "..."
        }

    Hugging Face agents additionally require:

        {
            "device": 0
        }

    Example mixed/local agent:

        {
            "agent_id": "qwen",
            "provider": "hf",
            "model_id": "Qwen/...",
            "device": 1
        }

    Example Claude agent:

        {
            "agent_id": "claude_1",
            "provider": "anthropic",
            "model_id": "<exact Claude model ID>"
        }
    """

    if not agent_specs:
        raise ValueError(
            "agent_specs is empty."
        )

    agent_ids = [
        spec.get("agent_id")
        for spec in agent_specs
    ]

    if any(
        not x
        for x in agent_ids
    ):
        raise ValueError(
            "Every agent specification "
            "must contain agent_id."
        )

    if (
        len(set(agent_ids))
        != len(agent_ids)
    ):
        raise ValueError(
            "Agent IDs must be unique."
        )

    agents = {}
    local_model_cache = {}

    for spec in agent_specs:
        aid = spec["agent_id"]

        provider = (
            str(
                spec.get(
                    "provider",
                    ""
                )
            )
            .strip()
            .lower()
        )

        model_id = str(
            spec.get(
                "model_id",
                ""
            )
        ).strip()

        model_id_env = str(
            spec.get(
                "model_id_env",
                ""
            )
        ).strip()

        if not model_id and model_id_env:
            model_id = os.getenv(
                model_id_env,
                ""
            ).strip()

        if not model_id:
            if model_id_env:
                raise RuntimeError(
                    f"Model ID for agent {aid} is not configured. "
                    f"Set environment variable {model_id_env}."
                )

            raise ValueError(
                f"Missing model_id for agent {aid}."
            )

        if not model_id:
            raise ValueError(
                f"Missing model_id for "
                f"agent {aid}."
            )

        if provider in {
            "hf",
            "huggingface",
        }:
            if "device" not in spec:
                raise ValueError(
                    f"HF agent {aid} requires "
                    "a device assignment."
                )

            device = int(
                spec["device"]
            )

            cache_key = (
                "hf",
                model_id,
                device,
            )

            if cache_key in local_model_cache:
                agents[aid] = copy.copy(
                    local_model_cache[
                        cache_key
                    ]
                )
                agents[aid].agent_id = aid

                print(
                    f"[REUSE] HF "
                    f"agent={aid} "
                    f"model={model_id} "
                    f"device=cuda:{device}",
                    flush=True,
                )

            else:
                agents[aid] = HFAgent(
                    model_name=model_id,
                    device=device,
                    agent_id=aid,
                )

                local_model_cache[
                    cache_key
                ] = agents[aid]

                torch.cuda.empty_cache()

        elif provider in {
            "hf_multimodal",
            "huggingface_multimodal",
        }:
            if "device" not in spec:
                raise ValueError(
                    f"HF multimodal agent {aid} "
                    "requires a device assignment."
                )

            device = int(
                spec["device"]
            )

            cache_key = (
                "hf_multimodal",
                model_id,
                device,
            )

            if cache_key in local_model_cache:
                agents[aid] = copy.copy(
                    local_model_cache[
                        cache_key
                    ]
                )
                agents[aid].agent_id = aid

                print(
                    f"[REUSE] HF multimodal "
                    f"agent={aid} "
                    f"model={model_id} "
                    f"device=cuda:{device}",
                    flush=True,
                )

            else:
                agents[aid] = HFMultimodalAgent(
                    model_name=model_id,
                    device=device,
                    agent_id=aid,
                )

                local_model_cache[
                    cache_key
                ] = agents[aid]

                torch.cuda.empty_cache()

        elif provider in {
            "anthropic",
            "claude",
        }:
            agents[aid] = (
                AnthropicAgent(
                    model_name=model_id,
                    agent_id=aid,
                )
            )

        elif provider in {
            "gemini",
            "google",
        }:
            agents[aid] = (
                GeminiAgent(
                    model_name=model_id,
                    agent_id=aid,
                )
            )

        else:
            raise ValueError(
                f"Unsupported provider "
                f"'{provider}' for agent "
                f"{aid}."
            )

    return agents


# ============================================================
# Backward-compatible HF loader
# ============================================================

def load_model_pipelines(
    model_names: List[str],
    device_map: Dict[str, int],
    agent_ids: Optional[
        List[str]
    ] = None,
):
    """
    Backward-compatible loader used by the current
    rolling-review runner.

    This wrapper loads Hugging Face agents only.

    It will be replaced by load_agents() once
    main_multi_model.py reads panel configuration files.
    """

    if agent_ids is None:
        agent_ids = list(
            model_names
        )

    if (
        len(model_names)
        != len(agent_ids)
    ):
        raise ValueError(
            "model_names and agent_ids "
            "must have identical lengths."
        )

    specs = []

    for repo, aid in zip(
        model_names,
        agent_ids,
    ):
        if aid not in device_map:
            raise KeyError(
                f"Missing device for "
                f"{aid}."
            )

        specs.append({
            "agent_id": aid,
            "provider": "hf",
            "model_id": repo,
            "device": int(
                device_map[aid]
            ),
        })

    return load_agents(specs)
