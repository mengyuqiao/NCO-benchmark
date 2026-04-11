# model_loader.py
import os
from llama_cpp import Llama


def _resolve_gpu_binding(device: int):
    """
    记录目标 GPU 信息。
    注意：
    llama.cpp Python 层不像 transformers 那样支持在 from_pretrained 时
    很自然地给每个实例单独指定 cuda:X。
    所以这里主要是保留映射信息，方便打印和调试。
    """
    return str(device)


class LlamaCppAgent:
    """
    llama.cpp 通用 Agent：
    - 使用 Llama.from_pretrained(...)
    - __call__ 保持和你原来 Agent 相同风格
    - 返回 [{"generated_text": "..."}]，尽量兼容旧 peg_core
    """

    def __init__(
        self,
        model_path: str,
        device: int,
        filename: str = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        self.device = device
        self.model_path = model_path
        self.filename = filename or os.getenv("GGUF_FILENAME", "*.gguf")
        self.n_ctx = int(os.getenv("N_CTX", n_ctx))
        self.n_gpu_layers = int(os.getenv("N_GPU_LAYERS", n_gpu_layers))
        self.verbose = verbose or (os.getenv("LLAMA_VERBOSE", "0") == "1")
        self.gpu_binding = _resolve_gpu_binding(device)

        print(
            f"[LOAD] llama.cpp model={model_path} on target_gpu={self.gpu_binding} "
            f"filename={self.filename} n_ctx={self.n_ctx} n_gpu_layers={self.n_gpu_layers}",
            flush=True
        )

        self.llm = Llama.from_pretrained(
            repo_id=self.model_path,
            filename=self.filename,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

    def format_messages(self, prompt: str):
        return [{"role": "user", "content": prompt.strip()}]

    def __call__(self, prompt: str, max_new_tokens: int = 1000, **gen_kwargs):
        """
        保持和你原先 HF Agent 尽量一致：
        输入:
            agent(prompt, max_new_tokens=..., do_sample=..., temperature=..., top_p=...)
        输出:
            [{"generated_text": "..."}]
        """
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 1.0)
        top_p = gen_kwargs.get("top_p", 1.0)

        # 若不采样，则把 temperature 压低，尽量接近 deterministic
        if not do_sample:
            temperature = 0.0

        output = self.llm.create_chat_completion(
            messages=self.format_messages(prompt),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        text = output["choices"][0]["message"]["content"]
        return [{"generated_text": text.strip()}]


def load_model_pipelines(model_names: list, device_map: dict, agent_ids: list = None):
    """
    返回 {agent_id: Agent} 字典

    参数：
    - model_names: 实际加载的 repo 列表（可重复）
    - agent_ids: 外部唯一标识；若为 None，则默认等同于 model_names
    - device_map: key 必须是 agent_id，value 是 GPU 编号

    环境变量可选：
    - GGUF_FILENAME
    - N_CTX
    - N_GPU_LAYERS
    - LLAMA_VERBOSE
    """
    if agent_ids is None:
        agent_ids = model_names

    assert len(model_names) == len(agent_ids), "model_names 与 agent_ids 数量需一致"

    agents = {}
    for repo, aid in zip(model_names, agent_ids):
        device = device_map[aid]

        print(
            f"[INIT] Loading llama.cpp model: repo={repo} as id={aid} on target_gpu={device}",
            flush=True
        )

        agents[aid] = LlamaCppAgent(
            model_path=repo,
            device=device,
        )

    return agents