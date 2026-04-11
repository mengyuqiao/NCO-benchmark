import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def fallback_pipeline(model, tokenizer, eos_token_id=None):
    """
    纯 forward() + greedy 的应急生成器：
    - 逐 token 生成，直到 max_new_tokens 或遇到 eos
    - 只返回“新生成的续写文本”（不包含 prompt）
    """
    def generator(prompt, max_new_tokens=1000, **kwargs):
        device = next(model.parameters()).device
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        attn = encoded.get("attention_mask", None)

        generated = []
        cur_ids = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(input_ids=cur_ids, attention_mask=attn) if attn is not None else model(input_ids=cur_ids)
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                generated.append(next_token)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)
                if attn is not None:
                    attn = torch.cat([attn, torch.ones_like(next_token)], dim=1)

        if len(generated) == 0:
            continuation = ""
        else:
            new_ids = torch.cat(generated, dim=1)[0]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
        return [{"generated_text": continuation.strip()}]
    return generator


def _device_map_for_single_gpu(device: int):
    # ✅ 关键：用 device_map 直接把权重 load 到指定 GPU，避免 CPU->GPU 再 .to() 的峰值
    return {"": f"cuda:{device}"}


class StandardAgent:
    """
    通用 CausalLM Agent：
    - 使用 generate()；只返回续写片段（不含 prompt）
    - __call__ 接受任意生成参数（未知参数将被忽略），避免 TypeError
    - 失败时回落到 forward() + greedy 循环生成（同样只返回续写）
    """
    def __init__(self, model_name: str, device: int):
        self.device = device
        self.model_name = model_name

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            # ✅ 关键：不要 .to(cuda)；用 device_map 直接 load 到对应 GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=_device_map_for_single_gpu(device),
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            def _gen(prompt, max_new_tokens=1000, **kwargs):
                do_sample = kwargs.get("do_sample", False)
                temperature = kwargs.get("temperature", 1.0)
                top_p = kwargs.get("top_p", 1.0)

                eos_id = self.tokenizer.eos_token_id
                pad_id = self.tokenizer.pad_token_id

                enc = self.tokenizer(prompt, return_tensors="pt")
                # inputs 放到模型所在 device
                model_device = next(self.model.parameters()).device
                enc = {k: v.to(model_device) for k, v in enc.items()}

                input_len = enc["input_ids"].shape[1]
                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
                new_tokens = gen_ids[0, input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return [{"generated_text": text.strip()}]

            self.generator = _gen

        except Exception as e:
            print(f"[Fallback] {model_name} will use forward() decoding: {e}", flush=True)
            # ✅ 仍然用 CausalLM（保证 logits 存在）
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=_device_map_for_single_gpu(device),
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.generator = fallback_pipeline(
                self.model, self.tokenizer, eos_token_id=self.tokenizer.eos_token_id
            )

    def __call__(self, prompt: str, max_new_tokens: int = 1000, **gen_kwargs):
        try:
            return self.generator(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)
        except TypeError:
            return self.generator(prompt, max_new_tokens=max_new_tokens)


class QwenAgent:
    """
    Qwen 专用：
    - 用 chat_template 包装成聊天格式
    - 不用 pipeline（避免内部 model.to(device) 导致显存峰值/OOM）
    - 只返回续写（不含 prompt）
    """
    def __init__(self, model_path: str, device: int):
        self.device = device
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ✅ 关键：直接 device_map 到指定 GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=_device_map_for_single_gpu(device),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def format_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt.strip()}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def __call__(self, prompt: str, max_new_tokens: int = 1000, **gen_kwargs):
        chat_prompt = self.format_chat(prompt)

        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 1.0)
        top_p = gen_kwargs.get("top_p", 1.0)

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        enc = self.tokenizer(chat_prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        enc = {k: v.to(model_device) for k, v in enc.items()}

        input_len = enc["input_ids"].shape[1]
        gen_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        new_tokens = gen_ids[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return [{"generated_text": text.strip()}]


def load_model_pipelines(model_names: list, device_map: dict, agent_ids: list = None):
    """
    返回 {agent_id: Agent} 字典

    参数：
    - model_names: 用于真实加载的 HF repo 列表（可重复）
    - agent_ids:  外部唯一标识（用于 agents dict key / peg_core key / CSV 列名）
                 若为 None，则默认等同于 model_names（兼容你旧用法）
    - device_map: key 必须是 agent_id（唯一），value 是 GPU 编号
    """
    if agent_ids is None:
        agent_ids = model_names

    assert len(model_names) == len(agent_ids), "model_names 与 agent_ids 数量需一致"

    agents = {}
    for repo, aid in zip(model_names, agent_ids):
        device = device_map[aid]

        # ✅ 识别 Qwen-like：依据 repo 判断（不要用 aid，aid 可能带 ::g1 后缀）
        is_qwen_like = ("Qwen" in repo) or ("Distill-Qwen" in repo)

        if is_qwen_like:
            print(f"[INIT] Loading Qwen-like model: repo={repo} as id={aid} on cuda:{device}", flush=True)
            agents[aid] = QwenAgent(repo, device=device)
        else:
            print(f"[INIT] Loading Standard model: repo={repo} as id={aid} on cuda:{device}", flush=True)
            agents[aid] = StandardAgent(repo, device=device)

        # 可选：降低后续加载峰值
        torch.cuda.empty_cache()

    return agents