import torch
from transformers import AutoTokenizer, pipeline


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
            new_ids = torch.cat(generated, dim=1)[0]  # (1, T) -> (T,)
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
        return [{"generated_text": continuation.strip()}]
    return generator


class StandardAgent:
    """
    通用 CausalLM Agent：
    - 使用 generate()；只返回续写片段（不含 prompt）
    - __call__ 接受任意生成参数（未知参数将被忽略），避免 TypeError
    - 失败时回落到 forward() + greedy 循环生成（同样只返回续写）
    """
    def __init__(self, model_name: str, device: int):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # pad_token 兜底到 eos，避免部分模型报错
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(f"cuda:{device}")

            def _gen(prompt, max_new_tokens=1000, **kwargs):
                # 支持的常用参数；未知参数自动忽略
                do_sample = kwargs.get("do_sample", False)
                temperature = kwargs.get("temperature", 1.0)
                top_p = kwargs.get("top_p", 1.0)
                eos_id = self.tokenizer.eos_token_id
                pad_id = self.tokenizer.pad_token_id

                enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_ids = enc["input_ids"]
                input_len = input_ids.shape[1]

                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id
                )
                # 只取续写部分，防止回显 prompt
                new_tokens = gen_ids[0, input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return [{"generated_text": text.strip()}]

            self.generator = _gen

        except Exception as e:
            print(f"[Fallback] {model_name} will use forward() decoding: {e}")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(f"cuda:{device}")
            self.generator = fallback_pipeline(
                self.model, self.tokenizer, eos_token_id=self.tokenizer.eos_token_id
            )

    def __call__(self, prompt: str, max_new_tokens: int = 1000, **gen_kwargs):
        # 吞掉未知参数，保持接口稳健
        try:
            return self.generator(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)
        except TypeError:
            # 某些内部生成器不接受这些参数，则去掉再试
            safe_kwargs = {}
            return self.generator(prompt, max_new_tokens=max_new_tokens, **safe_kwargs)


class QwenAgent:
    """
    Qwen 专用：
    - 用 chat_template 包装成聊天格式
    - pipeline(text-generation) 强制 return_full_text=False（只要续写）
    - __call__ 同样吞掉未知生成参数
    """
    def __init__(self, model_path: str, device: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype="auto",
        )

    def format_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt.strip()}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, prompt: str, max_new_tokens: int = 1000, **gen_kwargs):
        chat_prompt = self.format_chat(prompt)
        # 常用可控参数；未知参数不传给 pipeline，避免 TypeError
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 1.0)
        top_p = gen_kwargs.get("top_p", 1.0)

        outputs = self.pipe(
            chat_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False  # 关键：只返回续写
        )
        # 新版 pipeline 的 key 通常是 generated_text；兜底拿 text
        text = outputs[0].get("generated_text") or outputs[0].get("text") or ""
        return [{"generated_text": text.strip()}]


def load_model_pipelines(model_names: list, device_map: dict):
    """
    返回 {model_name: Agent} 字典
    - Qwen 用 QwenAgent（chat 模式；只返回续写）
    - 其它用 StandardAgent（只返回续写；失败回退 forward() 解码）
    """
    agents = {}
    for name in model_names:
        device = device_map[name]
        if "Qwen" in name:
            print(f"[INIT] Loading Qwen model: {name} on cuda:{device}")
            agents[name] = QwenAgent(name, device=device)
        else:
            print(f"[INIT] Loading Standard model: {name} on cuda:{device}")
            agents[name] = StandardAgent(name, device=device)
    return agents
