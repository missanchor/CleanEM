"""
Modal encoder implementations.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseEncoder


def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _default_mcm_cache_key(
    *,
    dataset_name: Optional[str],
    num_rows: int,
    num_cols: int,
    model_name_or_path: Optional[str],
    max_new_tokens: int,
    generation_backend: str,
) -> str:
    # Keep it simple + readable; stable across runs.
    safe_dataset = (dataset_name or "dataset").replace("/", "_")
    safe_model = (model_name_or_path or "model").replace("/", "_")
    return f"mcm_text_cache__{safe_dataset}__r{num_rows}_c{num_cols}__{generation_backend}__t{int(max_new_tokens)}__{safe_model}.pt"


def _process_chunk_worker(
    chunk_data,
    text_encoder,
    text_processor,
    max_new_tokens: int = 10,
    cache_device: str = "cpu",
):
    """
    Worker函数：处理一个chunk的prompts
    注意：这是模块级别的函数，可以被pickle用于multiprocessing
    """
    chunk_prompts, chunk_indices = chunk_data
    chunk_embeddings = []

    for i, prompt in enumerate(chunk_prompts):
        try:
            # 使用text_encoder实例的方法
            embedding = text_encoder.generate_and_encode(
                prompt=prompt,
                text_processor=text_processor,
                max_new_tokens=max_new_tokens,
                device=cache_device,
            )
            chunk_embeddings.append((embedding, chunk_indices[i]))
        except Exception as e:
            # Fallback to encoding
            text_inputs = text_processor.process(prompt)
            text_inputs = {k: v.to(cache_device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                H_text = text_encoder(text_inputs)
                if H_text.dim() == 3:
                    last_token_emb = H_text[:, -1, :].squeeze(0).unsqueeze(0).cpu()
                else:
                    last_token_emb = H_text.unsqueeze(0).cpu()
                chunk_embeddings.append((last_token_emb, chunk_indices[i]))
    return chunk_embeddings



class StructureAwareTransformer(BaseEncoder):
    """
    Transformer encoder that leverages row/column structure via custom masks.
    """

    def __init__(
        self,
        d_cell: int,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_projection = nn.Linear(d_cell, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.d_model = d_model

    def _create_structure_mask(
        self,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Handle both batch and single-row cases
        if row_indices.dim() == 1:
            # Single row case - add batch dimension
            row_indices = row_indices.unsqueeze(0)
            col_indices = col_indices.unsqueeze(0)

        batch_size, num_cols = row_indices.shape
        row_i = row_indices.unsqueeze(2)
        row_j = row_indices.unsqueeze(1)
        col_i = col_indices.unsqueeze(2)
        col_j = col_indices.unsqueeze(1)

        same_row = row_i == row_j
        same_col = col_i == col_j
        can_attend = same_row | same_col

        mask = torch.zeros_like(can_attend, dtype=torch.float)
        mask = mask.masked_fill(~can_attend, float("-inf"))
        return mask

    def forward(self, tabular_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        cell_embeddings = tabular_inputs["cell_embeddings"]
        row_indices = tabular_inputs["row_indices"]
        col_indices = tabular_inputs["col_indices"]

        x = self.cell_projection(cell_embeddings)
        attn_mask = self._create_structure_mask(row_indices, col_indices)

        # Simplified: assume shared structure within the batch.
        attn_mask_2d = attn_mask[0]

        H_table = self.transformer_encoder(
            x,
            mask=attn_mask_2d,
        )
        return H_table


class PretrainedTextEncoder(BaseEncoder):
    """
    Text encoder that either:
    1. Wraps a pretrained HF/ModelScope model (kept frozen by default); or
    2. Falls back to a lightweight Transformer when no checkpoint is provided.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        *,
        model_name_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        pretrained_model_kwargs: Optional[Dict[str, Any]] = None,
        output_proj_dim: Optional[int] = None,
        freeze_pretrained: bool = True,
        freeze_base_model: Optional[bool] = None,
        adapter_hidden_dim: Optional[int] = None,
        adapter_dropout: float = 0.0,
    ):
        """
        Args:
            vocab_size: Size of the synthetic vocabulary (ignored when loading a
                pretrained model).
            d_model: Hidden size of the lightweight encoder.
            nhead: Number of attention heads for the lightweight encoder.
            num_layers: Number of transformer layers for the lightweight encoder.
            max_seq_len: Maximum sequence length supported by the lightweight encoder.
            dropout: Dropout probability applied before the transformer or on top
                of the pretrained model outputs.
            model_name_or_path: Local path or identifier of a pretrained model.
            trust_remote_code: Passed to HuggingFace `from_pretrained`.
            pretrained_model_kwargs: Extra keyword arguments for `from_pretrained`.
            output_proj_dim: Optional projection dimension; leave as None to use
                the raw pretrained hidden size directly (no adapter).
            freeze_pretrained: Whether to freeze the backbone weights when loading
                a pretrained checkpoint.
            freeze_base_model: Deprecated alias for `freeze_pretrained`.
            adapter_hidden_dim: Size of the optional lightweight adapter. When
                provided, a small two-layer MLP is added on top of the encoder
                outputs (trainable even if the base model is frozen).
            adapter_dropout: Dropout rate applied inside the adapter module.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_transformers_model = model_name_or_path is not None
        if freeze_base_model is not None and freeze_base_model != freeze_pretrained:
            raise ValueError(
                "Received both `freeze_pretrained` and legacy `freeze_base_model` "
                "with conflicting values. Please keep them consistent."
            )
        if freeze_base_model is not None:
            freeze_pretrained = freeze_base_model
        self.freeze_pretrained = freeze_pretrained
        self.output_projection: Optional[nn.Linear] = None
        self.adapter: Optional[nn.Module] = None

        # Placeholders to keep attribute access consistent.
        self.model: Optional[nn.Module] = None
        self.token_embedding: Optional[nn.Embedding] = None
        self.position_embedding: Optional[nn.Embedding] = None
        self.transformer: Optional[nn.TransformerEncoder] = None

        if self.use_transformers_model:
            load_kwargs: Dict[str, Any]
            if pretrained_model_kwargs is None:
                load_kwargs = {}
            else:
                load_kwargs = dict(pretrained_model_kwargs)
            load_kwargs.setdefault("trust_remote_code", trust_remote_code)

            try:
                from transformers import AutoModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "The `transformers` package is required to load pretrained "
                    "text encoders. Install it via `pip install transformers`."
                ) from exc

            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )

            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.model.config, "d_model", None)
            if hidden_size is None:
                raise ValueError(
                    "Unable to determine the hidden size of the pretrained model. "
                    "Please ensure the config exposes `hidden_size` or `d_model`."
                )

            target_dim = output_proj_dim or hidden_size
            if output_proj_dim is not None and output_proj_dim != hidden_size:
                self.output_projection = nn.Linear(hidden_size, target_dim)
                if self.freeze_pretrained:
                    for param in self.output_projection.parameters():
                        param.requires_grad = False
            self.d_model = target_dim

            if self.freeze_pretrained and self.model is not None:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.d_model = d_model

        if adapter_hidden_dim is not None:
            if adapter_hidden_dim <= 0:
                raise ValueError("`adapter_hidden_dim` must be a positive integer.")
            self.adapter = nn.Sequential(
                nn.Linear(self.d_model, adapter_hidden_dim),
                nn.ReLU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden_dim, self.d_model),
            )

    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_transformers_model:
            if self.model is None:
                raise RuntimeError("Pretrained model not initialized.")

            model_kwargs: Dict[str, torch.Tensor] = {}
            for key in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "position_ids",
                "inputs_embeds",
            ):
                if key in text_inputs:
                    model_kwargs[key] = text_inputs[key]

            if self.freeze_pretrained:
                with torch.no_grad():
                    outputs = self.model(**model_kwargs)
            else:
                outputs = self.model(**model_kwargs)

            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]

            hidden_states = self.dropout(hidden_states)
            if self.output_projection is not None:
                hidden_states = self.output_projection(hidden_states)
            if self.adapter is not None:
                hidden_states = hidden_states + self.adapter(hidden_states)
            return hidden_states

        if self.token_embedding is None or self.position_embedding is None or self.transformer is None:
            raise RuntimeError("Lightweight encoder components not initialized.")

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        batch_size, seq_len = input_ids.shape

        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        embeddings = self.dropout(embeddings)

        src_key_padding_mask = attention_mask == 0

        H_text = self.transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )
        if self.adapter is not None:
            H_text = H_text + self.adapter(H_text)
        return H_text

    def train(self, mode: bool = True):
        if self.use_transformers_model and self.freeze_pretrained:
            super().train(False)
            if self.model is not None:
                self.model.eval()
            if self.output_projection is not None and all(not p.requires_grad for p in self.output_projection.parameters()):
                self.output_projection.eval()
            return self
        return super().train(mode)

    def _ensure_text_generator(self, device: Optional[str] = None) -> str:
        """
        Lazily load the causal LM (e.g., Qwen) used for prompt generation and move it
        onto the requested device.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The `transformers` package is required for text generation. "
                "Install it via `pip install transformers`."
            ) from exc

        model_path = "/mnt/data/welkinni/Qwen2.5-0.5B-Instruct/qwen/Qwen2.5-0.5B-Instruct"
        if not hasattr(self, "_generator") or self._generator is None:
            print(f"Loading generator model from {model_path}...")
            self._generator_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self._generator = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self._generator_device = None
            self._generator.eval()
            print("✓ Generator model loaded")

        encoder_device = next(self.parameters()).device
        generator_device = device or encoder_device
        if getattr(self, "_generator_device", None) != generator_device:
            self._generator.to(generator_device)
            self._generator_device = generator_device
        return generator_device

    def _ensure_vllm_local_llm(self, model: str):
        """
        Lazily create and cache an in-process vLLM `LLM` instance for text generation.
        """
        if not model:
            raise ValueError("vllm_model is required for generation_backend='vllm'")
        cached_model = getattr(self, "_vllm_local_model", None)
        cached_llm = getattr(self, "_vllm_local_llm", None)
        if cached_llm is None or cached_model != model:
            try:
                from vllm import LLM  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "generation_backend='vllm' requires the `vllm` package. Install it via `pip install vllm`."
                ) from exc
            # vLLM manages its own device placement / KV cache; we just keep a handle here.
            self._vllm_local_llm = LLM(
                model=model,
                trust_remote_code=True,
            )
            self._vllm_local_model = model
        return self._vllm_local_llm

    def _vllm_local_completions(
        self,
        *,
        model: str,
        prompts: List[str],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> List[str]:
        """
        Run in-process generation via vLLM python API.

        Returns generated continuations (not including the prompt), aligned with `prompts`.
        """
        if not prompts:
            return []
        llm = self._ensure_vllm_local_llm(model)
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "generation_backend='vllm' requires the `vllm` package (SamplingParams missing)."
            ) from exc

        sampling_params = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            n=1,
        )

        req_outputs = llm.generate(prompts, sampling_params)
        # vLLM returns outputs in the same order as input prompts.
        out_texts: List[str] = []
        for r in req_outputs:
            text = ""
            try:
                if getattr(r, "outputs", None):
                    # Take top-1 sample.
                    text = getattr(r.outputs[0], "text", "") or ""
            except Exception:
                text = ""
            out_texts.append(text)

        # Robustness: if vLLM returns fewer items, pad; if returns empty, fall back per-slot.
        if len(out_texts) < len(prompts):
            out_texts.extend([""] * (len(prompts) - len(out_texts)))
        out_texts = out_texts[: len(prompts)]
        for i in range(len(out_texts)):
            if not out_texts[i].strip():
                out_texts[i] = prompts[i]
        return out_texts

    def _encode_text_batch(
        self,
        texts: List[str],
        text_processor,
    ) -> List[torch.Tensor]:
        """
        Encode a batch of texts with the current text encoder and return per-sample
        last-token embeddings (each shaped [1, d_model]).
        """
        if not texts:
            return []

        encoder_device = next(self.parameters()).device
        processed_inputs = [text_processor.process(text) for text in texts]

        tensor_keys = [
            key for key, value in processed_inputs[0].items() if isinstance(value, torch.Tensor)
        ]
        batch_inputs: Dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            stacked = torch.stack([sample[key] for sample in processed_inputs], dim=0)
            batch_inputs[key] = stacked.to(encoder_device)

        with torch.no_grad():
            response_encoding = self(batch_inputs)

        if response_encoding.dim() == 3:
            last_hidden = response_encoding[:, -1, :]
        elif response_encoding.dim() == 2:
            last_hidden = response_encoding
        else:
            last_hidden = response_encoding.unsqueeze(0)

        last_hidden = last_hidden.detach().cpu()
        return [last_hidden[i].unsqueeze(0) for i in range(last_hidden.shape[0])]

    def _batch_generate_and_encode(
        self,
        prompts: List[str],
        text_processor,
        max_new_tokens: int = 10,
        device: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """
        Vectorized variant of `generate_and_encode` that handles multiple prompts at once.
        """
        if not prompts:
            return []

        generator_device = self._ensure_text_generator(device)
        tokenizer_inputs = self._generator_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(generator_device)

        attention_mask = tokenizer_inputs.get("attention_mask")
        if attention_mask is not None:
            prompt_lengths = attention_mask.sum(dim=1)
        else:
            input_ids = tokenizer_inputs["input_ids"]
            prompt_lengths = torch.full(
                (input_ids.size(0),),
                fill_value=input_ids.size(1),
                device=input_ids.device,
            )

        with torch.no_grad():
            outputs = self._generator.generate(
                **tokenizer_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._generator_tokenizer.eos_token_id,
            )

        texts_to_encode: List[str] = []
        prompt_lengths_list = prompt_lengths.tolist()
        for idx in range(outputs.size(0)):
            start = int(prompt_lengths_list[idx])
            generated_ids = outputs[idx][start:]
            generated_text = self._generator_tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            if not generated_text.strip():
                generated_text = prompts[idx]
            texts_to_encode.append(generated_text)

        return self._encode_text_batch(texts_to_encode, text_processor)

    def generate_and_encode(
        self,
        prompt: str,
        text_processor,
        max_new_tokens: int = 10,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate text response from prompt and return the last token's embedding.

        This method:
        1. Uses a generative model (e.g., Qwen) to generate a text response to the prompt
        2. Tokenizes the generated response
        3. Encodes the response and returns the last token's embedding

        Args:
            prompt: Input prompt string
            text_processor: TextProcessor instance for tokenization
            max_new_tokens: Maximum number of tokens to generate
            device: Optional device to use. If None, uses the encoder's device.

        Returns:
            torch.Tensor: Last token embedding of the generated response, shape [d_model]

        Note:
            This method requires a decoder-capable model (e.g., Qwen, LLaMA).
            Default model: /mnt/data/welkinni/Qwen2.5-0.5B-Instruct/qwen/Qwen2.5-0.5B-Instruct
        """
        # Move to device + lazily load generator
        if device is None:
            device = next(self.parameters()).device
        generator_device = self._ensure_text_generator(device)

        # Tokenize prompt
        inputs = self._generator_tokenizer(prompt, return_tensors="pt").to(generator_device)

        # Generate response
        with torch.inference_mode():
            outputs = self._generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._generator_tokenizer.eos_token_id,
            )

        # Decode generated text (skip the prompt)
        generated_text = self._generator_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # If no text generated, fall back to encoding the last token of the input
        if not generated_text.strip():
            # Fallback: use last token of the input encoding
            encoder_device = next(self.parameters()).device
            with torch.no_grad():
                processed_inputs = {
                    k: (
                        v.unsqueeze(0).to(encoder_device)
                        if isinstance(v, torch.Tensor) and v.dim() == 1
                        else v.to(encoder_device)
                    )
                    for k, v in text_processor.process(prompt).items()
                    if isinstance(v, torch.Tensor)
                }
                full_encoding = self.model(**processed_inputs)
                if hasattr(full_encoding, 'last_hidden_state'):
                    last_hidden = full_encoding.last_hidden_state[0, -1, :]
                else:
                    last_hidden = full_encoding[0][0, -1, :]
            return last_hidden.cpu()

        # Encode the generated response
        batch_embeddings = self._encode_text_batch([generated_text], text_processor)
        if not batch_embeddings:
            raise ValueError("Failed to encode generated response.")
        return batch_embeddings[0]

    def precompute_embeddings_for_mcm(
        self,
        dirty_rows: List[List[Any]],
        column_names: Optional[List[str]],
        text_descriptions: List[str],
        text_processor,
        device: str = "cuda:0",
        batch_size: int = 32,
        use_last_token: bool = False,
        generate_response: bool = False,
        max_new_tokens: int = 10,
        generation_row_batch_size: int = 1,
        generation_backend: str = "hf",  # "hf" | "vllm"
        generation_prompt_batch_size: int = 256,  # micro-batch size in #prompts (not #rows)
        vllm_model: Optional[str] = None,
        cache_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        force_recompute_cache: bool = False,
        use_multiprocessing: bool = False,  # 多进程通常不适合 torch+tokenizers；默认关闭
        num_workers: Optional[int] = None,
    ) -> Dict:
        """
        Precompute text embeddings for MCM experiment with intelligent per-column prompt handling.

        This method automatically:
        1. Detects if per-column prompts are needed (for MCM)
        2. Chooses between generate_and_encode vs standard encoding
        3. Caches embeddings efficiently

        Args:
            dirty_rows: List of dirty rows
            column_names: Column names for per-column prompts
            text_descriptions: Text descriptions for each row
            text_processor: TextProcessor instance
            device: Device to use
            batch_size: Batch size for processing
            use_last_token: Whether to extract last token embedding
            generate_response: Whether to use generate_and_encode
            max_new_tokens: Max tokens for generation
            generation_row_batch_size: Number of table rows to process per generation batch

        Returns:
            Dict mapping (row_idx, col_idx) or row_idx to cached embedding
        """
        from tqdm import tqdm
        from .runner.experiments.embeddings import _precompute_text_embeddings

        # Auto-detect if we need per-column prompts
        per_column_prompts = None
        if column_names is not None:
            per_column_prompts = []
            for idx, row in enumerate(dirty_rows):
                # Serialize row data
                row_serialized_parts = []
                for c in range(len(column_names) if column_names else len(row)):
                    col_name = column_names[c] if column_names and c < len(column_names) else f'Col{c}'
                    row_serialized_parts.append(f"{col_name}: {row[c]}")
                row_serialized = ", ".join(row_serialized_parts)

                # Build per-column prompts for this row
                row_prompts = {}
                for col_idx, col_name in enumerate(column_names):
                    col_prompt = (
                        f"Instruction: Check for errors (missing, typo, column pattern violations, rule violations) in column '{col_name}' for this record. "
                        f"Record: {row_serialized}. "
                        f"The error checking analysis result is:"
                    )
                    row_prompts[col_idx] = col_prompt
                per_column_prompts.append(row_prompts)
        else:
            raise ValueError("column_names is required for MCM experiment")

        generation_row_batch_size = max(1, int(generation_row_batch_size))
        generation_prompt_batch_size = max(1, int(generation_prompt_batch_size))

        if int(max_new_tokens) >= 512:
            print(
                f"  ⚠ max_new_tokens={int(max_new_tokens)} 非常大，生成会极慢。"
                "通常 16~128 就够用（或配合 stop/更短输出）。"
            )

        if generate_response:
            # Use generate_and_encode for each prompt
            print("  → 使用生成式模型生成响应并编码...")
            cached_embeddings = {}
            cache_device = device
            num_rows = len(per_column_prompts)

            generation_backend = (generation_backend or "hf").lower().strip()
            if generation_backend not in {"hf", "vllm"}:
                raise ValueError(f"Unsupported generation_backend={generation_backend!r}, expected 'hf' or 'vllm'")

            # Disk cache (optional)
            if cache_path is None:
                num_cols = len(column_names) if column_names is not None else 0
                cache_filename = _default_mcm_cache_key(
                    dataset_name=dataset_name,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    model_name_or_path=getattr(self, "model_name_or_path", None),
                    max_new_tokens=int(max_new_tokens),
                    generation_backend=generation_backend,
                )
                cache_path = os.path.join(os.getcwd(), cache_filename)

            if cache_path and (not force_recompute_cache) and os.path.exists(cache_path):
                try:
                    loaded = torch.load(cache_path, map_location="cpu")
                    if isinstance(loaded, dict) and len(loaded) > 0:
                        print(f"✓ 从磁盘加载文本embedding缓存: {cache_path}（{len(loaded)} 条）")
                        return loaded
                except Exception as e:
                    print(f"  ⚠ 读取缓存失败，将重新计算: {cache_path} ({e})")

            # vLLM path: no multiprocessing (GPU engine + model handles don't mix well with mp)
            if generation_backend == "vllm":
                use_multiprocessing = False
            else:
                # 预加载模型避免重复加载
                self._ensure_text_generator(cache_device)

            if use_multiprocessing and num_rows > 1:
                # 多进程版本：避免tokenizers死锁
                print(f"  → 启用多进程并行处理...")
                try:
                    import multiprocessing
                    from multiprocessing import Pool

                    # 设置tokenizers并行环境变量
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"

                    # 计算worker数量
                    if num_workers is None:
                        num_workers = min(multiprocessing.cpu_count(), 4)  # 限制最大4个worker

                    # 准备所有prompts和索引映射
                    all_prompts: List[str] = []
                    all_index_mapping: List[Tuple[int, int]] = []

                    for row_idx in range(num_rows):
                        row_prompts = per_column_prompts[row_idx]
                        col_indices = sorted(row_prompts.keys())
                        for col_idx in col_indices:
                            all_prompts.append(row_prompts[col_idx])
                            all_index_mapping.append((row_idx, col_idx))

                    # 将prompts分成多个chunk，每个worker处理一个chunk
                    chunk_size = max(1, len(all_prompts) // num_workers)
                    chunks = []
                    for i in range(0, len(all_prompts), chunk_size):
                        chunk_prompts = all_prompts[i:i+chunk_size]
                        chunk_indices = all_index_mapping[i:i+chunk_size]
                        chunks.append((chunk_prompts, chunk_indices))

                    print(f"  → 分为 {len(chunks)} 个chunk，每个约 {chunk_size} 个prompts")

                    # 定义worker函数和初始化函数（模块级别，可被pickle）
                    from functools import partial

                    def worker_init():
                        """Worker初始化：设置环境变量避免死锁"""
                        os.environ["TOKENIZERS_PARALLELISM"] = "false"

                    # 创建partial函数，绑定self和必要参数
                    process_chunk_func = partial(
                        _process_chunk_worker,
                        text_encoder=self,
                        text_processor=text_processor,
                        max_new_tokens=max_new_tokens,
                        cache_device=cache_device,
                    )

                    # 使用multiprocessing并行处理
                    with Pool(processes=num_workers, initializer=worker_init) as pool:
                        results = list(tqdm(
                            pool.imap(process_chunk_func, chunks),
                            total=len(chunks),
                            desc="并行生成响应",
                            ncols=100
                        ))

                    # 合并结果
                    for chunk_results in results:
                        for embedding, (row_idx, col_idx) in chunk_results:
                            cached_embeddings[(row_idx, col_idx)] = embedding

                    print(f"✓ 多进程处理完成，共缓存 {len(cached_embeddings)} 条")
                    return cached_embeddings

                except Exception as e:
                    print(f"  ⚠ 多进程处理失败，回退到串行处理: {e}")
                    # 如果多进程失败，回退到串行处理
                    use_multiprocessing = False

            if not use_multiprocessing:
                # 串行版本：原始实现
                total_batches = math.ceil(num_rows / generation_row_batch_size)
                batch_iter = range(0, num_rows, generation_row_batch_size)

                for batch_start in tqdm(batch_iter, total=total_batches, desc="生成响应并缓存文本 Embeddings", ncols=100):
                    batch_end = min(batch_start + generation_row_batch_size, num_rows)
                    prompts: List[str] = []
                    index_mapping: List[Tuple[int, int]] = []

                    for row_idx in range(batch_start, batch_end):
                        row_prompts = per_column_prompts[row_idx]
                        col_indices = sorted(row_prompts.keys())
                        for col_idx in col_indices:
                            prompts.append(row_prompts[col_idx])
                            index_mapping.append((row_idx, col_idx))

                    # micro-batch at prompt-level to avoid huge generate() batches
                    for sub_start in range(0, len(prompts), generation_prompt_batch_size):
                        sub_end = min(sub_start + generation_prompt_batch_size, len(prompts))
                        sub_prompts = prompts[sub_start:sub_end]
                        sub_indices = index_mapping[sub_start:sub_end]

                        try:
                            if generation_backend == "vllm":
                                gen_texts = self._vllm_local_completions(
                                    model=str(vllm_model or ""),
                                    prompts=sub_prompts,
                                    max_tokens=int(max_new_tokens),
                                    temperature=0.0,
                                )
                                batch_embeddings = self._encode_text_batch(gen_texts, text_processor)
                            else:
                                batch_embeddings = self._batch_generate_and_encode(
                                    prompts=sub_prompts,
                                    text_processor=text_processor,
                                    max_new_tokens=max_new_tokens,
                                    device=cache_device,
                                )
                        except Exception as e:
                            raise ValueError(
                                f"生成响应失败(backend={generation_backend}): {e}, "
                                f"model={getattr(self, 'model_name_or_path', '未知')}"
                            )

                        if len(batch_embeddings) != len(sub_prompts):
                            raise ValueError("生成响应数量与列数不匹配，无法缓存所有列的embedding。")

                        for embedding, (row_idx, col_idx) in zip(batch_embeddings, sub_indices):
                            cached_embeddings[(row_idx, col_idx)] = embedding

            print(f"✓ 已启用文本embedding缓存，共缓存 {len(cached_embeddings)} 条")
            if cache_path:
                try:
                    cache_dir = os.path.dirname(cache_path)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    torch.save(cached_embeddings, cache_path)
                    print(f"✓ 已保存文本embedding缓存到磁盘: {cache_path}")
                except Exception as e:
                    print(f"  ⚠ 保存缓存失败（忽略）: {cache_path} ({e})")
            return cached_embeddings
        else:
            # Use standard _precompute_text_embeddings
            return _precompute_text_embeddings(
                self,
                text_descriptions,
                text_processor,
                device=device,
                batch_size=batch_size,
                per_column_prompts=per_column_prompts,
                column_names=column_names,
                use_text_output_last_token_embedding=use_last_token,
            )

    def precompute_embeddings_simple(
        self,
        text_descriptions: List[str],
        text_processor,
        device: str = "cuda:0",
        batch_size: int = 32,
        use_last_token: bool = False,
        generate_response: bool = False,
        max_new_tokens: int = 10,
    ) -> Dict[int, torch.Tensor]:
        """
        Simple precompute method for per-row text descriptions (no per-column prompts).

        Args:
            text_descriptions: List of text descriptions (one per row)
            text_processor: TextProcessor instance
            device: Device to use
            batch_size: Batch size for processing
            use_last_token: Whether to extract last token embedding
            generate_response: Whether to use generate_and_encode
            max_new_tokens: Max tokens for generation

        Returns:
            Dict mapping row_idx to cached embedding
        """
        from tqdm import tqdm
        from .runner.experiments.embeddings import _precompute_text_embeddings

        if generate_response:
            print("  → 使用生成式模型生成响应并编码...")
            cached_embeddings = {}
            for idx, prompt in enumerate(tqdm(text_descriptions, desc="生成响应并缓存文本 Embeddings", ncols=100)):
                try:
                    last_token_emb = self.generate_and_encode(
                        prompt=prompt,
                        text_processor=text_processor,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )
                    cached_embeddings[idx] = last_token_emb
                except Exception as e:
                    print(f"  ⚠ 生成响应失败，使用编码方式: {e}")
                    # Fallback to encoding
                    text_inputs = text_processor.process(prompt)
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}
                    with torch.no_grad():
                        H_text = self(text_inputs)
                        if H_text.dim() == 3:
                            last_token_emb = H_text[:, -1, :].squeeze(0).unsqueeze(0).cpu()  # [1, d_model]
                        else:
                            last_token_emb = H_text.unsqueeze(0).cpu()  # [1, d_model]
                        cached_embeddings[idx] = last_token_emb
            print(f"✓ 已启用文本embedding缓存，共缓存 {len(cached_embeddings)} 条")
            return cached_embeddings
        else:
            # Use standard _precompute_text_embeddings
            return _precompute_text_embeddings(
                self,
                text_descriptions,
                text_processor,
                device=device,
                batch_size=batch_size,
                use_text_output_last_token_embedding=use_last_token,
            )


class SimpleMLPEncoder(BaseEncoder):
    """
    Ablation-friendly MLP encoder that ignores table structure.
    """

    def __init__(self, d_cell: int, d_model: int, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = d_cell
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = d_model
        layers.append(nn.Linear(in_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        self.d_model = d_model

    def forward(self, tabular_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        cell_embeddings = tabular_inputs["cell_embeddings"]
        return self.mlp(cell_embeddings)


__all__ = [
    "StructureAwareTransformer",
    "PretrainedTextEncoder",
    "SimpleMLPEncoder",
]


