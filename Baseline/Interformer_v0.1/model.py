"""InterFormer-style PCVR model.

The implementation keeps the paper's three-way split: Interaction Arch for
non-sequential features, Sequence Arch with context-conditioned feed-forward
updates, and Cross Arch summaries for bidirectional exchange.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    all_masked = mask.all(dim=1, keepdim=True)
    if mask.shape[1] == 0:
        return mask
    first_column = torch.zeros_like(mask)
    first_column[:, :1] = True
    return torch.where(all_masked, mask & ~first_column, mask)


def masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    if padding_mask is None:
        return tokens.mean(dim=1)
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def masked_last(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    indices = lengths.clamp_min(1).clamp_max(tokens.shape[1]).to(torch.long) - 1
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch_indices, indices]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class FeatureEmbeddingBank(nn.Module):
    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        emb_dim: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = list(feature_specs)
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []
        for vocab_size, _offset, _length in self.feature_specs:
            should_skip = int(vocab_size) <= 0 or (emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold)
            if should_skip:
                self._embedding_index.append(-1)
            else:
                self._embedding_index.append(len(self.embeddings))
                self.embeddings.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight)
            embedding.weight.data[0].zero_()

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if not self.feature_specs:
            return int_feats.new_zeros(batch_size, 0, self.emb_dim, dtype=torch.float32)
        tokens: list[torch.Tensor] = []
        for feature_index, (vocab_size, offset, length) in enumerate(self.feature_specs):
            embedding_index = self._embedding_index[feature_index]
            if embedding_index < 0:
                tokens.append(int_feats.new_zeros(batch_size, self.emb_dim, dtype=torch.float32))
                continue
            values = int_feats[:, offset : offset + length].to(torch.long).clamp(min=0, max=int(vocab_size))
            embedded = self.embeddings[embedding_index](values)
            valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
            pooled = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            tokens.append(pooled)
        return torch.stack(tokens, dim=1)


class NonSequentialTokenizer(nn.Module):
    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        emb_dim: int,
        d_model: int,
        num_tokens: int = 0,
        emb_skip_threshold: int = 0,
        force_auto_split: bool = False,
    ) -> None:
        super().__init__()
        self.bank = FeatureEmbeddingBank(feature_specs, emb_dim, emb_skip_threshold)
        self.groups = [list(group) for group in groups] or [[index] for index in range(len(feature_specs))]
        self.feature_count = len(feature_specs)
        self.num_tokens = int(num_tokens) if num_tokens > 0 else len(self.groups)
        self.auto_split = force_auto_split or self.num_tokens != len(self.groups)
        if self.auto_split:
            input_dim = max(1, self.feature_count * emb_dim)
            self.project = nn.Sequential(
                nn.Linear(input_dim, self.num_tokens * d_model),
                nn.SiLU(),
                nn.LayerNorm(self.num_tokens * d_model),
            )
        else:
            self.project = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        self.d_model = d_model

    @property
    def embeddings(self) -> Iterable[nn.Embedding]:
        return self.bank.embeddings

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        feature_tokens = self.bank(int_feats)
        if self.num_tokens <= 0:
            return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)
        if self.auto_split:
            if feature_tokens.shape[1] == 0:
                flat = int_feats.new_zeros(batch_size, 1, dtype=torch.float32)
            else:
                flat = feature_tokens.reshape(batch_size, -1)
            return self.project(flat).view(batch_size, self.num_tokens, self.d_model)
        grouped_tokens: list[torch.Tensor] = []
        for group in self.groups:
            valid_indices = [index for index in group if 0 <= index < feature_tokens.shape[1]]
            if valid_indices:
                grouped_tokens.append(feature_tokens[:, valid_indices, :].mean(dim=1))
            else:
                grouped_tokens.append(int_feats.new_zeros(batch_size, self.bank.output_dim, dtype=torch.float32))
        return self.project(torch.stack(grouped_tokens, dim=1))


class DenseTokenProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        if input_dim > 0:
            self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        else:
            self.project = None

    def forward(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.project is None:
            return None
        return self.project(features).unsqueeze(1)


class SequenceTokenizer(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        emb_dim: int,
        d_model: int,
        num_time_buckets: int = 0,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_sizes = [int(value) for value in vocab_sizes]
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []
        for vocab_size in self.vocab_sizes:
            should_skip = vocab_size <= 0 or (emb_skip_threshold > 0 and vocab_size > emb_skip_threshold)
            if should_skip:
                self._embedding_index.append(-1)
            else:
                self._embedding_index.append(len(self.embeddings))
                self.embeddings.append(nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0))
        input_dim = max(1, len(self.vocab_sizes) * emb_dim)
        self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
        self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0) if num_time_buckets > 0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight)
            embedding.weight.data[0].zero_()
        if self.time_embedding is not None:
            nn.init.xavier_normal_(self.time_embedding.weight)
            self.time_embedding.weight.data[0].zero_()

    def forward(self, sequence: torch.Tensor, time_buckets: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, feature_count, seq_len = sequence.shape
        pieces: list[torch.Tensor] = []
        for feature_index in range(feature_count):
            embedding_index = self._embedding_index[feature_index] if feature_index < len(self._embedding_index) else -1
            if embedding_index < 0:
                pieces.append(sequence.new_zeros(batch_size, seq_len, self.emb_dim, dtype=torch.float32))
                continue
            vocab_size = self.vocab_sizes[feature_index]
            values = sequence[:, feature_index, :].to(torch.long).clamp(min=0, max=vocab_size)
            pieces.append(self.embeddings[embedding_index](values))
        if pieces:
            token_input = torch.cat(pieces, dim=-1)
        else:
            token_input = sequence.new_zeros(batch_size, seq_len, 1, dtype=torch.float32)
        tokens = self.project(token_input)
        if self.time_embedding is not None and time_buckets is not None:
            time_values = time_buckets.to(torch.long).clamp(min=0, max=self.time_embedding.num_embeddings - 1)
            tokens = tokens + self.time_embedding(time_values)
        return tokens


class EmbeddingParameterMixin:
    def get_sparse_params(self) -> list[nn.Parameter]:
        sparse_ptrs = {module.weight.data_ptr() for module in self.modules() if isinstance(module, nn.Embedding)}
        return [parameter for parameter in self.parameters() if parameter.data_ptr() in sparse_ptrs]

    def get_dense_params(self) -> list[nn.Parameter]:
        sparse_ptrs = {parameter.data_ptr() for parameter in self.get_sparse_params()}
        return [parameter for parameter in self.parameters() if parameter.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> set[int]:
        reinitialized: set[int] = set()
        for module in self.modules():
            if not isinstance(module, nn.Embedding):
                continue
            if module.num_embeddings - 1 <= cardinality_threshold:
                continue
            nn.init.xavier_normal_(module.weight)
            module.weight.data[0].zero_()
            reinitialized.add(module.weight.data_ptr())
        return reinitialized


def choose_num_heads(d_model: int, requested_heads: int) -> int:
    requested_heads = max(1, requested_heads)
    if d_model % requested_heads == 0:
        return requested_heads
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    batch_size, query_len, d_model = q.shape
    head_dim = d_model // num_heads
    q = q.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, v.shape[1], num_heads, head_dim).transpose(1, 2)
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
    )
    return output.transpose(1, 2).contiguous().view(batch_size, query_len, d_model)


def causal_valid_attention_mask(padding_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, token_count = padding_mask.shape
    causal = torch.ones(token_count, token_count, dtype=torch.bool, device=padding_mask.device).tril()
    key_valid = ~padding_mask
    mask = causal.unsqueeze(0) & key_valid.unsqueeze(1)
    query_invalid = padding_mask.unsqueeze(-1)
    fallback = torch.eye(token_count, dtype=torch.bool, device=padding_mask.device).unsqueeze(0)
    mask = torch.where(query_invalid, fallback, mask)
    return mask.unsqueeze(1).expand(batch_size, num_heads, token_count, token_count)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if length == 0:
        return torch.empty(0, dim, device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    values = torch.zeros(length, dim, device=device)
    values[:, 0::2] = torch.sin(positions * frequencies)
    values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
    return values


class PersonalizedFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.gate = nn.Linear(d_model, d_model)
        self.bias = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(context)).unsqueeze(1)
        bias = self.bias(context).unsqueeze(1)
        return self.ffn(sequence * gate + bias)


class CrossSummary(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.summary_queries = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sequences: list[torch.Tensor],
        masks: list[torch.Tensor],
        lengths: list[torch.Tensor],
    ) -> torch.Tensor:
        if not sequences:
            raise ValueError("InterFormer requires at least one sequence domain")
        all_tokens = torch.cat(sequences, dim=1)
        all_masks = safe_key_padding_mask(torch.cat(masks, dim=1))
        query = self.summary_queries.unsqueeze(0).expand(all_tokens.shape[0], -1, -1)
        attended, _weights = self.attention(query, all_tokens, all_tokens, key_padding_mask=all_masks, need_weights=False)
        cls_summary = attended[:, 0]
        pma_summary = attended[:, 1]
        recent = torch.stack(
            [masked_last(tokens, seq_len) for tokens, seq_len in zip(sequences, lengths, strict=True)],
            dim=1,
        ).mean(dim=1)
        joined = torch.cat([cls_summary, pma_summary, recent], dim=-1)
        gated = torch.sigmoid(self.gate(joined)) * (cls_summary + pma_summary + recent) / 3.0
        return self.norm(gated)


class InterFormerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float, add_context_token: bool) -> None:
        super().__init__()
        self.add_context_token = add_context_token
        self.cross_summary = CrossSummary(d_model, num_heads, dropout)
        self.ns_norm = nn.LayerNorm(d_model)
        self.ns_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ns_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )
        self.ns_summary = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.pffn = PersonalizedFeedForward(d_model, hidden_mult, dropout)
        self.seq_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.seq_norm = nn.LayerNorm(d_model)
        self.seq_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, d_model),
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
        masks: list[torch.Tensor],
        lengths: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        sequence_summary = self.cross_summary(sequences, masks, lengths).unsqueeze(1)
        interaction_tokens = torch.cat([ns_tokens, sequence_summary], dim=1)
        interaction_base = self.ns_norm(interaction_tokens)
        interacted, _weights = self.ns_attention(interaction_base, interaction_base, interaction_base, need_weights=False)
        interacted = interaction_tokens + interacted
        interacted = interacted + self.ns_ffn(interacted)
        ns_tokens = interacted[:, : ns_tokens.shape[1]]

        context = torch.sigmoid(self.ns_summary(masked_mean(ns_tokens))) * masked_mean(ns_tokens)
        updated_sequences: list[torch.Tensor] = []
        for tokens, mask in zip(sequences, masks, strict=True):
            personalized = tokens + self.pffn(tokens, context)
            if self.add_context_token:
                context_token = context.unsqueeze(1)
                attention_input = torch.cat([context_token, personalized], dim=1)
                context_mask = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
                attention_mask = torch.cat([context_mask, safe_key_padding_mask(mask)], dim=1)
                attended, _weights = self.seq_attention(
                    attention_input,
                    attention_input,
                    attention_input,
                    key_padding_mask=attention_mask,
                    need_weights=False,
                )
                attended = attended[:, 1:]
            else:
                attended, _weights = self.seq_attention(
                    personalized,
                    personalized,
                    personalized,
                    key_padding_mask=safe_key_padding_mask(mask),
                    need_weights=False,
                )
            updated = self.seq_norm(personalized + attended)
            updated_sequences.append(updated + self.seq_ffn(updated))
        return ns_tokens, updated_sequences


class PCVRInterFormer(EmbeddingParameterMixin, nn.Module):
    def __init__(
        self,
        user_int_feature_specs: list[tuple[int, int, int]],
        item_int_feature_specs: list[tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict[str, list[int]],
        user_ns_groups: list[list[int]],
        item_ns_groups: list[list[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = "full",
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = "group",
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
    ) -> None:
        super().__init__()
        del num_queries, seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, seq_id_threshold
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = d_model
        self.action_num = action_num
        self.seq_domains = sorted(seq_vocab_sizes)
        force_auto_split = ns_tokenizer_type == "rankmixer"
        self.user_tokenizer = NonSequentialTokenizer(
            user_int_feature_specs,
            user_ns_groups,
            emb_dim,
            d_model,
            user_ns_tokens,
            emb_skip_threshold,
            force_auto_split=force_auto_split,
        )
        self.item_tokenizer = NonSequentialTokenizer(
            item_int_feature_specs,
            item_ns_groups,
            emb_dim,
            d_model,
            item_ns_tokens,
            emb_skip_threshold,
            force_auto_split=force_auto_split,
        )
        self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
        self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold)
                for domain, vocab_sizes in seq_vocab_sizes.items()
            }
        )
        self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
        self.num_ns += int(user_dense_dim > 0) + int(item_dense_dim > 0)
        self.blocks = nn.ModuleList(
            [
                InterFormerBlock(d_model, num_heads, hidden_mult, dropout_rate, add_context_token=layer_index == 0)
                for layer_index in range(max(1, num_blocks))
            ]
        )
        self.final_cross = CrossSummary(d_model, num_heads, dropout_rate)
        self.final_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * hidden_mult, action_num),
        )

    def _encode_non_sequence(self, inputs: ModelInput) -> torch.Tensor:
        parts = [self.user_tokenizer(inputs.user_int_feats)]
        user_dense = self.user_dense(inputs.user_dense_feats)
        if user_dense is not None:
            parts.append(user_dense)
        parts.append(self.item_tokenizer(inputs.item_int_feats))
        item_dense = self.item_dense(inputs.item_dense_feats)
        if item_dense is not None:
            parts.append(item_dense)
        return torch.cat(parts, dim=1)

    def _encode_sequences(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        for domain in self.seq_domains:
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device)
            tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
            tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
            sequences.append(tokens)
            masks.append(make_padding_mask(seq_len, raw_sequence.shape[2]))
            lengths.append(seq_len)
        return sequences, masks, lengths

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        ns_tokens = self._encode_non_sequence(inputs)
        sequences, masks, lengths = self._encode_sequences(inputs)
        for block in self.blocks:
            ns_tokens, sequences = block(ns_tokens, sequences, masks, lengths)
        ns_summary = masked_mean(ns_tokens)
        seq_summary = self.final_cross(sequences, masks, lengths)
        gate = self.final_gate(torch.cat([ns_summary, seq_summary], dim=-1))
        return gate * ns_summary + (1.0 - gate) * seq_summary

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings