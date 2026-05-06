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
    """Embedding lookup for int features with optional hash trick and embedding
    sharing.

    Parameters
    ----------
    hash_bucket_size:
        When > 0, features whose vocab exceeds this value use a *hash
        embedding*: ``hash(value) % hash_bucket_size`` with a table of size
        ``hash_bucket_size + 1`` instead of ``vocab + 1``.  This keeps GPU
        memory bounded for ultra-high-cardinality features (e.g. 278M item
        IDs) while still capturing their signal.

    external_embeddings:
        Dict mapping *feature_index* → ``nn.Embedding``.  When a feature has
        an external embedding, the bank uses that table instead of creating
        its own, enabling embedding sharing between tokenizers (e.g. item ID
        shared between non-sequence and sequence branches).
    """

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        emb_dim: int,
        emb_skip_threshold: int = 0,
        hash_bucket_size: int = 0,
        external_embeddings: dict[int, nn.Embedding] | None = None,
    ) -> None:
        super().__init__()
        self.feature_specs = list(feature_specs)
        self.emb_dim = emb_dim
        self.hash_bucket_size = int(hash_bucket_size)
        self._external = dict(external_embeddings or {})

        self.embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []     # feature_idx → embedding_idx in self.embeddings (-1 = skip/zero, -2 = hash)
        self._hash_vocab: dict[int, int] = {}     # feature_idx → original vocab_size (for hashed features)

        for feature_index, (vocab_size, _offset, _length) in enumerate(self.feature_specs):
            vocab = int(vocab_size)
            if feature_index in self._external:
                # Use external (shared) embedding — index = -3 means "look up via _external dict"
                self._embedding_index.append(-3)
                continue
            if vocab <= 0:
                self._embedding_index.append(-1)
                continue
            if emb_skip_threshold > 0 and vocab > emb_skip_threshold:
                if self.hash_bucket_size > 0:
                    # Hash embedding: compact table, hash-based lookup
                    self._embedding_index.append(-2)
                    self._hash_vocab[feature_index] = vocab
                else:
                    self._embedding_index.append(-1)
                continue
            self._embedding_index.append(len(self.embeddings))
            self.embeddings.append(nn.Embedding(vocab + 1, emb_dim, padding_idx=0))
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight)
            embedding.weight.data[0].zero_()
        # Re-init external embeddings too (they may be shared, but init once is fine)
        for emb in self._external.values():
            nn.init.xavier_normal_(emb.weight)
            emb.weight.data[0].zero_()

    @staticmethod
    def _hash_values(values: torch.Tensor, bucket_size: int) -> torch.Tensor:
        """Simple multiplicative hash: (v * 0x9E3779B9) % bucket_size + 1.
        Padding value 0 stays at 0 (since we add 1 to the result for valid values).
        """
        hashed = (values.to(torch.int64) * 0x9E3779B9) % bucket_size
        hashed = hashed + 1  # shift so that 0 stays reserved for padding
        hashed[values == 0] = 0
        return hashed.to(torch.long)

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if not self.feature_specs:
            return int_feats.new_zeros(batch_size, 0, self.emb_dim, dtype=torch.float32)
        tokens: list[torch.Tensor] = []
        for feature_index, (vocab_size, offset, length) in enumerate(self.feature_specs):
            emb_idx = self._embedding_index[feature_index] if feature_index < len(self._embedding_index) else -1
            values = int_feats[:, offset : offset + length].to(torch.long)
            values = values.clamp(min=0, max=max(1, int(vocab_size)))

            if emb_idx == -1:
                # Skip / zero
                tokens.append(int_feats.new_zeros(batch_size, self.emb_dim, dtype=torch.float32))
            elif emb_idx == -2:
                # Hash embedding lookup
                hashed = self._hash_values(values, self.hash_bucket_size)
                embedded = self._get_hash_embedding(feature_index, hashed)
                valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
                pooled = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                tokens.append(pooled)
            elif emb_idx == -3:
                # External / shared embedding
                ext_emb = self._external[feature_index]
                embedded = ext_emb(values)
                valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
                pooled = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                tokens.append(pooled)
            else:
                embedded = self.embeddings[emb_idx](values)
                valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
                pooled = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                tokens.append(pooled)
        return torch.stack(tokens, dim=1)

    def _get_hash_embedding(self, feature_index: int, hashed_values: torch.Tensor) -> torch.Tensor:
        """Get or create the hash embedding table for a feature."""
        # We lazily create hash embedding tables in a side dict so that
        # external callers don't need to pre-build them.
        if not hasattr(self, '_hash_embeddings'):
            self._hash_embeddings: dict[int, nn.Embedding] = {}
        if feature_index not in self._hash_embeddings:
            emb = nn.Embedding(self.hash_bucket_size + 1, self.emb_dim, padding_idx=0)
            nn.init.xavier_normal_(emb.weight)
            emb.weight.data[0].zero_()
            emb = emb.to(hashed_values.device)
            self._hash_embeddings[feature_index] = emb
        return self._hash_embeddings[feature_index](hashed_values)


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
        hash_bucket_size: int = 0,
        external_embeddings: dict[int, nn.Embedding] | None = None,
    ) -> None:
        super().__init__()
        self.bank = FeatureEmbeddingBank(
            feature_specs, emb_dim, emb_skip_threshold,
            hash_bucket_size=hash_bucket_size,
            external_embeddings=external_embeddings,
        )
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
        hash_bucket_size: int = 0,
        external_embeddings: dict[int, nn.Embedding] | None = None,
        external_vocab_sizes: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.vocab_sizes = [int(value) for value in vocab_sizes]
        self.emb_dim = emb_dim
        self.hash_bucket_size = int(hash_bucket_size)
        self._external = dict(external_embeddings or {})
        self._external_vocab = dict(external_vocab_sizes or {})

        self.embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []  # -1=skip/zero, -2=hash, -3=external
        self._hash_vocab: dict[int, int] = {}

        for feat_idx, vocab_size in enumerate(self.vocab_sizes):
            vocab = int(vocab_size)
            if feat_idx in self._external:
                self._embedding_index.append(-3)
                continue
            if vocab <= 0:
                self._embedding_index.append(-1)
                continue
            if emb_skip_threshold > 0 and vocab > emb_skip_threshold:
                if self.hash_bucket_size > 0:
                    self._embedding_index.append(-2)
                    self._hash_vocab[feat_idx] = vocab
                else:
                    self._embedding_index.append(-1)
                continue
            self._embedding_index.append(len(self.embeddings))
            self.embeddings.append(nn.Embedding(vocab + 1, emb_dim, padding_idx=0))
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

    @staticmethod
    def _hash_values(values: torch.Tensor, bucket_size: int) -> torch.Tensor:
        hashed = (values.to(torch.int64) * 0x9E3779B9) % bucket_size
        hashed = hashed + 1
        hashed[values == 0] = 0
        return hashed.to(torch.long)

    def _get_hash_embedding(self, feat_idx: int, hashed_values: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, '_hash_embeddings'):
            self._hash_embeddings: dict[int, nn.Embedding] = {}
        if feat_idx not in self._hash_embeddings:
            emb = nn.Embedding(self.hash_bucket_size + 1, self.emb_dim, padding_idx=0)
            nn.init.xavier_normal_(emb.weight)
            emb.weight.data[0].zero_()
            emb = emb.to(hashed_values.device)
            self._hash_embeddings[feat_idx] = emb
        return self._hash_embeddings[feat_idx](hashed_values)

    def forward(self, sequence: torch.Tensor, time_buckets: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, feature_count, seq_len = sequence.shape
        pieces: list[torch.Tensor] = []
        for feature_index in range(feature_count):
            emb_idx = self._embedding_index[feature_index] if feature_index < len(self._embedding_index) else -1
            vocab_size = self.vocab_sizes[feature_index] if feature_index < len(self.vocab_sizes) else 0
            values = sequence[:, feature_index, :].to(torch.long)
            values = values.clamp(min=0, max=max(1, int(vocab_size)))

            if emb_idx == -1:
                pieces.append(sequence.new_zeros(batch_size, seq_len, self.emb_dim, dtype=torch.float32))
            elif emb_idx == -2:
                hashed = self._hash_values(values, self.hash_bucket_size)
                pieces.append(self._get_hash_embedding(feature_index, hashed))
            elif emb_idx == -3:
                ext_emb = self._external[feature_index]
                pieces.append(ext_emb(values))
            else:
                pieces.append(self.embeddings[emb_idx](values))
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


# ═══════════════════════════════════════════════════════════════════════
#  Paper-aligned Cross Arch components
# ═══════════════════════════════════════════════════════════════════════

class LinearCompressedEmbedding(nn.Module):
    """Linear projection along the *token* axis to compress / expand the
    number of tokens while preserving the per-token dimension ``d_model``.

    This corresponds to Eq. (5) + the LCE described in Appendix A.2.1 of the
    InterFormer paper.
    """

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_tokens, output_tokens))
        self.bias = nn.Parameter(torch.zeros(output_tokens))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, D] → transpose → matmul → transpose back → [B, T_out, D]
        x_t = x.transpose(1, 2)
        compressed = torch.matmul(x_t, self.weight) + self.bias
        return compressed.transpose(1, 2)


class SelfGating(nn.Module):
    """Self-gating mechanism (Highway Transformer style).

    ``gating(x) = sigmoid(x * MLP(x)) * x``
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * self.mlp(x)) * x


class NonSequenceSummarizer(nn.Module):
    """NS → Seq summary: LCE + SelfGating (Eq. 10 in the paper)."""

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.compress = LinearCompressedEmbedding(input_tokens, output_tokens)
        self.gating = SelfGating(d_model, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gating(self.compress(x))


class PoolingMultiHeadAttention(nn.Module):
    """PMA (Pooling by Multi-Head Attention) with learnable seed vectors."""

    def __init__(
        self, num_seed_vectors: int, d_model: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_seed_vectors = num_seed_vectors
        self.seed_vectors = (
            nn.Parameter(torch.randn(num_seed_vectors, d_model) * 0.02)
            if num_seed_vectors > 0
            else None
        )
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.num_seed_vectors == 0:
            return x[:, :0, :]
        query = self.seed_vectors.unsqueeze(0).expand(x.size(0), -1, -1)
        pooled, _ = self.attn(
            query, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return pooled


class SequenceSummarizer(nn.Module):
    """Seq → NS summary: CLS + PMA + Recent tokens, gated (Eq. 11).

    ``num_cls_tokens`` CLS tokens are assumed to be at the *beginning* of the
    sequence (prepended before the first InterFormer block).
    """

    def __init__(
        self,
        num_cls_tokens: int,
        num_pma_tokens: int,
        num_recent_tokens: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.num_recent_tokens = num_recent_tokens
        self.pma = PoolingMultiHeadAttention(
            num_seed_vectors=num_pma_tokens,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        total_tokens = num_cls_tokens + num_pma_tokens + num_recent_tokens
        self.gating = SelfGating(d_model, hidden_dim, dropout)
        self.output_tokens = total_tokens

    def forward(
        self,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = seq.size(0)
        # CLS tokens: first num_cls_tokens positions
        cls_tokens = seq[:, : self.num_cls_tokens, :]
        # PMA on non-CLS portion
        body = seq[:, self.num_cls_tokens :, :]
        body_mask = (
            key_padding_mask[:, self.num_cls_tokens :]
            if key_padding_mask is not None
            else None
        )
        pma_tokens = self.pma(body, key_padding_mask=body_mask)
        # Recent tokens
        recent_tokens = seq.new_zeros(
            batch_size, self.num_recent_tokens, seq.size(-1)
        )
        if self.num_recent_tokens > 0:
            for i in range(batch_size):
                valid_len = lengths[i].clamp_min(1).item()
                start = max(0, valid_len - self.num_recent_tokens)
                end = valid_len
                if end > start:
                    recent_tokens[i, : end - start] = seq[
                        i, start:end, :
                    ]
        combined = torch.cat([cls_tokens, pma_tokens, recent_tokens], dim=1)
        return self.gating(combined)


# ═══════════════════════════════════════════════════════════════════════
#  Interaction Arch backbones (DOT / DCN / DHEN)
# ═══════════════════════════════════════════════════════════════════════

class DotProductInteraction(nn.Module):
    """Attention-style dot-product feature interaction."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(x.size(-1))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return self.proj(torch.matmul(weights, x))


class CrossLayer(nn.Module):
    """Single DCN-v2 cross layer with low-rank token mixing."""

    def __init__(
        self, num_tokens: int, d_model: int, low_rank: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        low_rank = min(low_rank, num_tokens)
        self.token_u = nn.Linear(num_tokens, low_rank, bias=False)
        self.token_v = nn.Linear(low_rank, num_tokens, bias=True)
        self.feature_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        token_mixed = self.token_v(self.token_u(x.transpose(1, 2))).transpose(1, 2)
        feature_mixed = self.feature_proj(x)
        return x0 * feature_mixed + self.dropout(token_mixed) + x


class DCNInteraction(nn.Module):
    """DCN-v2 interaction block: cross layers + deep network (Eq. in Sec 3.2)."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        num_layers: int = 2,
        low_rank: int = 32,
        deep_hidden: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossLayer(num_tokens, d_model, low_rank, dropout)
            for _ in range(num_layers)
        ])
        self.deep = nn.Sequential(
            nn.Linear(d_model, deep_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        cross = x
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        return self.norm(cross + self.deep(x))


class DHENLayer(nn.Module):
    """Single DHEN layer: DOT + DCN ensemble with shortcut."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        dcn_layers: int,
        low_rank: int,
        deep_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dot_interaction = DotProductInteraction(d_model, dropout)
        self.dcn_interaction = DCNInteraction(
            num_tokens, d_model, dcn_layers, low_rank, deep_hidden, dropout
        )
        self.ensemble_proj = nn.Linear(d_model * 2, d_model)
        self.shortcut = nn.Sequential(
            nn.Linear(d_model, deep_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dot_out = self.dot_interaction(x)
        dcn_out = self.dcn_interaction(x)
        ensemble = self.ensemble_proj(torch.cat([dot_out, dcn_out], dim=-1))
        shortcut = self.shortcut(x)
        return self.norm(ensemble + shortcut)


class DHENInteraction(nn.Module):
    """Stacked DHEN layers."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        num_layers: int = 2,
        dcn_layers: int = 2,
        low_rank: int = 32,
        deep_hidden: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DHENLayer(num_tokens, d_model, dcn_layers, low_rank, deep_hidden, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DotInteractionStack(nn.Module):
    """Stacked DOT interaction layers with residual norm."""

    def __init__(self, d_model: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DotProductInteraction(d_model, dropout) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for interaction, norm in zip(self.layers, self.norms):
            x = norm(x + interaction(x))
        return x


class DCNInteractionStack(nn.Module):
    """Stacked DCN interaction blocks."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        num_layers: int,
        low_rank: int,
        deep_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DCNInteraction(num_tokens, d_model, 1, low_rank, deep_hidden, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for interaction, norm in zip(self.layers, self.norms):
            x = norm(x + interaction(x))
        return x


VALID_INTERACTION_BACKBONES = {"dhen", "dot", "dcn"}


class InteractionArch(nn.Module):
    """Paper's Interaction Arch (Sec 4.2): behavior-aware NS interaction.

    Fuses NS tokens with sequence summary, applies a backbone interaction
    module (DHEN / DOT / DCN), then projects back to the original NS token
    count via LCE + MLP.
    """

    def __init__(
        self,
        non_seq_tokens: int,
        seq_summary_tokens: int,
        d_model: int,
        backbone: str = "dhen",
        interaction_layers: int = 2,
        dcn_layers: int = 2,
        low_rank: int = 32,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone not in VALID_INTERACTION_BACKBONES:
            raise ValueError(
                f"Unsupported interaction backbone: {backbone}. "
                f"Choose from {VALID_INTERACTION_BACKBONES}"
            )
        total_tokens = non_seq_tokens + seq_summary_tokens
        if backbone == "dhen":
            self.interaction = DHENInteraction(
                num_tokens=total_tokens,
                d_model=d_model,
                num_layers=interaction_layers,
                dcn_layers=dcn_layers,
                low_rank=low_rank,
                deep_hidden=hidden_dim,
                dropout=dropout,
            )
        elif backbone == "dot":
            self.interaction = DotInteractionStack(
                d_model=d_model, num_layers=interaction_layers, dropout=dropout
            )
        else:  # dcn
            self.interaction = DCNInteractionStack(
                num_tokens=total_tokens,
                d_model=d_model,
                num_layers=interaction_layers,
                low_rank=low_rank,
                deep_hidden=hidden_dim,
                dropout=dropout,
            )
        self.output_projector = LinearCompressedEmbedding(
            input_tokens=total_tokens, output_tokens=non_seq_tokens
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, non_seq: torch.Tensor, seq_summary: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat([non_seq, seq_summary], dim=1)
        interacted = self.interaction(fused)
        projected = self.output_projector(interacted)
        return self.output_norm(self.output_mlp(projected))


# ═══════════════════════════════════════════════════════════════════════
#  Sequence Arch with RoPE
# ═══════════════════════════════════════════════════════════════════════

class RotaryMultiHeadAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_base = rope_base
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _build_rope_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        inv_freq = 1.0 / (
            self.rope_base
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=dtype)
                / self.head_dim
            )
        )
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x_rot = x.float()
        x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
        rotated = torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )
        return rotated.to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        cos, sin = self._build_rope_cache(
            seq_len=x.size(1), device=x.device, dtype=x.dtype
        )
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(
                -1, self.num_heads, x.size(1), -1
            )
        else:
            attn_mask = None
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
        )
        output = output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.out_proj(output)


class SequenceArch(nn.Module):
    """Paper's Sequence Arch (Sec 4.3): PFFN + RoPE attention.

    Uses non-sequence summary to personalize the sequence FFN, then applies
    rotary self-attention for sequence modeling.  When ``item_gate`` is
    provided, the PFFN context is additionally modulated by a target-item
    gate so that the sequence focuses on item-relevant behaviours (similar in
    spirit to DIN's target attention).
    """

    def __init__(
        self,
        num_summary_tokens: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.pffn_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.pffn = PersonalizedFeedForward(d_model, hidden_dim // d_model, dropout)
        self.attn = RotaryMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        # Target-item gate: MLP that produces a scalar gate per dimension
        self.item_gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        non_seq_summary: torch.Tensor,
        seq: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        item_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute PFFN context from NS summary
        context = non_seq_summary.mean(dim=1)  # [B, D]
        # Modulate by target item if provided (item-aware personalization)
        if item_summary is not None:
            item_gate = self.item_gate_proj(item_summary.mean(dim=1))
            context = context * item_gate
        seq = seq + self.pffn(self.pffn_norm(seq), context)
        seq = seq + self.attn(self.attn_norm(seq), key_padding_mask=key_padding_mask)
        return seq


# ═══════════════════════════════════════════════════════════════════════
#  Item-specific interaction (target-item enhancement)
# ═══════════════════════════════════════════════════════════════════════

class ItemInteractionArch(nn.Module):
    """Dedicated item-feature interaction module.

    Processes item-side features (item_int + item_dense) with their own DHEN
    backbone before fusing into the main NS token stream.  This gives item
    features—which are critical for PCVR since the prediction is *per-item*—a
    dedicated interaction pathway instead of being drowned among user tokens.
    """

    def __init__(
        self,
        item_tokens: int,
        d_model: int,
        backbone: str = "dhen",
        interaction_layers: int = 2,
        dcn_layers: int = 2,
        low_rank: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone == "dhen":
            self.interaction = DHENInteraction(
                num_tokens=item_tokens,
                d_model=d_model,
                num_layers=interaction_layers,
                dcn_layers=dcn_layers,
                low_rank=low_rank,
                deep_hidden=hidden_dim,
                dropout=dropout,
            )
        elif backbone == "dot":
            self.interaction = DotInteractionStack(
                d_model=d_model, num_layers=interaction_layers, dropout=dropout
            )
        else:  # dcn
            self.interaction = DCNInteractionStack(
                num_tokens=item_tokens,
                d_model=d_model,
                num_layers=interaction_layers,
                low_rank=low_rank,
                deep_hidden=hidden_dim,
                dropout=dropout,
            )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, item_tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.interaction(item_tokens))


class ItemAwareGating(nn.Module):
    """Target-item aware gating for cross-modal information flow.

    In PCVR, the target item determines which user behaviours are relevant.
    This module uses the item representation to produce:
    - ``ns_gate``: modulates the NS→Seq summary (sequence should focus on
      item-relevant history)
    - ``seq_gate``: modulates the Seq→NS summary (item-aware behaviour
      compression)
    - ``fuse_gate``: final fusion weight between NS and Seq representations
    """

    def __init__(self, d_model: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.item_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model * 3),  # 3 gates
        )

    def forward(
        self, item_summary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(ns_gate, seq_gate, fuse_gate)``, each [B, D]."""
        gates = self.item_proj(item_summary)  # [B, 3*D]
        ns_gate, seq_gate, fuse_gate = gates.chunk(3, dim=-1)
        return (
            torch.sigmoid(ns_gate),
            torch.sigmoid(seq_gate),
            torch.sigmoid(fuse_gate),
        )


# ═══════════════════════════════════════════════════════════════════════
#  InterFormer Block (paper-aligned)
# ═══════════════════════════════════════════════════════════════════════

class InterFormerBlock(nn.Module):
    """Single InterFormer block as described in the paper (Algorithm 1).

    Flow:
    1. Cross Arch: summarize NS → seq context, summarize Seq → NS context
    2. Interaction Arch: update NS tokens with seq summary
    3. Sequence Arch: update each sequence with NS summary
    """

    def __init__(
        self,
        non_seq_tokens: int,
        num_cls_tokens: int,
        num_pma_tokens: int,
        num_recent_tokens: int,
        d_model: int,
        num_heads: int,
        interaction_backbone: str = "dhen",
        interaction_layers: int = 2,
        dcn_layers: int = 2,
        cross_low_rank: int = 32,
        summary_hidden: int = 256,
        interaction_hidden: int = 512,
        sequence_hidden: int = 256,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        # NS → Seq summary
        self.non_seq_summarizer = NonSequenceSummarizer(
            input_tokens=non_seq_tokens,
            output_tokens=num_cls_tokens,
            d_model=d_model,
            hidden_dim=summary_hidden,
            dropout=dropout,
        )
        # Seq → NS summary
        self.sequence_summarizer = SequenceSummarizer(
            num_cls_tokens=num_cls_tokens,
            num_pma_tokens=num_pma_tokens,
            num_recent_tokens=num_recent_tokens,
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=summary_hidden,
            dropout=dropout,
        )
        seq_summary_tokens = self.sequence_summarizer.output_tokens
        self.interaction_arch = InteractionArch(
            non_seq_tokens=non_seq_tokens,
            seq_summary_tokens=seq_summary_tokens,
            d_model=d_model,
            backbone=interaction_backbone,
            interaction_layers=interaction_layers,
            dcn_layers=dcn_layers,
            low_rank=cross_low_rank,
            hidden_dim=interaction_hidden,
            dropout=dropout,
        )
        self.sequence_arch = SequenceArch(
            num_summary_tokens=num_cls_tokens,
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=sequence_hidden,
            dropout=dropout,
            rope_base=rope_base,
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
        masks: list[torch.Tensor],
        lengths: list[torch.Tensor],
        item_summary: torch.Tensor | None = None,
        ns_gate: torch.Tensor | None = None,
        seq_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # 1. Cross Arch: NS → Seq summary (modulated by item gate)
        ns_summary = self.non_seq_summarizer(ns_tokens)  # [B, cls, D]
        if ns_gate is not None:
            ns_summary = ns_summary * ns_gate.unsqueeze(1)

        # 2. Cross Arch: Seq → NS summary (modulated by item gate)
        all_seq = torch.cat(sequences, dim=1)
        all_mask = safe_key_padding_mask(torch.cat(masks, dim=1))
        all_len = torch.stack(lengths, dim=1).sum(dim=1)
        seq_summary = self.sequence_summarizer(all_seq, all_len, all_mask)
        if seq_gate is not None:
            seq_summary = seq_summary * seq_gate.unsqueeze(1)

        # 3. Interaction Arch: update NS tokens
        ns_tokens = self.interaction_arch(ns_tokens, seq_summary)

        # 4. Sequence Arch: update each sequence (item-aware personalization)
        updated_sequences: list[torch.Tensor] = []
        for tokens, mask in zip(sequences, masks, strict=True):
            key_mask = safe_key_padding_mask(mask)
            updated = self.sequence_arch(ns_summary, tokens, key_mask,
                                         item_summary=item_summary)
            updated_sequences.append(updated)
        return ns_tokens, updated_sequences


# ═══════════════════════════════════════════════════════════════════════
#  Main model
# ═══════════════════════════════════════════════════════════════════════

class PCVRInterFormer(EmbeddingParameterMixin, nn.Module):
    """Paper-aligned InterFormer for PCVR prediction.

    Key architecture decisions (matching the paper):
    - Interaction Arch uses DHEN/DOT/DCN backbone for explicit feature crosses
    - Sequence Arch uses PFFN + RoPE attention
    - Cross Arch uses LCE+SelfGating (NS→Seq) and CLS+PMA+Recent (Seq→NS)
    - CLS tokens are prepended once before the first block
    """

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
        use_rope: bool = True,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = "group",
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # ── New Interaction Arch hyperparams ──
        interaction_backbone: str = "dhen",
        interaction_layers: int = 2,
        dcn_layers: int = 2,
        cross_low_rank: int = 32,
        num_cls_tokens: int = 4,
        num_pma_tokens: int = 2,
        num_recent_tokens: int = 2,
        # ── Feature engineering hyperparams ──
        hash_bucket_size: int = 100000,
        item_vocab_size: int = 0,
        item_id_seq_domain: str = "seq_c",
    ) -> None:
        super().__init__()
        num_heads = choose_num_heads(d_model, num_heads)
        self.d_model = d_model
        self.action_num = action_num
        self.seq_domains = sorted(seq_vocab_sizes)
        self.num_cls_tokens = num_cls_tokens
        self.use_rope = use_rope

        # ── Shared item embedding (if item_vocab_size > 0) ──
        # The top-level item_id and domain_c_seq_47 share the same ID space.
        # We create ONE embedding table and pass it to both the NS item
        # tokenizer and the seq_c sequence tokenizer.
        shared_item_emb: nn.Embedding | None = None
        item_ns_external: dict[int, nn.Embedding] = {}
        seq_c_external: dict[int, nn.Embedding] = {}
        seq_c_external_vocab: dict[int, int] = {}

        if item_vocab_size > 0:
            shared_item_emb = nn.Embedding(
                item_vocab_size + 1, emb_dim, padding_idx=0
            )
            # Auto-detect: the item_id in item_int is the feature with the
            # largest vocab (should equal item_vocab_size).
            for idx, (vs, _off, _len) in enumerate(item_int_feature_specs):
                if int(vs) == item_vocab_size:
                    item_ns_external[idx] = shared_item_emb
                    break
            # Auto-detect in seq_c: the feature with vocab == item_vocab_size
            if item_id_seq_domain in seq_vocab_sizes:
                for idx, vs in enumerate(seq_vocab_sizes[item_id_seq_domain]):
                    if int(vs) == item_vocab_size:
                        seq_c_external[idx] = shared_item_emb
                        seq_c_external_vocab[idx] = item_vocab_size
                        break

        # ---- Tokenizers ----
        force_auto_split = ns_tokenizer_type == "rankmixer"
        self.user_tokenizer = NonSequentialTokenizer(
            user_int_feature_specs, user_ns_groups, emb_dim, d_model,
            user_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split,
            hash_bucket_size=hash_bucket_size,
        )
        self.item_tokenizer = NonSequentialTokenizer(
            item_int_feature_specs, item_ns_groups, emb_dim, d_model,
            item_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split,
            hash_bucket_size=hash_bucket_size,
            external_embeddings=item_ns_external if item_ns_external else None,
        )
        self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
        self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
        self.sequence_tokenizers = nn.ModuleDict()
        for domain, vocab_sizes in seq_vocab_sizes.items():
            ext_emb = seq_c_external if domain == item_id_seq_domain and seq_c_external else None
            ext_vocab = seq_c_external_vocab if domain == item_id_seq_domain and seq_c_external_vocab else None
            self.sequence_tokenizers[domain] = SequenceTokenizer(
                vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold,
                hash_bucket_size=hash_bucket_size,
                external_embeddings=ext_emb,
                external_vocab_sizes=ext_vocab,
            )

        # Total NS tokens (after concatenation)
        self.num_user = self.user_tokenizer.num_tokens + int(user_dense_dim > 0)
        self.num_item = self.item_tokenizer.num_tokens + int(item_dense_dim > 0)
        self.num_ns = self.num_user + self.num_item

        # ── Common hidden dims ──
        summary_hidden = d_model * hidden_mult
        sequence_hidden = d_model * hidden_mult
        interaction_hidden = d_model * hidden_mult * 2

        # ── Item-specific modules ──
        self.item_interaction = ItemInteractionArch(
            item_tokens=self.num_item,
            d_model=d_model,
            backbone=interaction_backbone,
            interaction_layers=max(1, interaction_layers // 2),
            dcn_layers=dcn_layers,
            low_rank=cross_low_rank,
            hidden_dim=summary_hidden,
            dropout=dropout_rate,
        )
        self.item_aware_gating = ItemAwareGating(d_model, hidden_mult=2)
        self.item_summarizer = NonSequenceSummarizer(
            input_tokens=self.num_item,
            output_tokens=num_cls_tokens,
            d_model=d_model,
            hidden_dim=summary_hidden,
            dropout=dropout_rate,
        )

        # ---- Initial NS summarizer (prepends CLS tokens once) ----
        self.initial_ns_summarizer = NonSequenceSummarizer(
            input_tokens=self.num_ns,
            output_tokens=num_cls_tokens,
            d_model=d_model,
            hidden_dim=summary_hidden,
            dropout=dropout_rate,
        )

        # ---- InterFormer blocks ----
        self.blocks = nn.ModuleList([
            InterFormerBlock(
                non_seq_tokens=self.num_ns,
                num_cls_tokens=num_cls_tokens,
                num_pma_tokens=num_pma_tokens,
                num_recent_tokens=num_recent_tokens,
                d_model=d_model,
                num_heads=num_heads,
                interaction_backbone=interaction_backbone,
                interaction_layers=interaction_layers,
                dcn_layers=dcn_layers,
                cross_low_rank=cross_low_rank,
                summary_hidden=summary_hidden,
                interaction_hidden=interaction_hidden,
                sequence_hidden=sequence_hidden,
                dropout=dropout_rate,
                rope_base=rope_base,
            )
            for _ in range(max(1, num_blocks))
        ])

        # ---- Final summarizers ----
        self.final_ns_summarizer = NonSequenceSummarizer(
            input_tokens=self.num_ns,
            output_tokens=num_cls_tokens,
            d_model=d_model,
            hidden_dim=summary_hidden,
            dropout=dropout_rate,
        )
        self.final_seq_summarizer = SequenceSummarizer(
            num_cls_tokens=num_cls_tokens,
            num_pma_tokens=num_pma_tokens,
            num_recent_tokens=num_recent_tokens,
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=summary_hidden,
            dropout=dropout_rate,
        )
        seq_summary_tokens = self.final_seq_summarizer.output_tokens

        # ---- Fusion gate & classifier ----
        fused_dim = (num_cls_tokens + seq_summary_tokens) * d_model
        self.final_gate = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * hidden_mult, action_num),
        )

    def _encode_non_sequence(
        self, inputs: ModelInput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(user_tokens, item_tokens)``."""
        user_parts = [self.user_tokenizer(inputs.user_int_feats)]
        user_dense = self.user_dense(inputs.user_dense_feats)
        if user_dense is not None:
            user_parts.append(user_dense)
        user_tokens = torch.cat(user_parts, dim=1)

        item_parts = [self.item_tokenizer(inputs.item_int_feats)]
        item_dense = self.item_dense(inputs.item_dense_feats)
        if item_dense is not None:
            item_parts.append(item_dense)
        item_tokens = torch.cat(item_parts, dim=1)
        return user_tokens, item_tokens

    def _encode_sequences(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        for domain in self.seq_domains:
            raw_sequence = inputs.seq_data[domain]
            seq_len = inputs.seq_lens[domain].to(raw_sequence.device)
            tokens = self.sequence_tokenizers[domain](
                raw_sequence, inputs.seq_time_buckets.get(domain)
            )
            # Position encoding: RoPE is inside SequenceArch (attention-level),
            # so we don't add sinusoidal here when use_rope=True.
            if not self.use_rope:
                tokens = tokens + sinusoidal_positions(
                    tokens.shape[1], self.d_model, tokens.device
                ).unsqueeze(0)
            sequences.append(tokens)
            masks.append(make_padding_mask(seq_len, raw_sequence.shape[2]))
            lengths.append(seq_len)
        return sequences, masks, lengths

    def _embed(self, inputs: ModelInput) -> torch.Tensor:
        user_tokens, item_tokens = self._encode_non_sequence(inputs)
        sequences, masks, lengths = self._encode_sequences(inputs)

        # ── Item-side processing ──
        # Dedicated item interaction (target-item feature crossing)
        item_tokens = self.item_interaction(item_tokens)
        # Item summary for gating cross-modal flow
        item_summary = self.item_summarizer(item_tokens)  # [B, cls, D]
        # Compute three gates from target item
        ns_gate, seq_gate, fuse_gate = self.item_aware_gating(
            item_summary.mean(dim=1)
        )

        # ── Concatenate into full NS tokens ──
        ns_tokens = torch.cat([user_tokens, item_tokens], dim=1)

        # ---- Prepend CLS tokens (NS summary) to each sequence ----
        ns_summary = self.initial_ns_summarizer(ns_tokens)  # [B, cls, D]
        # Modulate initial summary by item gate
        ns_summary = ns_summary * ns_gate.unsqueeze(1)
        sequences = [
            torch.cat([ns_summary, seq], dim=1) for seq in sequences
        ]
        # Adjust masks: CLS tokens are always valid (not masked)
        cls_mask = torch.zeros(
            masks[0].shape[0], self.num_cls_tokens,
            dtype=torch.bool, device=masks[0].device
        )
        masks = [torch.cat([cls_mask, m], dim=1) for m in masks]

        # ---- Stacked InterFormer blocks (with item gating) ----
        for block in self.blocks:
            ns_tokens, sequences = block(
                ns_tokens, sequences, masks, lengths,
                item_summary=item_summary,
                ns_gate=ns_gate, seq_gate=seq_gate,
            )

        # ---- Final summaries ----
        final_ns = self.final_ns_summarizer(ns_tokens)  # [B, cls, D]
        all_seq = torch.cat(sequences, dim=1)
        all_mask = safe_key_padding_mask(torch.cat(masks, dim=1))
        all_len = torch.stack(lengths, dim=1).sum(dim=1) + self.num_cls_tokens * len(lengths)
        final_seq = self.final_seq_summarizer(all_seq, all_len, all_mask)  # [B, seq_sum, D]

        # ---- Item-aware gated fusion ----
        ns_flat = final_ns.flatten(start_dim=1)
        seq_flat = final_seq.flatten(start_dim=1)
        fused = torch.cat([ns_flat, seq_flat], dim=-1)
        base_gate = self.final_gate(fused)
        # Blend base gate with item-derived fuse gate
        gate = base_gate * fuse_gate
        gate = gate.unsqueeze(1)  # [B, 1, d_model]
        
        ns_gated = final_ns * gate
        seq_gated = final_seq * (1.0 - gate)
        return torch.cat([ns_gated.flatten(start_dim=1), seq_gated.flatten(start_dim=1)], dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings