"""Transformer components for structured state prediction and correction."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from stageworld.errors import DataContractError


def safe_cross_attention(
    attention: nn.MultiheadAttention,
    query: Tensor,
    key_value: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cross-attend without NaNs for rows containing no valid keys."""

    if key_value.ndim != 3 or valid.shape != key_value.shape[:2]:
        raise DataContractError(
            code="ATTENTION_SHAPE_MISMATCH",
            message="Cross-attention key/value and valid mask do not align.",
        )
    row_has_value = valid.any(dim=1)
    safe_valid = valid.clone()
    if (~row_has_value).any():
        safe_valid[~row_has_value, 0] = True
    safe_values = key_value.masked_fill(~valid.unsqueeze(-1), 0.0)
    result, _ = attention(
        query,
        safe_values,
        safe_values,
        key_padding_mask=~safe_valid,
        need_weights=False,
    )
    result = result * row_has_value[:, None, None].to(result.dtype)
    return result, row_has_value


class ContinuousTimeEncoder(nn.Module):
    frequencies: Tensor

    def __init__(self, output_dim: int, fourier_bands: int = 8, scale: float = 365.25):
        super().__init__()
        self.scale = float(scale)
        frequencies = 2.0 ** torch.arange(fourier_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(1 + 2 * fourier_bands, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, times: Tensor) -> Tensor:
        normalized = times.float().unsqueeze(-1) / self.scale
        angles = 2.0 * math.pi * normalized * self.frequencies
        encoded = torch.cat((normalized, angles.sin(), angles.cos()), dim=-1)
        return self.projection(encoded)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        values = values + attended
        return values + self.ffn(self.norm2(values))


class TokenResampler(nn.Module):
    """Learned queries over genuine input tokens; never reshapes global vectors."""

    def __init__(self, dim: int, output_tokens: int, heads: int, blocks: int, dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_tokens, dim) * (dim**-0.5))
        self.query_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.blocks = nn.ModuleList(SelfAttentionBlock(dim, heads, dropout) for _ in range(blocks))

    def forward(self, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch = values.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, row_valid = safe_cross_attention(
            self.cross_attention,
            self.query_norm(queries),
            self.input_norm(values),
            valid,
        )
        output = queries + attended
        for block in self.blocks:
            output = block(output)
        output_valid = row_valid[:, None].expand(-1, output.shape[1])
        output = output * output_valid.unsqueeze(-1).to(output.dtype)
        return output, output_valid


class StateInitializer(nn.Module):
    def __init__(self, dim: int, memory_tokens: int, heads: int, blocks: int, dropout: float):
        super().__init__()
        self.memory_queries = nn.Parameter(torch.randn(memory_tokens, dim) * (dim**-0.5))
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(SelfAttentionBlock(dim, heads, dropout) for _ in range(blocks))

    def forward(self, observations: Tensor, valid: Tensor) -> Tensor:
        if not valid.any(dim=1).all():
            raise DataContractError(
                code="EMPTY_BASELINE_OBSERVATION",
                message="Every patient requires at least one valid baseline token.",
            )
        memory = self.memory_queries.unsqueeze(0).expand(observations.shape[0], -1, -1)
        attended, _ = safe_cross_attention(
            self.cross_attention,
            self.query_norm(memory),
            self.input_norm(observations),
            valid,
        )
        memory = memory + attended
        for block in self.blocks:
            memory = block(memory)
        return memory


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_block = SelfAttentionBlock(dim, heads, dropout)
        self.memory_norm = nn.LayerNorm(dim)
        self.condition_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self, memory: Tensor, conditions: Tensor, condition_valid: Tensor, time: Tensor
    ) -> Tensor:
        memory = self.self_block(memory)
        attended, _ = safe_cross_attention(
            self.cross_attention,
            self.memory_norm(memory),
            self.condition_norm(conditions),
            condition_valid,
        )
        memory = memory + attended
        gamma, beta = self.film(time).chunk(2, dim=-1)
        conditioned = self.ffn_norm(memory) * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return memory + self.ffn(conditioned)


class StateTransition(nn.Module):
    def __init__(self, dim: int, heads: int, blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedTransitionBlock(dim, heads, dropout) for _ in range(blocks)
        )

    def forward(self, memory: Tensor, conditions: Tensor, valid: Tensor, time: Tensor) -> Tensor:
        for block in self.blocks:
            memory = block(memory, conditions, valid, time)
        return memory


class ObservationUpdateBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.memory_norm = nn.LayerNorm(dim)
        self.observation_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.post = SelfAttentionBlock(dim, heads, dropout)

    def forward(self, memory: Tensor, observations: Tensor, valid: Tensor) -> Tensor:
        attended, row_valid = safe_cross_attention(
            self.cross_attention,
            self.memory_norm(memory),
            self.observation_norm(observations),
            valid,
        )
        gate = self.gate(torch.cat((memory, attended), dim=-1))
        updated = memory + gate * attended
        updated = self.post(updated)
        return torch.where(row_valid[:, None, None], updated, memory)


class ObservationUpdater(nn.Module):
    def __init__(self, dim: int, heads: int, blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            ObservationUpdateBlock(dim, heads, dropout) for _ in range(blocks)
        )

    def forward(self, memory: Tensor, observations: Tensor, valid: Tensor) -> Tensor:
        for block in self.blocks:
            memory = block(memory, observations, valid)
        return memory


class GaussianStateHead(nn.Module):
    def __init__(self, dim: int, stochastic_dim: int):
        super().__init__()
        self.distribution_head = nn.Linear(dim, stochastic_dim * 2)

    def forward(
        self,
        memory: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.distribution_head(memory).chunk(2, dim=-1)
        log_std = log_std.clamp(min=-8.0, max=3.0)
        if deterministic:
            sample = mean
        else:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            sample = mean + log_std.exp() * noise
        return mean, log_std, sample


class FutureObservationDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        output_tokens: int,
        output_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_tokens, dim) * (dim**-0.5))
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, output_dim * 2)

    def forward(self, state_tokens: Tensor) -> tuple[Tensor, Tensor]:
        query = self.queries.unsqueeze(0).expand(state_tokens.shape[0], -1, -1)
        decoded, _ = self.attention(
            query,
            self.norm(state_tokens),
            self.norm(state_tokens),
            need_weights=False,
        )
        mean, log_std = self.output(decoded).chunk(2, dim=-1)
        return mean, log_std.clamp(min=-6.0, max=2.0)


class SharedSurvivalDecoder(nn.Module):
    def __init__(self, dim: int, intervals: int, causes: int, heads: int, dropout: float):
        super().__init__()
        self.interval_queries = nn.Parameter(torch.randn(intervals, dim) * (dim**-0.5))
        self.stage_embedding = nn.Embedding(3, dim)
        self.time_encoder = ContinuousTimeEncoder(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, causes))

    def forward(
        self, state_tokens: Tensor, query_time: Tensor, stage_index: int,
        valid: Tensor | None = None,
    ) -> Tensor:
        batch = state_tokens.shape[0]
        queries = self.interval_queries.unsqueeze(0).expand(batch, -1, -1)
        stage = self.stage_embedding(
            torch.full((batch,), stage_index, dtype=torch.long, device=state_tokens.device)
        )
        queries = queries + stage[:, None, :] + self.time_encoder(query_time)[:, None, :]
        decoded, _ = self.attention(
            queries,
            self.norm(state_tokens),
            self.norm(state_tokens),
            key_padding_mask=None if valid is None else ~valid,
            need_weights=False,
        )
        return torch.nn.functional.softplus(self.output(decoded)) + 1e-6
