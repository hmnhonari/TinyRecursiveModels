"""TRM-style embedding + initial transformer encoder driving the reusable
``iterative_reasoning`` halting module.

Two architectures are exposed via ``cell_type``:

- ``cell_type='mlp'``: per-token MLP cell that concatenates ``[input_encoding,
  latent]`` and produces a latent update. Training uses BPTT (the latent is not
  detached between iteration steps) and the bptt loss covers the latent at the
  sampled halt step.

- ``cell_type='transformer'``: causal RoPE transformer cell that takes
  ``latent + input_encoding`` as the next token sequence. The latent is
  detached between steps (no BPTT) and the per-step loss is applied to every
  latent in the trajectory.

The halting + KL machinery is provided by
:class:`iterative_reasoning.IterativeReasoningModel`. The cell architectures
above plug into it as ``recursive_cell``s.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import nn

# Sibling package shim: iterative_reasoning lives next to recursive_reasoning on disk.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from iterative_reasoning import (  # noqa: E402
    IterativeReasoningConfig,
    IterativeReasoningModel,
    IterativeReasoningOutput,
    loss_from_trajectory_outputs,
    masked_mean,
)

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


# =====================================================================
# Config and carry
# =====================================================================


# Field names IterativeReasoningConfig accepts. Used to forward yaml fields
# straight into the dataclass without re-declaring every option here. Computed
# once at import time so a new field on IterativeReasoningConfig automatically
# becomes forwardable from this integration's config.
_IR_FIELD_NAMES = {f.name for f in dataclasses.fields(IterativeReasoningConfig)}


class IterativeReasoningModel_ACTV1Config(BaseModel):
    """Configuration for the iterative-reasoning based TRM variant.

    Only fields that this integration uses *directly* (input embedding,
    encoder, cell architecture, halt-head init) are declared explicitly. Every
    field accepted by :class:`IterativeReasoningConfig` —
    ``halt_eps``, ``force_final_halt``, ``stochastic_halting``,
    ``halting_kl_coef``, ``kl_z_all_steps``, ``kl_mode``, ``connection_type``,
    ``gumbel_sigmoid_halting``, ``gumbel_sigmoid_temperature``, ... — is
    accepted via pydantic extras (``model_config = ConfigDict(extra='allow')``)
    and forwarded to ``IterativeReasoningConfig`` at construction. This keeps a
    single source of truth for those defaults in the standalone
    ``iterative_reasoning`` module. ``halt_max_steps`` is the one renamed alias
    (forwarded as ``max_steps``).
    """

    model_config = ConfigDict(extra="allow")

    # --- pretrain.py / dataset-driven required fields ---
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # --- shared transformer hyperparams (used by encoder and transformer cell) ---
    hidden_size: int
    expansion: float
    num_heads: int
    encoder_layers: int = 2
    cell_layers: int = 2
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Encoder positional encoding (always on by default); the transformer cell
    # has its own independent positional encoding via cell_pos_encodings.
    pos_encodings: str = "rope"
    cell_pos_encodings: str = "rope"  # 'rope' or 'none'; transformer cell only
    puzzle_emb_len: int = 16

    # --- iterative reasoning integration knobs ---
    cell_type: str = "transformer"  # 'mlp' or 'transformer'
    halt_max_steps: int = 4
    halt_bias_init: float = -5.0
    # Optional[bool] so None means "use the cell_type default" (mlp -> False,
    # transformer -> True). Forwarded as plain bool to IterativeReasoningConfig.
    detach_latent_between_steps: Optional[bool] = None
    loss_on_all_latents: Optional[bool] = None

    forward_dtype: str = "bfloat16"


@dataclass
class IterativeReasoningInnerCarry:
    pass


@dataclass
class IterativeReasoningCarry:
    inner_carry: IterativeReasoningInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# =====================================================================
# Building blocks
# =====================================================================


class _EncoderBlock(nn.Module):
    """TRM-style non-causal attention block used by the initial encoder."""

    def __init__(self, config: IterativeReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class _CausalBlock(nn.Module):
    """Causal RoPE self-attention block used by the transformer reasoning cell."""

    def __init__(self, config: IterativeReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=True,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class _InputEncoder(nn.Module):
    """Token + puzzle embedding (TRM-style) followed by ``encoder_layers``
    non-causal attention blocks.

    Produces ``[batch, seq_len + puzzle_emb_len, hidden_size]``.
    """

    def __init__(self, config: IterativeReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )

        self.puzzle_emb_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if config.puzzle_emb_len == 0
            else config.puzzle_emb_len
        )
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.blocks = nn.ModuleList(
            [_EncoderBlock(config) for _ in range(config.encoder_layers)]
        )

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),
                    embedding,
                ),
                dim=-2,
            )
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def cos_sin(self) -> Optional[CosSin]:
        return self.rotary_emb() if hasattr(self, "rotary_emb") else None

    def forward(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        hidden = self._input_embeddings(input, puzzle_identifiers)
        cos_sin = self.cos_sin()
        for block in self.blocks:
            hidden = block(cos_sin=cos_sin, hidden_states=hidden)
        return hidden


# =====================================================================
# Reasoning cells — both return (latent_update, halt_logit) so they plug
# directly into IterativeReasoningModel.
# =====================================================================


def _build_halt_head(hidden_size: int, init_bias: float) -> CastedLinear:
    halt_head = CastedLinear(hidden_size, 1, bias=True)
    with torch.no_grad():
        halt_head.weight.zero_()
        halt_head.bias.fill_(init_bias)  # type: ignore[union-attr]
    return halt_head


class _MLPCell(nn.Module):
    """Per-token MLP cell.

    Concatenates ``[input_encoding, latent]`` along the channel dim and passes
    the result through ``cell_layers`` stages of ``Linear -> SwiGLU + RMSNorm``.
    The halt logit is read from the first sequence position (the puzzle slot).
    """

    def __init__(self, config: IterativeReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.input_proj = CastedLinear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.mlps = nn.ModuleList(
            [SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion) for _ in range(config.cell_layers)]
        )
        self.halt_head = _build_halt_head(config.hidden_size, config.halt_bias_init)

    def forward(self, input_encoding: torch.Tensor, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_proj(torch.cat([input_encoding, latent], dim=-1))
        for mlp in self.mlps:
            hidden = rms_norm(hidden + mlp(hidden), variance_epsilon=self.norm_eps)
        halt_logit = self.halt_head(hidden[:, 0]).squeeze(-1).to(torch.float32)
        return hidden, halt_logit


class _TransformerCell(nn.Module):
    """Causal transformer cell, optionally with RoPE positional encoding.

    Sets ``hidden = latent + input_encoding`` and runs it through ``cell_layers``
    causal self-attention blocks. The cell's positional encoding is controlled
    by ``config.cell_pos_encodings`` and is independent of the encoder's
    ``config.pos_encodings`` — passing ``rotary_emb=None`` disables RoPE inside
    the cell's attention.
    """

    def __init__(
        self,
        config: IterativeReasoningModel_ACTV1Config,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.rotary_emb = rotary_emb
        self.blocks = nn.ModuleList([_CausalBlock(config) for _ in range(config.cell_layers)])
        self.halt_head = _build_halt_head(config.hidden_size, config.halt_bias_init)

    def forward(self, input_encoding: torch.Tensor, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = latent + input_encoding
        cos_sin = self.rotary_emb() if self.rotary_emb is not None else None
        for block in self.blocks:
            hidden = block(cos_sin=cos_sin, hidden_states=hidden)
        halt_logit = self.halt_head(hidden[:, 0]).squeeze(-1).to(torch.float32)
        return hidden, halt_logit


# =====================================================================
# Inner model — embed -> encoder -> IterativeReasoningModel -> lm_head
# =====================================================================


class _Inner(nn.Module):
    def __init__(self, config: IterativeReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.encoder = _InputEncoder(config)
        self.puzzle_emb_len = self.encoder.puzzle_emb_len

        if config.cell_type == "mlp":
            cell: nn.Module = _MLPCell(config)
            detach_default = False
            loss_on_all_default = False
        elif config.cell_type == "transformer":
            cell_rotary_emb: Optional[RotaryEmbedding] = None
            if config.cell_pos_encodings == "rope":
                # Separate RotaryEmbedding instance from the encoder's, so the
                # cell's positional encoding is independently configurable.
                cell_rotary_emb = RotaryEmbedding(
                    dim=config.hidden_size // config.num_heads,
                    max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                    base=config.rope_theta,
                )
            elif config.cell_pos_encodings != "none":
                raise ValueError(
                    f"Unsupported cell_pos_encodings={config.cell_pos_encodings!r}. Expected 'rope' or 'none'."
                )
            cell = _TransformerCell(config, rotary_emb=cell_rotary_emb)
            detach_default = True
            loss_on_all_default = True
        else:
            raise ValueError(f"Unsupported cell_type={config.cell_type!r}. Expected 'mlp' or 'transformer'.")

        detach = detach_default if config.detach_latent_between_steps is None else config.detach_latent_between_steps
        loss_on_all = loss_on_all_default if config.loss_on_all_latents is None else config.loss_on_all_latents

        # Default latent_dim to hidden_size so the latent_log_std_head (when
        # stochastic_latents=True with source='output') has a valid input/output
        # dim. The user can still override by setting latent_dim explicitly.
        extras = dict(getattr(config, "__pydantic_extra__", None) or {})
        latent_dim = extras.get("latent_dim", config.hidden_size)

        ir_config = self._build_iterative_config(
            config,
            detach_latent_between_steps=detach,
            loss_on_all_latents=loss_on_all,
            latent_dim=latent_dim,
        )
        self.iterative_model = IterativeReasoningModel(
            config=ir_config,
            recursive_cell=cell,
        )

        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def puzzle_emb(self):
        return self.encoder.puzzle_emb

    @staticmethod
    def _build_iterative_config(
        config: IterativeReasoningModel_ACTV1Config,
        **overrides: Any,
    ) -> IterativeReasoningConfig:
        """Build an :class:`IterativeReasoningConfig` from this integration's
        pydantic config.

        Forwards any field whose name appears in
        :data:`_IR_FIELD_NAMES` (i.e. any field defined on
        ``IterativeReasoningConfig``), looking first at the declared pydantic
        fields and then at pydantic extras. This means new fields on
        ``IterativeReasoningConfig`` are picked up automatically — no need to
        re-declare them here. ``halt_max_steps`` is renamed to ``max_steps``;
        ``overrides`` win over everything else (used to inject the
        cell-type-specific ``detach_latent_between_steps`` /
        ``loss_on_all_latents`` defaults).
        """

        extras = dict(getattr(config, "__pydantic_extra__", None) or {})
        kwargs: Dict[str, Any] = {}
        for name in _IR_FIELD_NAMES:
            if hasattr(config, name) and name not in {"max_steps", "latent_dim"}:
                kwargs[name] = getattr(config, name)
            elif name in extras:
                kwargs[name] = extras[name]
            # else: let IterativeReasoningConfig use its own default
        kwargs["max_steps"] = config.halt_max_steps
        kwargs.update(overrides)
        return IterativeReasoningConfig(**kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_encoding = self.encoder(batch["inputs"], batch["puzzle_identifiers"])
        ir_output: IterativeReasoningOutput = self.iterative_model(input_encoding=input_encoding)

        # Strip puzzle-emb positions to match TRM's lm_head convention.
        final_logits = self.lm_head(ir_output.final_latent)[:, self.puzzle_emb_len:]
        trajectory_logits = self.lm_head(ir_output.trajectory_latents)[..., self.puzzle_emb_len:, :]

        return {
            "logits": final_logits,
            "trajectory_logits": trajectory_logits,
            "halt_logits": ir_output.halt_logits,
            "halt_probabilities": ir_output.halt_probabilities,
            "trajectory_lengths": ir_output.trajectory_lengths,
            "trajectory_mask": ir_output.trajectory_mask,
            "selected_trajectory_log_prob": ir_output.selected_trajectory_log_prob,
            "ir_output": ir_output,
        }


# =====================================================================
# ACT wrapper — single forward = full max_steps rollout. ``halted`` is
# always True so the pretrain loop moves to the next batch each step.
# =====================================================================


class IterativeReasoningModel_ACTV1(nn.Module):
    """ACT-compatible wrapper around :class:`_Inner`.

    Single-iteration contract: each call to ``forward`` runs the full
    ``halt_max_steps`` rollout *inside* :class:`IterativeReasoningModel` and
    returns ``halted=True`` for every sequence. We deliberately do not iterate
    the outer ``carry`` loop (the way TRM's ACT wrapper does H_cycles steps
    across calls) — all recursion lives inside ``iterative_reasoning`` so we
    never run two nested iteration loops over the same batch. The pretrain
    loop therefore moves on to the next batch after one forward call.
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = IterativeReasoningModel_ACTV1Config(**config_dict)
        self.inner = _Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> IterativeReasoningCarry:
        batch_size = batch["inputs"].shape[0]
        return IterativeReasoningCarry(
            inner_carry=IterativeReasoningInnerCarry(),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: IterativeReasoningCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[IterativeReasoningCarry, Dict[str, torch.Tensor]]:
        # One call = one full iterative_reasoning rollout, so always use the fresh batch.
        outputs = self.inner(batch)
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        steps = outputs["trajectory_lengths"].to(torch.int32) + 1
        halted = torch.ones((batch_size,), dtype=torch.bool, device=device)
        new_carry = IterativeReasoningCarry(
            inner_carry=IterativeReasoningInnerCarry(),
            steps=steps,
            halted=halted,
            current_data=dict(batch),
        )
        return new_carry, outputs
