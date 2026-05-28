import os
import sys
from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

# Sibling package shim: iterative_reasoning lives next to recursive_reasoning on disk.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class IterativeReasoningLossHead(nn.Module):
    """Loss head for the iterative_reasoning-based TRM variant.

    Per iteration step the LM loss is computed against ``labels`` (with the
    same per-input mean reduction used by :class:`ACTLossHead`), giving a
    ``[num_steps, batch_size]`` per-latent loss tensor. That tensor is passed
    to ``IterativeReasoningModel.transform_loss`` to combine the bptt term and
    the halting KL term. ``halting_kl_coef`` from the model config controls
    the KL weight; pass ``halting_kl_coef`` here to override it.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        halting_kl_coef: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.halting_kl_coef = halting_kl_coef

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def _per_step_lm_loss(
        self,
        trajectory_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        loss_divisor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LM loss for every trajectory step.

        ``trajectory_logits`` has shape ``[T, B, seq_len, V]``; the result is
        ``[T, B]`` containing the per-input loss (sum over positions divided by
        the per-input valid-token count) at each iteration step.
        """

        num_steps, batch_size, seq_len, vocab_size = trajectory_logits.shape
        flat_logits = trajectory_logits.reshape(num_steps * batch_size, seq_len, vocab_size)
        flat_labels = labels.unsqueeze(0).expand(num_steps, batch_size, seq_len).reshape(num_steps * batch_size, seq_len)

        loss_kwargs = {"ignore_index": IGNORE_LABEL_ID}
        # stablemax_cross_entropy accepts valid_mask; softmax_cross_entropy does not.
        if "valid_mask" in self.loss_fn.__code__.co_varnames:
            loss_kwargs["valid_mask"] = (
                valid_mask.unsqueeze(0).expand(num_steps, batch_size, seq_len).reshape(num_steps * batch_size, seq_len)
            )

        per_token = self.loss_fn(flat_logits, flat_labels, **loss_kwargs).reshape(num_steps, batch_size, seq_len)
        # Match ACTLossHead's per-input normalization: sum over positions / per-input count.
        return (per_token / loss_divisor.view(1, batch_size, 1)).sum(dim=-1)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        ir_output = outputs["ir_output"]
        trajectory_logits = outputs["trajectory_logits"]
        batch_size = labels.shape[0]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor.unsqueeze(-1)).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        per_step_loss = self._per_step_lm_loss(trajectory_logits, labels, mask, loss_divisor)

        ir_model = self.model.inner.iterative_model
        ir_cfg = ir_model.config

        if ir_cfg.alternative_training:
            # SAC-style reparameterized-KL branch: let the iterative_reasoning
            # module assemble bptt + log_pi (+ entropy) for us.
            if self.halting_kl_coef is None:
                transformed = ir_model.transform_loss(per_step_loss, ir_output)
            else:
                transformed = ir_model.transform_loss(
                    per_step_loss, ir_output, halting_kl_coef=self.halting_kl_coef
                )
            total_loss = transformed.total_loss * batch_size
            metrics["lm_loss"] = (transformed.bptt_loss * batch_size).detach()
            metrics["halting_kl_loss"] = (transformed.halting_kl_loss * batch_size).detach()
            if transformed.log_pi_term is not None:
                metrics["log_pi_term"] = (transformed.log_pi_term * batch_size).detach()
            if transformed.entropy_term is not None:
                metrics["entropy_term"] = (transformed.entropy_term * batch_size).detach()
        else:
            # Standard halting-KL branch with TRM-style loss scaling.
            #
            # iterative_reasoning's transform_loss bptt is a global mean over
            # (T, B) valid entries — appropriate for PPO-style models, but
            # ~1/T smaller than TRM's ``sum_b per_input_mean[b]`` when scaled
            # back by batch_size. Here we replicate TRM's reduction directly:
            # per-input mean over valid T (so each input weights equally),
            # then sum over the batch (so the pretrain loop's
            # 1/global_batch_size scaling yields the standard mean).
            if ir_cfg.loss_on_all_latents:
                bptt_mask = torch.ones_like(ir_output.trajectory_mask, dtype=torch.bool)
            else:
                bptt_mask = ir_output.trajectory_mask
            bptt_mask_f = bptt_mask.to(per_step_loss.dtype)
            per_input_lm = (per_step_loss * bptt_mask_f).sum(dim=0) / bptt_mask_f.sum(dim=0).clamp_min(1)
            bptt_loss_sum = per_input_lm.sum()  # sum over batch (TRM convention)

            # Halting KL: iterative_reasoning's halting_kl_loss returns a mean
            # over the batch, so multiply by batch_size to convert to a
            # sum-over-batch consistent with bptt_loss_sum.
            kl_z_all_steps = ir_cfg.kl_z_all_steps or ir_cfg.loss_on_all_latents
            halting_kl_mean, _, _ = ir_model.halting_kl_loss(
                per_step_loss,
                ir_output,
                z_all_steps=kl_z_all_steps,
                mode=ir_cfg.kl_mode,
            )
            halting_kl_sum = halting_kl_mean * batch_size

            halting_kl_coef = ir_cfg.halting_kl_coef if self.halting_kl_coef is None else self.halting_kl_coef
            total_loss = bptt_loss_sum + halting_kl_coef * halting_kl_sum
            metrics["lm_loss"] = bptt_loss_sum.detach()
            metrics["halting_kl_loss"] = halting_kl_sum.detach()

        # ---- iterative_reasoning abstraction logs ----
        # Mirrors the recursive_info entries logged by
        # cleanrl/cleanrl/ppo_atari_envpool.py and adds per-step latent
        # change norms. All metrics here are sum-over-valid-examples; the
        # pretrain loop divides by `count` (sum of valid_metrics) downstream.
        with torch.no_grad():
            valid_f = valid_metrics.to(torch.float32)
            # Number of latent steps the model spent on each input. trajectory_lengths
            # is the 0-based halt index, so steps = lengths + 1.
            halt_steps = ir_output.trajectory_lengths.to(torch.float32) + 1.0
            metrics["trajectory_length"] = (halt_steps * valid_f).sum()
            # Log probability of halting at the selected step (per input).
            metrics["trajectory_log_prob"] = (
                ir_output.selected_trajectory_log_prob.to(torch.float32) * valid_f
            ).sum()

            halt_probs = ir_output.halt_probabilities.to(torch.float32)  # [T, B]
            traj_mask = ir_output.trajectory_mask  # [T, B], bool
            traj_mask_f = traj_mask.to(halt_probs.dtype)
            # Per-input mean over valid steps, summed across batch.
            per_input_halt_p = (halt_probs * traj_mask_f).sum(dim=0) / traj_mask_f.sum(dim=0).clamp_min(1)
            metrics["halt_probability_mean"] = (per_input_halt_p * valid_f).sum()
            # Halt probability at the selected halt step.
            selected_halt_p = ir_output.halt_probabilities.to(torch.float32).gather(
                0, ir_output.trajectory_lengths.unsqueeze(0)
            ).squeeze(0)
            metrics["halt_probability_selected"] = (selected_halt_p * valid_f).sum()

            # Average change in latent ||z_i - z_{i-1}|| per recursion step.
            # trajectory_latent_delta_norms has shape [T, B]; log one scalar per step
            # plus an overall mean.
            delta_norms = ir_output.trajectory_latent_delta_norms.to(torch.float32)
            for t in range(delta_norms.shape[0]):
                metrics[f"latent_delta_norm_step_{t + 1}"] = (delta_norms[t] * valid_f).sum()
            metrics["latent_delta_norm_mean"] = (delta_norms.mean(dim=0) * valid_f).sum()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs and torch.is_tensor(outputs[k])}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

