"""
GrafoPropagation v26-APEX — System-2 Latent Search
ResidualWorldModel · PolicyValueHead · GumbelMCTS · System2LatentSearch

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import RMSNorm, trunc_normal_
from .transformer import TransformerBlock


# ─────────────────────────────────────────────────────────────────────────────
# Residual World Model
# ─────────────────────────────────────────────────────────────────────────────

class ResidualWorldModel(nn.Module):
    """
    Learned transition function:  state' = f(state, action).
    Each action applies a low-rank linear perturbation in the latent space,
    followed by a coherence transformer to restore consistency.
    """

    def __init__(
        self,
        d: int,
        n_actions: int,
        n_heads: int,
        head_dim: int,
        d_ff: int,
        dropout: float,
        kappa_init: float = 4.0,
        use_kappa_weights: bool = True,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.input_norm = RMSNorm(d)

        # Per-action weight matrices
        self.action_W = nn.Parameter(torch.empty(n_actions, d, d))
        for a in range(n_actions):
            nn.init.xavier_uniform_(self.action_W[a], gain=0.3)
        self.action_scale = nn.Parameter(torch.full((n_actions,), 0.1))
        self.action_proj = nn.Sequential(
            nn.Linear(d, d, bias=False), RMSNorm(d),
            nn.GELU(approximate="tanh"),
        )
        nn.init.xavier_uniform_(self.action_proj[0].weight, gain=0.5)
        self.post_norm = RMSNorm(d)
        self.coherence = TransformerBlock(
            d, n_heads, head_dim, d_ff, dropout, 0.0,
            kappa_init, use_kappa_weights,
            rope=None, dual_scale=False, asymmetric_qk=False,
        )

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state      : (B, K, D) latent state
        action_idx : (B,) integer action indices

        Returns
        -------
        (B, K, D) next latent state
        """
        B, K, D = state.shape
        normed = self.input_norm(state)
        W = self.action_W[action_idx]  # (B, D, D)
        scale = self.action_scale[action_idx].view(B, 1, 1)
        delta = torch.bmm(normed, W.transpose(-1, -2)) * scale
        delta = self.action_proj(delta)
        raw = state + delta
        out = F.normalize(raw.float(), p=2, dim=-1, eps=1e-8).to(raw.dtype)
        out = self.post_norm(out * math.sqrt(D))
        out, _ = self.coherence(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Policy-Value Head
# ─────────────────────────────────────────────────────────────────────────────

class PolicyValueHead(nn.Module):
    """Shared-trunk policy π(a|s) and value V(s) predictor."""

    def __init__(self, d: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d, d), RMSNorm(d), nn.GELU(approximate="tanh"))
        self.policy_head = nn.Linear(d, n_actions)
        self.value_head = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, state: torch.Tensor):
        h = self.shared(state.mean(dim=1))
        pi = F.softmax(self.policy_head(h), dim=-1)
        v = torch.tanh(self.value_head(h))
        return pi, v


# ─────────────────────────────────────────────────────────────────────────────
# Gumbel MCTS
# ─────────────────────────────────────────────────────────────────────────────

class GumbelMCTS(nn.Module):
    """
    Sequential Halving with Gumbel noise (Danihelka et al., 2022).
    Deterministically selects the best action after `num_simulations`
    model-based rollouts, halving the candidate set at each phase.

    Training mode: differentiable soft-search via weighted mixture.
    Inference mode: hard sequential-halving search.
    """

    def __init__(
        self,
        world_model: nn.Module,
        actor_critic: nn.Module,
        n_actions: int,
        num_simulations: int,
        rollout_depth: int,
        device: torch.device,
    ):
        super().__init__()
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.n_actions = n_actions
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.device = device
        self.n_phases = max(1, int(math.ceil(math.log2(max(n_actions, 2)))))

    def _rollout_value(self, states: torch.Tensor) -> torch.Tensor:
        cur = states
        for _ in range(self.rollout_depth - 1):
            pi, _ = self.actor_critic(cur)
            cur = self.world_model(cur, pi.argmax(-1))
        _, v = self.actor_critic(cur)
        return v.squeeze(-1)

    @torch.no_grad()
    def search(self, initial_states: torch.Tensor):
        """Hard sequential-halving search (inference)."""
        B, K, D = initial_states.shape
        A, dev = self.n_actions, self.device

        root_pi, _ = self.actor_critic(initial_states)
        alpha = torch.full((A,), 0.3 / A, device=dev)
        dir_noise = torch.distributions.Dirichlet(alpha).sample((B,))
        noisy_pi = 0.75 * root_pi + 0.25 * dir_noise
        log_pi = torch.log(noisy_pi.clamp(min=1e-8))
        gumbel = -torch.log(
            -torch.log(torch.rand(B, A, device=dev, dtype=torch.float32).clamp(1e-10, 1 - 1e-10))
        )
        q_sum = torch.zeros(B, A, device=dev)
        q_count = torch.zeros(B, A, device=dev)
        active = torch.ones(B, A, dtype=torch.bool, device=dev)

        sims_per_phase = max(1, self.num_simulations // self.n_phases)
        for phase in range(self.n_phases):
            n_active_global = int(active.any(0).sum().item())
            if n_active_global <= 1:
                break
            sims_per_cand = max(1, sims_per_phase // n_active_global)
            pairs = active.nonzero(as_tuple=False)
            if pairs.numel() == 0:
                break
            for _ in range(sims_per_cand):
                b_idx = pairs[:, 0]
                a_idx = pairs[:, 1].long()
                children = self.world_model(initial_states[b_idx], a_idx)
                values = self._rollout_value(children)
                q_sum.index_put_((b_idx, a_idx), q_sum[b_idx, a_idx] + values, accumulate=False)
                q_count.index_put_((b_idx, a_idx), q_count[b_idx, a_idx] + 1.0, accumulate=False)

            visited = q_count > 0
            q_mean = q_sum / q_count.clamp(min=1e-8)
            q_ref = (
                (q_mean * visited.float()).sum(-1, keepdim=True)
                / visited.float().sum(-1, keepdim=True).clamp(min=1.0)
            )
            q_compl = torch.where(visited, q_mean, q_ref.expand_as(q_mean))
            c_scale = math.log(sims_per_cand * (phase + 1) + 1)
            scores = log_pi + gumbel + c_scale * q_compl
            scores = scores.masked_fill(~active, float("-inf"))
            n_keep = max(1, math.ceil(n_active_global / 2))
            mean_sc = scores.mean(0).masked_fill(~active.any(0), float("-inf"))
            _, top_k = torch.topk(mean_sc, min(n_keep, A))
            new_act = torch.zeros(B, A, dtype=torch.bool, device=dev)
            new_act[:, top_k] = True
            active = new_act & active

        visited = q_count > 0
        q_mean = q_sum / q_count.clamp(min=1e-8)
        q_ref = (
            (q_mean * visited.float()).sum(-1, keepdim=True)
            / visited.float().sum(-1, keepdim=True).clamp(min=1.0)
        )
        q_compl = torch.where(visited, q_mean, q_ref.expand_as(q_mean))
        c_final = math.log(self.num_simulations + 1)
        f_scores = log_pi + gumbel + c_final * q_compl
        f_scores = f_scores.masked_fill(~active, float("-inf"))
        best_a = f_scores.argmax(-1)
        refined = self.world_model(initial_states, best_a)
        improved_policy = F.softmax(f_scores, dim=-1)
        return refined, improved_policy

    @torch.enable_grad()
    def soft_search(self, initial_states: torch.Tensor) -> torch.Tensor:
        """Differentiable soft-search (training): Gumbel-weighted mixture of children."""
        B, K, D = initial_states.shape
        pi, _ = self.actor_critic(initial_states)
        log_pi = torch.log(pi.clamp(min=1e-8))
        gumbel = -torch.log(
            -torch.log(torch.rand_like(pi).clamp(1e-10, 1 - 1e-10))
        )
        weights = F.softmax((log_pi + gumbel) / 0.5, dim=-1)

        S_exp = initial_states.repeat_interleave(self.n_actions, dim=0)
        a_exp = torch.arange(self.n_actions, device=self.device).repeat(B)
        children = self.world_model(S_exp, a_exp)
        children = children.view(B, self.n_actions, K, D)
        return (children * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# System-2 Latent Search
# ─────────────────────────────────────────────────────────────────────────────

class System2LatentSearch(nn.Module):
    """
    Iterative latent search block inspired by System-2 cognitive processes.

    Algorithm
    ---------
    1. Initialise K "thought tokens" from a learned base.
    2. For each branch: concatenate thoughts + input, run a search block,
       evaluate via vMF concentration score.
    3. Merge branches by softmax-weighted combination.
    4. Refine final thoughts with GumbelMCTS (soft in training, hard at eval).
    5. Prepend refined thoughts to the sequence.
    """

    def __init__(
        self,
        K: int,
        d: int,
        n_heads: int,
        head_dim: int,
        d_ff: int,
        dropout: float,
        branches: int,
        max_iters: int,
        epsilon: float,
        n_actions: int,
        mcts_sims: int,
        rollout_depth: int,
        device: torch.device,
        kappa_init: float = 4.0,
        use_kappa_weights: bool = True,
    ):
        super().__init__()
        self.K = K
        self.branches = branches
        self.max_iters = max_iters
        self.eps = epsilon

        self.base_think = nn.Parameter(torch.empty(1, K, d))
        trunc_normal_(self.base_think, std=0.02)
        self.branch_noise = nn.Parameter(torch.randn(branches, 1, K, d) * 0.05)

        self.search_block = TransformerBlock(
            d, n_heads, head_dim, d_ff, dropout, 0.0,
            kappa_init, use_kappa_weights,
            rope=None, dual_scale=False, asymmetric_qk=False,
        )
        self.norm = RMSNorm(d)

        self.world_model = ResidualWorldModel(
            d, n_actions, n_heads, head_dim, d_ff, dropout,
            kappa_init, use_kappa_weights,
        )
        self.actor_critic = PolicyValueHead(d, n_actions)
        self.gumbel_mcts = GumbelMCTS(
            self.world_model, self.actor_critic,
            n_actions, mcts_sims, rollout_depth, device,
        )

    def diversity_loss(self) -> torch.Tensor:
        """Penalise thought-token alignment to encourage diverse reasoning paths."""
        if self.K <= 1:
            return self.base_think.new_zeros(())
        t = F.normalize(self.base_think.squeeze(0), dim=-1)  # (K, D)
        return (t @ t.T).triu(1).pow(2).sum() * (2.0 / (self.K * (self.K - 1)))

    def forward(self, seq_x: torch.Tensor, fmask=None):
        B, T, D = seq_x.shape

        ext_mask = None
        if fmask is not None:
            ext_mask = torch.cat([
                torch.zeros(B, self.K, device=fmask.device, dtype=fmask.dtype),
                fmask,
            ], dim=1)

        best = self.base_think.expand(B, -1, -1)

        for it in range(self.max_iters):
            prev = best.detach()
            branch_t = best.unsqueeze(0) + self.branch_noise  # (branches, B, K, D)
            evals, thoughts = [], []

            for b in range(self.branches):
                h = torch.cat([branch_t[b], seq_x], dim=1)
                hout, _ = self.search_block(h, ext_mask)
                th = self.norm(hout[:, :self.K])
                ksc = self.search_block.attn.get_kappa(hout[:, :self.K]).mean([1, 2])
                evals.append(ksc)
                thoughts.append(th)

            w = F.softmax(torch.stack(evals, 0), 0)  # (branches,)
            best = (torch.stack(thoughts, 0) * w.unsqueeze(-1).unsqueeze(-1)).sum(0)

            if torch.norm(best - prev, dim=-1).mean() < self.eps and it > 0:
                break

        if self.training:
            refined = self.gumbel_mcts.soft_search(best)
        else:
            refined, _ = self.gumbel_mcts.search(best)

        return torch.cat([refined, seq_x], dim=1), ext_mask
