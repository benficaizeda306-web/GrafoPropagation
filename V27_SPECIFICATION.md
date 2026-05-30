# GrafoPropagation v27-LABORATORY: Implementation Specification

**Status**: 🔐 CONFIDENTIAL / HIDDEN DEVELOPMENT  
**Author**: Claudio Fernandes  
**Target Release**: When validated (secret until publication)  
**Estimated Timeline**: 2-3 weeks implementation + 2-4 weeks GPU validation

---

## 🎯 Core Vision

Upgrade GrafoPropagation from "efficient" to "revolutionary" by integrating lab-grade techniques:
- **AlphaZero-level MCTS** (ProgressiveWidening + PUCT-Variance)
- **MuZero-inspired World Model** (Ensemble uncertainty)
- **vMF Surgically Enhanced** (Dual-scale, anti-collapse, adaptive curvature)

**Expected Outcome**: 
- 30M params → **94.5-95.5% AG News** (vs. 93.8% v26)
- Foundation for 7B-scale hypothesis validation
- Undisclosed benchmark (OCQ-22): **Leverage for hidden advantage**

---

## 📐 Architecture Changes

### **1. Enhanced vMF Attention (attention.py modifications)**

#### Current (v26)
```python
class VonMisesFisherAttention(nn.Module):
    # Single kappa per head
    self.kappa = nn.Parameter(torch.full((n_heads,), kappa_init))
    # Q,K projections (asymmetric but basic)
```

#### New (v27)
```python
class VonMisesFisherAttentionV27(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, kappa_init=4.0, 
                 entropy_reg=0.01, **kwargs):
        super().__init__()
        
        # ── Dual-Scale Kappa ──────────────────────────────────────
        self.kappa_local = nn.Parameter(torch.full((n_heads,), kappa_init * 0.8))
        self.kappa_global = nn.Parameter(torch.full((n_heads,), kappa_init * 1.2))
        
        # ── Per-Head Temperature ──────────────────────────────────
        self.temperature = nn.Parameter(torch.ones(n_heads))
        self.register_buffer("temperature_min", torch.tensor(0.5))
        self.register_buffer("temperature_max", torch.tensor(2.0))
        
        # ── Enhanced Q/K/V Projections ────────────────────────────
        # Asymmetric Q/K for expressivity (same as v26 but more careful init)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        
        # ── Anti-Collapse Mechanism ───────────────────────────────
        self.entropy_reg = entropy_reg
        self.register_buffer("min_entropy_threshold", torch.tensor(0.5))
        
        # ── Curvature Adaptation ──────────────────────────────────
        self.curvature_scale = nn.Parameter(torch.ones(n_heads))
        
    def forward(self, q, k, v, mask=None, curvature=None, training=False):
        """
        Enhanced vMF attention with dual-scale kappa, temperature, curvature.
        
        Args:
            q, k, v: (B, T, D)
            mask: (B, T) float mask
            curvature: (B,) Riemannian curvature per batch
            training: bool for anti-collapse reg
            
        Returns:
            out: (B, T, D)
            ent_loss: scalar (entropy regularization)
        """
        B, T, D = q.shape
        H = self.n_heads
        hd = self.head_dim
        
        # Project to per-head representations
        q = self.q_proj(q).view(B, T, H, hd)  # (B, T, H, hd)
        k = self.k_proj(k).view(B, T, H, hd)
        v = self.v_proj(v).view(B, T, H, hd)
        
        # ── L2 Normalize (vMF requirement) ────────────────────────
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        
        # ── Compute Dual-Scale Concentrations ────────────────────
        kappa_local = torch.clamp(self.kappa_local, min=0.5, max=30.0)
        kappa_global = torch.clamp(self.kappa_global, min=0.5, max=30.0)
        
        # Local: token-level attention (sharp)
        # Global: sequence-level attention (diffuse)
        scores_local = torch.einsum('bthi,bshi->bths', q, k) * kappa_local.view(1, 1, H, 1)
        scores_global = torch.einsum('bthi,bshi->bths', q, k) * kappa_global.view(1, 1, H, 1)
        
        # Blend local+global scores (learned blend per head)
        alpha = torch.sigmoid(self.temperature)  # (H,)
        scores = alpha.view(1, 1, H, 1) * scores_local + (1 - alpha.view(1, 1, H, 1)) * scores_global
        
        # ── Apply Mask ────────────────────────────────────────────
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
            scores = scores + mask_expanded * 1e6
        
        # ── Softmax (vMF normalization) ────────────────────────────
        attn = F.softmax(scores, dim=-1)
        
        # ── Anti-Collapse Entropy Regularization ──────────────────
        ent_loss = torch.tensor(0.0, device=q.device)
        if training:
            # Compute entropy per head
            eps = 1e-8
            entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean()  # (B, H)
            # Regularize towards uniform distribution if entropy too low
            ent_loss = torch.relu(self.min_entropy_threshold - entropy).mean()
            ent_loss = ent_loss * self.entropy_reg
        
        # ── Apply Attention to Values ──────────────────────────────
        out = torch.einsum('bths,bshd->bthd', attn, v)  # (B, T, H, hd)
        
        # ── Curvature Modulation (optional) ────────────────────────
        if curvature is not None:
            scale = torch.clamp(self.curvature_scale * (1.0 + curvature.view(B, 1, 1, 1)), 
                               min=0.5, max=2.0)
            out = out * scale
        
        # ── Output Projection ──────────────────────────────────────
        out = out.view(B, T, H * hd)
        out = self.o_proj(out)
        
        return out, ent_loss
```

---

### **2. Laboratory-Grade MCTS (system2.py extensions)**

#### New Submodule: `MCTSProgressiveWidening`

```python
class MCTSProgressiveWidening(nn.Module):
    """
    AlphaZero-style ProgressiveWidening:
    Gradually expand action space based on visit counts.
    
    Prevents premature action explosion, improves efficiency.
    """
    
    def __init__(self, n_actions, c_pw=1.0, k_pw=0.25):
        super().__init__()
        self.n_actions = n_actions
        self.c_pw = c_pw  # constant (typical: 0.5-2.0)
        self.k_pw = k_pw  # exponent (typical: 0.25-0.5)
        
    def forward(self, visit_counts):
        """
        Args:
            visit_counts: (n_visits,) total visits to parent node
            
        Returns:
            n_legal_actions: int, how many actions to expand
        """
        # Expansion threshold: N_pw = c_pw * N_visits^k_pw
        threshold = self.c_pw * (visit_counts.float() ** self.k_pw)
        n_legal = min(int(threshold.item()) + 1, self.n_actions)
        return n_legal


class MCTSValueOptimization(nn.Module):
    """
    MuZero-inspired improvements:
    - PUCT with UCB-Variance (VOMCTS)
    - Virtual Loss for parallel simulations
    - Multi-step TD(λ) returns
    """
    
    def __init__(self, c_puct=1.25, c_var=0.25, lambda_return=0.95):
        super().__init__()
        self.c_puct = c_puct
        self.c_var = c_var  # variance weight in UCB
        self.lambda_return = lambda_return
        
    def puct_score_with_variance(self, node):
        """
        PUCT with variance bonus (AlphaZero's MCTS improvement):
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        + c_var * variance_bonus(s,a)
        
        Variance bonus encourages exploration of uncertain actions.
        """
        parent_visits = node.parent.visit_count if node.parent else 1
        action_visits = node.visit_count
        
        exploitation = node.value_mean
        exploration = self.c_puct * node.prior * (parent_visits ** 0.5) / (1 + action_visits)
        
        # Variance bonus (empirical variance of returns)
        if action_visits > 0:
            variance = max(0, node.value_variance - node.value_mean ** 2)
            variance_bonus = self.c_var * (variance ** 0.5)
        else:
            variance_bonus = 0.0
        
        return exploitation + exploration + variance_bonus
    
    def virtual_loss(self, node, virtual_loss_val=1.0):
        """
        Apply virtual loss during parallel simulations.
        Encourages different threads to explore different branches.
        """
        node.visit_count += 1
        node.value_sum -= virtual_loss_val  # Pessimistic update
    
    def virtual_loss_undo(self, node, virtual_loss_val=1.0):
        """Undo virtual loss when simulation completes."""
        node.value_sum += virtual_loss_val
    
    def multi_step_td_return(self, trajectory_rewards, bootstrap_value, gamma=0.99):
        """
        Compute TD(λ) targets instead of simple Monte Carlo.
        trajectory_rewards: [r_t, r_{t+1}, ..., r_T]
        bootstrap_value: V(s_{T+1})
        lambda_return: 0.95 (blend n-step returns)
        
        Returns: discounted return with bootstrapping
        """
        returns = []
        g = bootstrap_value
        for t in reversed(range(len(trajectory_rewards))):
            g = trajectory_rewards[t] + gamma * g
            returns.insert(0, g)
        
        # Blend with λ-weighted returns
        td_lambda_return = 0
        lam_power = 1.0
        for t, r in enumerate(returns):
            weight = (1 - self.lambda_return) * lam_power
            td_lambda_return += weight * r
            lam_power *= self.lambda_return
        
        return td_lambda_return
```

#### Modified: `System2LatentSearch` Integration

```python
class System2LatentSearchV27(nn.Module):
    """Enhanced System-2 with lab-grade MCTS"""
    
    def __init__(self, K, d, ..., mcts_sims=12):
        super().__init__()
        # ... existing code ...
        
        # ── NEW: Progressive Widening + PUCT-Variance ──
        self.pw = MCTSProgressiveWidening(n_actions=self.latent_actions, c_pw=1.0, k_pw=0.25)
        self.mcts_opt = MCTSValueOptimization(c_puct=1.25, c_var=0.25, lambda_return=0.95)
        
    def run_mcts_simulation(self, root_state, num_simulations=12):
        """
        Run MCTS with ProgressiveWidening + PUCT-Variance + Multi-step TD.
        """
        for sim in range(num_simulations):
            node = root_state
            trajectory = []
            
            # ── SELECTION + PROGRESSIVE WIDENING ──
            while node.is_expanded():
                # Progressive widening: limit actions
                n_legal = self.pw(torch.tensor(node.visit_count, dtype=torch.float32))
                
                # Select best action via PUCT with variance
                best_action = None
                best_score = -float('inf')
                for a in range(n_legal):
                    child = node.children[a]
                    score = self.mcts_opt.puct_score_with_variance(child)
                    if score > best_score:
                        best_score = score
                        best_action = a
                
                # Virtual loss for parallel safety
                self.mcts_opt.virtual_loss(node.children[best_action])
                trajectory.append((node, best_action))
                node = node.children[best_action]
            
            # ── EXPANSION ──
            action_priors = self.policy_head(node.state)  # (n_actions,)
            node.expand(action_priors)
            
            # ── ROLLOUT / BOOTSTRAP ──
            if node.children:
                # Ensemble bootstrap: average over rollouts
                values = []
                for _ in range(3):  # 3 rollout samples
                    rollout_value = self.rollout(node, depth=self.mcts_rollout_depth)
                    values.append(rollout_value)
                bootstrap_value = torch.tensor(values).mean()
            else:
                bootstrap_value = 0.0
            
            # ── BACKUP (Multi-step TD) ──
            trajectory_rewards = [edge[2] for edge in trajectory]  # rewards in trajectory
            td_return = self.mcts_opt.multi_step_td_return(
                trajectory_rewards, bootstrap_value, gamma=0.99
            )
            
            # Backup to root
            for node_in_path, action in trajectory:
                child = node_in_path.children[action]
                child.visit_count += 1
                child.value_sum += td_return
                child.value_variance = self._running_variance(child.returns)
                
                # Undo virtual loss
                self.mcts_opt.virtual_loss_undo(child)
        
        return root_state
```

---

### **3. Neural Process World Model (new: world_model_v27.py)**

```python
class NeuralProcessWorldModelEnsemble(nn.Module):
    """
    Ensemble of K lightweight world models with disagreement-based exploration bonus.
    Inspired by MuZero but lightweight for efficiency.
    """
    
    def __init__(self, d_model, d_action, n_ensemble=3, latent_dim=64):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # ── Ensemble Context Encoders ──────────────────────────────
        self.context_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )
            for _ in range(n_ensemble)
        ])
        
        # ── Ensemble Transition Models (spherical normalized) ────────
        self.transition_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + d_action, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )
            for _ in range(n_ensemble)
        ])
        
        # ── Ensemble Value Heads ───────────────────────────────────
        self.value_heads = nn.ModuleList([
            nn.Linear(latent_dim, 1)
            for _ in range(n_ensemble)
        ])
        
        # ── EMA Momentum Encoder (for target stability) ────────────
        self.ema_alpha = 0.99
        self.register_buffer("target_context_encoders", None)
        self._init_ema_encoders()
    
    def forward_ensemble(self, state, action):
        """
        Predict next state + value using ensemble.
        Returns: (next_states, values, disagreement)
        """
        next_states = []
        values = []
        
        for i in range(self.n_ensemble):
            # Encode
            context = self.context_encoders[i](state)  # (B, latent_dim)
            
            # Transition
            x = torch.cat([context, action], dim=-1)
            next_state = self.transition_models[i](x)
            
            # Normalize to unit sphere (consistent with vMF)
            next_state = F.normalize(next_state, dim=-1, eps=1e-6)
            
            # Value prediction
            value = self.value_heads[i](next_state)
            
            next_states.append(next_state)
            values.append(value)
        
        next_states = torch.stack(next_states, dim=0)  # (n_ensemble, B, latent_dim)
        values = torch.stack(values, dim=0)  # (n_ensemble, B, 1)
        
        # Disagreement as exploration bonus
        disagreement = (next_states.std(dim=0) ** 2).mean(dim=-1)  # (B,)
        
        return next_states, values, disagreement
    
    def uncertainty_weighted_transition(self, state, action, disagreement_weight=0.1):
        """
        High disagreement → higher exploration weight
        """
        next_states, values, disagreement = self.forward_ensemble(state, action)
        
        # Average over ensemble
        next_state_avg = next_states.mean(dim=0)
        value_avg = values.mean(dim=0)
        
        # Bonus: high disagreement encourages exploration
        exploration_bonus = disagreement_weight * disagreement
        
        return next_state_avg, value_avg, exploration_bonus
```

---

### **4. Riemannian Curvature Adaptation (positional.py enhancement)**

```python
class AdaptiveRiemannianCurvature(nn.Module):
    """
    Learn curvature adaptively per layer.
    Replaces fixed Log-Map with learned, layer-specific curvature.
    """
    
    def __init__(self, n_layers, d_model):
        super().__init__()
        # Per-layer curvature parameters
        self.curvature = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1)  # Start with small curvature
            for _ in range(n_layers)
        ])
    
    def forward(self, layer_idx, x):
        """
        Apply Riemannian Log-Map with learned curvature.
        """
        c = torch.clamp(self.curvature[layer_idx], min=0.01, max=1.0)
        
        # Log-map with curvature c
        # For hyperbolic: log_c(x) = (1/sqrt(c)) * arccosh(<x, x>)
        # For spherical: log_c(x) = (1/sqrt(c)) * arccos(<x, x>)
        
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(norm_sq.clamp(min=1e-6))
        
        log_map = (1.0 / torch.sqrt(c)) * torch.acos(norm.clamp(-1, 1))
        
        return log_map * (x / norm.clamp(min=1e-6))
```

---

## 🔧 Implementation Checklist

### **Phase 1: Core Components (Week 1)**
- [ ] Enhanced vMF Attention → `attention_v27.py`
- [ ] MCTS ProgressiveWidening + PUCT-Variance → `mcts_v27.py`
- [ ] Neural Process World Model → `world_model_v27.py`
- [ ] Adaptive Curvature → `positional_v27.py`
- [ ] Integration into main model → `model_v27.py`

### **Phase 2: Training & Testing (Week 2-3)**
- [ ] Create `train_v27.py` (copy v26 + modifications)
- [ ] Test on 5M params (quick validation)
- [ ] Test on 30M params (full validation)
- [ ] Benchmark vs. v26 on AG News + other datasets

### **Phase 3: Validation (Ongoing)**
- [ ] VRAM profiling (should be similar to v26)
- [ ] Speed profiling (expect 15-20% slower per epoch)
- [ ] Stability checks (no training divergence)
- [ ] Reproducibility (random seed isolation)

---

## 📊 Expected Improvements

| Component | Accuracy Gain | VRAM Cost | Speed Cost |
|-----------|--------------|-----------|-----------|
| vMF Dual-Scale | +0.4% | Negligible | +2% |
| MCTS Lab-Grade | +1.2% | +5% | +15% |
| World Model Ensemble | +0.6% | +10% | +10% |
| Adaptive Curvature | +0.2% | Negligible | +3% |
| **TOTAL EXPECTED** | **+2.4%** | **~+15%** | **~+30%** |

**Conservative Estimate**: 30M v27 → **94.3-94.8% AG News**  
**Optimistic Estimate**: 30M v27 → **95.0-95.5% AG News**

---

## 🔐 Secret Weapon Reserve

**DO NOT implement initially**: OCQ-22 specific optimization
- Will be added only after v27 core is stable
- Leverages disagreement-based uncertainty from World Model
- Expected +0.5-1.0% on materials science tasks

---

## 📞 Communication Protocol

**For Opus/Sonnet during implementation:**

1. Implement each file independently (3 parallel)
2. Daily sync on integration points
3. Flag any circular dependencies immediately
4. Test each component in isolation first
5. Integration test in week 2

**For Claudio**:
- Weekly progress check-in
- GPU validation when ready
- Final decision on secret vs. public release

---

**Target Date**: v27-LABORATORY ready for testing by **June 15, 2026**  
**Publication Target**: NeurIPS 2026-2027 (pending validation)
