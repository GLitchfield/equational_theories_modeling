# Auxiliary Prediction Heads Implementation Plan

## Goal

Add auxiliary prediction heads to the encoder that predict known structural invariants of each equation. This anchors the encoder's representation against simplicity bias — the encoder can't collapse to "predict TRUE for everything" because it must simultaneously predict spine type, balance, and other structural features correctly.

## Motivation

The h=256 encoder finds an excellent representation at epoch 0 (94.7% bal_acc, balanced recall) then degrades during training as simplicity bias drives it toward TRUE-biased clustering. The literature (Shah et al. 2020, Xue et al. ICML 2023) shows that auxiliary tasks prevent this drift by anchoring specific features in the representation.

The auxiliary heads serve as a FLOOR, not a CEILING — they guarantee certain features survive training but don't limit what else the contrastive loss can learn. The remaining embedding dimensions are free to capture structure the auxiliary heads don't predict.

## Architecture

Auxiliary heads attach to the encoder's output (post-pooling, pre-L2-norm) and predict per-equation properties:

```
E1 parse tree → GATv2 Encoder → pooled representation (256*2=512 dim)
                                        ↓
                              ┌─────────┼──────────┐
                              ↓         ↓          ↓
                         output_proj  aux_spine  aux_balance  ...
                              ↓         ↓          ↓
                         embed (512) class(5)   class(2)
                              ↓
                         F.normalize
                              ↓
                         z_context (for contrastive)
```

Each auxiliary head is a single linear layer from the pooled representation to the prediction target. No hidden layers, no activation — the heads should be SIMPLE so they force the encoder to produce linearly separable features, not learn complex nonlinear mappings.

```python
class AuxiliaryHeads(nn.Module):
    def __init__(self, pool_dim: int):
        super().__init__()
        # pool_dim = hidden_dim * 2 (from mean+max pooling)
        # For h=256: pool_dim = 512
        # For h=128: pool_dim = 256
        
        self.spine_head = nn.Linear(pool_dim, 5)      # 5-class: left, right, mixed, trivial, family_c
        self.balance_head = nn.Linear(pool_dim, 2)     # binary: balanced or not
        self.singleton_head = nn.Linear(pool_dim, 2)   # binary: singleton-forcing or not
        self.num_vars_head = nn.Linear(pool_dim, 4)    # 4-class: 1, 2, 3, 4 variables
    
    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pooled: [batch, pool_dim] — encoder output BEFORE output_proj and L2-norm
        Returns:
            dict of logits for each auxiliary task
        """
        return {
            "spine": self.spine_head(pooled),       # [batch, 5]
            "balance": self.balance_head(pooled),    # [batch, 2]
            "singleton": self.singleton_head(pooled),# [batch, 2]
            "num_vars": self.num_vars_head(pooled),  # [batch, 4]
        }
```

## Which Invariants to Predict

Start with four that are most relevant to preventing the FALSE-detecting feature suppression:

**Spine type (5-class)**: The single most predictive structural feature for implication (McKenna's theorem). Spine incompatibility determines FALSE for ~1.5M pairs. If the encoder suppresses spine information, it loses a huge chunk of FALSE-detection capability. This is the most important head.

**Balance (binary)**: Balance asymmetry (E1 balanced, E2 unbalanced) predicts FALSE. The XOR magma separates balanced from unbalanced equations. If the encoder loses balance information, it can't detect this separation.

**Singleton forcing (binary)**: Equations that force singleton magmas imply everything. This is a TRUE-detection feature. Including it ensures the encoder maintains both TRUE-relevant and FALSE-relevant features, not just one side.

**Variable count (4-class)**: The number of distinct variables (1-4) is a basic structural property that correlates with equation complexity and behavior. Cheap to predict, easy sanity check.

These four heads add negligible parameters (~2K total for h=256). They're training signals, not model capacity.

## Auxiliary Labels

The invariant labels come from the invariants project. They need to be available as a per-equation lookup table:

```python
# Load auxiliary labels
aux_labels = pd.read_parquet("data/raw/equation_invariants.parquet")

# For each equation in a batch, look up its labels
# eq_id → {"spine": 0-4, "balance": 0-1, "singleton": 0-1, "num_vars": 0-3}
```

**If the invariants project hasn't computed these yet**: spine type, balance, singleton forcing, and variable count are all computable from the parse tree alone (no Vampire data needed, no cohomology). They're the first 7 invariants in the implementation order from the invariants doc. Claude Code can compute them inline if the parquet file isn't available — they're simple tree traversals.

**Critical**: the auxiliary labels are per-EQUATION, not per-pair. Each equation in a batch gets its own auxiliary prediction. For a pair (E1, E2), you compute auxiliary loss for E1 AND E2 separately.

## Loss Function

```python
# Total loss per step:
total_loss = contrastive_loss + aux_weight * auxiliary_loss

# Auxiliary loss is the average cross-entropy across all heads and both equations:
aux_loss_e1 = (
    F.cross_entropy(aux_preds_e1["spine"], spine_labels_e1) +
    F.cross_entropy(aux_preds_e1["balance"], balance_labels_e1) +
    F.cross_entropy(aux_preds_e1["singleton"], singleton_labels_e1) +
    F.cross_entropy(aux_preds_e1["num_vars"], num_vars_labels_e1)
) / 4

aux_loss_e2 = (
    F.cross_entropy(aux_preds_e2["spine"], spine_labels_e2) +
    F.cross_entropy(aux_preds_e2["balance"], balance_labels_e2) +
    F.cross_entropy(aux_preds_e2["singleton"], singleton_labels_e2) +
    F.cross_entropy(aux_preds_e2["num_vars"], num_vars_labels_e2)
) / 4

auxiliary_loss = (aux_loss_e1 + aux_loss_e2) / 2
```

### Auxiliary weight

Start with `aux_weight = 0.1`. The auxiliary loss should be secondary to the contrastive loss — enough to anchor features but not enough to dominate the optimization.

Cross-entropy for a 5-class problem at random initialization is ~ln(5) ≈ 1.6. For binary problems, ~ln(2) ≈ 0.7. The average auxiliary loss at init will be roughly 1.0. With aux_weight=0.1, that's 0.1 added to the contrastive loss of ~0.7 — about 14% of total loss. This should be enough to anchor features without overwhelming the primary task.

If the auxiliary loss drops quickly (the encoder already captures these features) and the contrastive loss improves, the weighting is right. If the auxiliary loss stays high while contrastive improves, the encoder is ignoring the auxiliary signal (weight too low). If contrastive stalls while auxiliary drops, the encoder is over-prioritizing auxiliary tasks (weight too high).

## Integration into Training

### Data pipeline changes

The DataLoader needs to provide auxiliary labels for both equations in each pair. Add to the collation:

```python
def collate_pairs(batch):
    eq1_list, eq2_list, labels = zip(*batch)
    eq1_batch = Batch.from_data_list(eq1_list)
    eq2_batch = Batch.from_data_list(eq2_list)
    labels = torch.tensor(labels, dtype=torch.float)
    
    # Auxiliary labels — per equation
    eq1_aux = {
        "spine": torch.tensor([eq.spine_label for eq in eq1_list]),
        "balance": torch.tensor([eq.balance_label for eq in eq1_list]),
        "singleton": torch.tensor([eq.singleton_label for eq in eq1_list]),
        "num_vars": torch.tensor([eq.num_vars_label for eq in eq1_list]),
    }
    eq2_aux = {
        "spine": torch.tensor([eq.spine_label for eq in eq2_list]),
        "balance": torch.tensor([eq.balance_label for eq in eq2_list]),
        "singleton": torch.tensor([eq.singleton_label for eq in eq2_list]),
        "num_vars": torch.tensor([eq.num_vars_label for eq in eq2_list]),
    }
    
    return eq1_batch, eq2_batch, labels, eq1_aux, eq2_aux
```

### Encoder changes

The encoder needs to expose the pooled representation (before output_proj) for the auxiliary heads:

```python
class EquationEncoder(nn.Module):
    def forward(self, x, edge_index, edge_attr, batch, return_pooled=False):
        # ... message passing ...
        h = ...  # final node representations
        
        # Pooling
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # [batch, pool_dim]
        
        # Projection + normalization
        raw_embedding = self.output_proj(pooled)
        embedding = F.normalize(raw_embedding, dim=-1)
        
        if return_pooled:
            return embedding, pooled
        return embedding
```

### Training loop changes

```python
# Forward pass with auxiliary predictions
z_context, pooled_e1 = context_encoder(eq1_batch, return_pooled=True)
with torch.no_grad():
    z_target = target_encoder(eq2_batch)

# Also need pooled representation for E2 (from context encoder, not target)
_, pooled_e2 = context_encoder(eq2_batch, return_pooled=True)

z_predicted = predictor(z_context)

# Contrastive loss (unchanged)
contrastive = contrastive_loss(z_predicted, z_target, labels)

# VICReg (unchanged)
vicreg_loss = vicreg(raw_embedding_e1)  # or however it's currently wired

# Auxiliary loss (NEW)
aux_preds_e1 = aux_heads(pooled_e1)
aux_preds_e2 = aux_heads(pooled_e2)
aux_loss = compute_auxiliary_loss(aux_preds_e1, eq1_aux, aux_preds_e2, eq2_aux)

# Total loss
total_loss = contrastive + vicreg_weight * vicreg_loss + aux_weight * aux_loss
```

**Note on E2 auxiliary predictions**: We need the CONTEXT encoder's representation of E2 for the auxiliary heads (not the target encoder's). The target encoder is EMA-updated and detached — we can't compute gradients through it. Running E2 through the context encoder a second time costs compute but produces gradients that flow back to the encoder. Alternatively, if E2 graphs are already being processed by the context encoder for another purpose, reuse that computation.

## Config

```yaml
# Auxiliary heads
aux_weight: 0.1
aux_heads:
  spine: 5        # 5-class classification
  balance: 2      # binary
  singleton: 2    # binary
  num_vars: 4     # 4-class

# Auxiliary label file
aux_labels_path: "data/raw/equation_invariants.parquet"
# OR compute inline from parse trees if parquet not available
```

## Testing Plan

### Test 1: Auxiliary heads at h=128 (baseline comparison)

```yaml
encoder.hidden_dim: 128
aux_weight: 0.1
# everything else same as current best (VICReg 0.1, margin 1.5, tw 1.0)
```

Compare against h=128 VICReg baseline (95.2%).

This tests whether auxiliary heads help even at h=128 where training is already stable. If accuracy improves, the encoder WAS suppressing some useful features. If accuracy is the same, h=128 was already capturing these features and the auxiliary heads are redundant at this width.

### Test 2: Auxiliary heads at h=256 (the real test)

```yaml
encoder.hidden_dim: 256
aux_weight: 0.1
vicreg_var_weight: 0.05
vicreg_cov_weight: 0.05
# margin 1.5, tw 1.0
```

Compare against h=256 VICReg 0.05 (94.7% at epoch 0, degraded to 92.8%).

**The key question**: does bal_acc stay above 94% through the full 10 epochs instead of degrading after epoch 0? If the auxiliary heads prevent the TRUE-bias drift, the wider encoder maintains its superior epoch-0 geometry through the full run.

**Watch for**: per-epoch TRUE recall and FALSE recall. In previous h=256 runs, TRUE recall climbed to 0.99 while FALSE recall dropped to 0.87. With auxiliary heads anchoring spine type and balance features, FALSE recall should hold steady because the encoder can't suppress the features that detect FALSE pairs.

### Test 3: Auxiliary weight sweep (if Test 2 helps)

Try aux_weight = 0.01, 0.05, 0.1, 0.25. The right weight is large enough to anchor features but small enough not to dominate the contrastive objective. Plot auxiliary loss and contrastive loss trajectories to find the sweet spot.

## Logging

Add per-step:
```python
wandb.log({
    "train/aux_loss": aux_loss.item(),
    "train/aux_spine_loss": spine_loss.item(),
    "train/aux_balance_loss": balance_loss.item(),
    "train/aux_singleton_loss": singleton_loss.item(),
    "train/aux_num_vars_loss": num_vars_loss.item(),
})
```

Add per-epoch (on validation/monitor batch):
```python
# Auxiliary head accuracy on monitor batch
wandb.log({
    "val/aux_spine_acc": spine_accuracy,
    "val/aux_balance_acc": balance_accuracy,
    "val/aux_singleton_acc": singleton_accuracy,
    "val/aux_num_vars_acc": num_vars_accuracy,
})
```

Watching auxiliary accuracy per epoch tells you whether the encoder is maintaining these features throughout training. If spine accuracy drops from 95% to 80% during training, the simplicity bias is overcoming the auxiliary anchor (weight too low). If it stays at 95%+, the anchor holds.

## Invariant Computation (if not available from invariants project)

If the invariants parquet doesn't exist yet, compute the four needed invariants inline. They all depend only on the parse tree:

**Spine type**: Walk from root of RHS to leftmost occurrence of LHS variable. Record L/R at each step. All-L = left, all-R = right, mixed = mixed. No LHS var in RHS or trivial = trivial. Product LHS = family_c.

**Balance**: Count each variable on LHS and RHS. Balanced if all counts match.

**Singleton forcing**: LHS variable doesn't appear in RHS.

**Variable count**: Count distinct variables in the equation.

These are the simplest invariants in the invariants doc (items 1-5 in implementation order). Each is a 10-line function over the parse tree. Claude Code already has the parser and tree utilities from the JEPA project.

## Notes for Claude Code

- Auxiliary heads attach to the POOLED representation (after mean+max pool, BEFORE output_proj and L2-norm). This is the richest internal representation — it's what the output projection compresses from.
- Each auxiliary head is a SINGLE nn.Linear. No hidden layers, no activation. The simplicity is intentional — it forces the encoder to produce linearly separable features.
- Auxiliary labels are PER-EQUATION, not per-pair. Look them up by equation ID.
- The auxiliary loss on E2 requires running E2 through the CONTEXT encoder (not the target encoder) to get gradients. This is an extra forward pass. If compute is tight, you can skip the E2 auxiliary loss and only compute it for E1 — the encoder sees both E1 and E2 during training, so E1-only auxiliary should still anchor the features.
- Start with aux_weight=0.1. Monitor the ratio of auxiliary loss to contrastive loss — it should stay between 5% and 20% of total loss.
- Log auxiliary HEAD ACCURACY per epoch, not just loss. Accuracy tells you whether the features are maintained; loss can decrease even if accuracy plateaus.
- The auxiliary heads do NOT participate in inference. At test time, only the contrastive distance is used for prediction. The heads are training-time-only anchors.
- Keep VICReg active alongside auxiliary heads. They address different problems: VICReg prevents dimensional collapse, auxiliary heads prevent simplicity bias.
