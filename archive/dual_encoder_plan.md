# Dual-Encoder Cohomology Pathway — Implementation Plan

## Goal

Add a learned cohomology pathway to the JEPA encoder. The GNN processes parse tree syntax; a small MLP processes the full 87-cell cohomology profile. Both outputs are concatenated before L2-normalization to produce a 512-dim embedding enriched with algebraic content.

This addresses the probing-identified gap: the GNN's pooled representation captures per-equation cohomology at 98% (count_p2 specifically; other cohom features in the 85-99% range), but the projection + L2-norm compresses it (count_p2 drops to 88%, others lose 1-4pp). The dedicated pathway preserves cohomological information through a channel that bypasses the projection bottleneck.

## Methodology resolutions (locked in 2026-05-14)

After scrutiny, the following refinements were locked before implementation. Architecture stays as designed; VICReg scope narrows, input gets richer, success criteria get sharper.

1. **VICReg scope:** Apply VICReg to the **384-dim z_syntax channel only**, not the full 512-dim raw embedding. The algebra channel is deterministic per equation, so VICReg's decorrelation pressure would fight the natural cohom covariance structure rather than preventing collapse.
2. **Cohom input: 174-dim (mask + values), not 87-dim with None→0.** Use an 87-dim binary mask ("does this equation satisfy the linear magma at this triple?") concatenated with 87-dim values (dim H² where satisfies=1, 0 otherwise). Removes the input-level ambiguity that affected ~33% of satisfying equations at p=2.
3. **Cohom value normalization:** Divide each cell's value by `p²` for its respective prime so all values land in `[0, 1]`. The mask half is already 0/1. Avoids p=7 cells (max value 48) swamping p=2 cells (max value 3) in the MLP's gradient.
4. **Primary success metric: cohom probing accuracy on the L2-norm output ≥ 95%** (vs 88% on `aux_h128_seed1_diag` baseline). bal_acc improvement is secondary and requires **mean over 2+ seeds** before claiming a real lift.
5. **Instrument cohom encoder gradient norm per-step and run an `accuracy_algebra_only` ablation per-epoch** (zero the syntax channel at eval time, measure bal_acc). Catches cohom_encoder collapse early.
6. **Track bal_acc separately for pairs where both equations have non-null cohom vs pairs involving null-profile equations.** Improvements should concentrate on the non-null half (the 32% all-null corpus equations effectively get a 384-dim embedding regardless of training).
7. **Pre-bake cohom profiles to `data/processed/cohom_profiles_174d.npy`** (shape `[N_eq, 174]`, mask concat values) for fast load + reproducibility. Build script: `scripts/build_cohom_profiles.py`. Source: `/home/gavin/git/equational_theories_data/invariants/11_cohomology/output/cohomology.parquet`.

## Architecture

```
E1 parse tree → GATv2 encoder → mean+max pool → pooled_rep (256-dim)
                                                       ↓
                                              output_proj (256 → 384)
                                                       ↓
                                                  z_syntax (384-dim)

E1 cohom profile (87-dim) → MLP (87 → 128 → 128) → z_algebra (128-dim)

concat(z_syntax, z_algebra) → F.normalize → embedding (512-dim)
```

The embedding dimension stays at 512. The GNN's projection narrows from 512 to 384 to make room for the 128-dim algebraic channel. The contrastive loss, predictor, VICReg, and aux heads all operate on the same 512-dim normalized embedding as before.

### The cohomology MLP

```python
class CohomologyEncoder(nn.Module):
    def __init__(self, input_dim: int = 87, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, cohom_profile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cohom_profile: [batch, 87] — raw cohomology profile.
                           None cells encoded as 0, non-null cells as dim H² value.
                           A separate binary mask could indicate null vs zero-valued,
                           but start simple with None→0.
        Returns:
            [batch, 128] — learned algebraic embedding
        """
        return self.net(cohom_profile)
```

Two layers, ReLU activation, 128-dim output. Total params: 87×128 + 128 + 128×128 + 128 = ~28K. Negligible relative to the GNN's 269K.

### Integration into the encoder

```python
class EquationEncoder(nn.Module):
    def __init__(self, ...):
        # Existing GATv2 layers (unchanged)
        self.gat_layers = ...
        self.layer_norms = ...
        
        # Modified output projection: 256 → 384 (was 256 → 512)
        self.output_proj = nn.Linear(pool_dim, 384)  # changed from 512
        
        # NEW: cohomology pathway
        self.cohom_encoder = CohomologyEncoder(input_dim=87, hidden_dim=128, output_dim=128)
    
    def forward(self, x, edge_index, edge_attr, batch, 
                cohom_profile=None, return_pooled=False):
        # GNN message passing (unchanged)
        h = ...
        
        # Pooling (unchanged)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # [batch, 256]
        
        # Syntactic projection (narrowed from 512 to 384)
        z_syntax = self.output_proj(pooled)  # [batch, 384]
        
        # Algebraic pathway
        if cohom_profile is not None:
            z_algebra = self.cohom_encoder(cohom_profile)  # [batch, 128]
            raw_embedding = torch.cat([z_syntax, z_algebra], dim=1)  # [batch, 512]
        else:
            # Fallback: zero-pad if no cohomology available
            # (for equations without cohomology data, e.g., new order-5 equations)
            z_algebra = torch.zeros(z_syntax.shape[0], 128, device=z_syntax.device)
            raw_embedding = torch.cat([z_syntax, z_algebra], dim=1)  # [batch, 512]
        
        # L2-normalize (unchanged)
        embedding = F.normalize(raw_embedding, dim=-1)
        
        if return_pooled:
            return embedding, pooled  # pooled unchanged for aux heads
        return embedding
```

### Key design decisions

**384/128 split (not 256/256 or 448/64):** The GNN carries the majority of the information (it already gets 96% accuracy on its own). 384 dims for syntax preserves most of the GNN's representational capacity. 128 dims for cohomology is enough to encode the 87-dim profile with room for learned combinations. The split can be tuned later, but 75/25 is a reasonable starting point.

**None → 0 encoding:** Equations with no satisfying linear magma at a given triple get 0 for that cell. This means "equation doesn't satisfy this linear magma" and "equation satisfies it but dim H² = 0" both map to 0. To distinguish them, add an 87-dim binary mask indicating which cells are non-null. This doubles the input to 174-dim. Start with the simple 87-dim version (None→0); add the mask if probing shows it matters.

**Zero-pad fallback:** If an equation has no cohomology data (not yet computed, or a new order-5 equation), the algebraic channel is all zeros. The L2-normalization redistributes this — the embedding is slightly different from a pure-syntax embedding because the algebraic channel pulls the norm. To handle this cleanly, the model should see some zero-padded examples during training so it learns to cope. Alternatively, if ALL training equations have cohomology, this only matters at inference time for new equations.

## Data Pipeline Changes

### Cohomology profile loading

```python
# Load cohomology data
cohom_df = pd.read_parquet("invariants/11_cohomology/cohomology.parquet")

# Build lookup: eq_id → 87-dim tensor
cohom_lookup = {}
for _, row in cohom_df.iterrows():
    profile = row["cohom_profile"]  # list of 87 int|None
    tensor = torch.tensor([v if v is not None else 0.0 for v in profile], dtype=torch.float32)
    cohom_lookup[row["eq_id"]] = tensor
```

### Collation changes

The DataLoader needs to provide cohomology profiles for both equations in each pair:

```python
def collate_pairs(batch):
    eq1_list, eq2_list, labels = zip(*batch)
    eq1_batch = Batch.from_data_list(eq1_list)
    eq2_batch = Batch.from_data_list(eq2_list)
    labels = torch.tensor(labels, dtype=torch.float)
    
    # Cohomology profiles
    eq1_cohom = torch.stack([cohom_lookup[eq.eq_id] for eq in eq1_list])  # [batch, 87]
    eq2_cohom = torch.stack([cohom_lookup[eq.eq_id] for eq in eq2_list])  # [batch, 87]
    
    # Existing aux labels (unchanged)
    eq1_aux = ...
    eq2_aux = ...
    
    return eq1_batch, eq2_batch, labels, eq1_aux, eq2_aux, eq1_cohom, eq2_cohom
```

### Training loop changes

```python
# Forward pass — context encoder gets cohomology
z_context, pooled_e1 = context_encoder(
    eq1_batch, cohom_profile=eq1_cohom, return_pooled=True
)

# Target encoder also gets cohomology (via EMA, no gradients)
with torch.no_grad():
    z_target = target_encoder(eq2_batch, cohom_profile=eq2_cohom)

# Predictor (unchanged — operates on 512-dim embedding)
z_predicted = predictor(z_context)

# Contrastive loss (unchanged)
contrastive = contrastive_loss(z_predicted, z_target, labels)

# VICReg on pre-norm output (now includes both syntax and algebra channels)
vicreg = vicreg_loss(raw_embedding_e1)

# Aux heads on pooled rep (unchanged — still 256-dim from GNN)
aux_preds_e1 = aux_heads(pooled_e1)

# E2 aux (run E2 through context encoder for aux gradients)
_, pooled_e2 = context_encoder(eq2_batch, cohom_profile=eq2_cohom, return_pooled=True)
aux_preds_e2 = aux_heads(pooled_e2)

# Total loss (unchanged structure)
total_loss = contrastive + vicreg_weight * vicreg + aux_weight * aux_loss
```

## Config

```yaml
# Dual-encoder settings
encoder:
  hidden_dim: 128
  embed_dim: 512           # total embedding unchanged
  syntax_dim: 384          # NEW: GNN projection output
  algebra_dim: 128         # NEW: cohomology MLP output
  num_layers: 4
  num_heads: 4

cohom_encoder:
  input_dim: 87            # raw 87-cell profile
  hidden_dim: 128
  output_dim: 128          # must equal encoder.algebra_dim

data:
  cohom_path: "invariants/11_cohomology/cohomology.parquet"

# Everything else unchanged
loss:
  margin: 1.5
  true_weight: 1.0
  vicreg_var_weight: 0.1
  vicreg_cov_weight: 0.1
  aux_weight: 0.1
  aux_heads: structural    # keep 10 structural heads only
```

## Testing Plan

### Test 1: Dual-encoder at h=128 (compare to aux_h128_seed1_diag baseline)

```yaml
encoder.hidden_dim: 128
encoder.syntax_dim: 384
cohom_encoder.output_dim: 128
seed: 1
# everything else same as aux_h128_seed1_diag
```

Compare against: aux_h128_seed1_diag (96.21% bal_acc, no cohomology pathway).

**Success criteria:**
- bal_acc ≥ 96.5% (improvement over syntax-only baseline)
- Recall balance maintained (TRUE/FALSE within 3pp)
- Stable training (no clipping pathology, no drift)

**Key diagnostic:** After training, reprobe the new embeddings with the same probing suite. Per-equation cohomology should probe at 95%+ (up from 88% at the L2-norm output). Pairwise obstruction should improve from 69%.

### Test 2: Ablation — how much does each channel contribute?

After Test 1 completes, run inference with the algebraic channel zeroed out (z_algebra = 0) and compute accuracy. Then run with the syntactic channel zeroed out (z_syntax = 0) and compute accuracy. The gap between full model and syntax-only tells you what the algebraic channel adds. The gap between full model and algebra-only tells you what the GNN adds.

This is a zero-cost ablation — no retraining, just modify the forward pass at eval time.

### Test 3: Dual-encoder at h=256 (if Test 1 succeeds)

Same recipe at h=256. The cohomology pathway doesn't interact with the h=256 instability mechanisms (VICReg, simplicity bias) because it's a separate input channel. But it might change the equilibrium by providing richer features that reduce the contrastive loss's pressure to overfit TRUE pairs.

## What to Monitor

Same diagnostic suite as previous runs, plus:

```python
# NEW: per-step cohomology MLP gradient norm
wandb.log({
    "train/grad_norm_cohom_encoder": cohom_encoder_grad_norm,
})

# NEW: per-epoch cohomology channel contribution
# Zero the algebra channel and measure accuracy drop
wandb.log({
    "val/accuracy_syntax_only": accuracy_with_algebra_zeroed,
    "val/accuracy_algebra_only": accuracy_with_syntax_zeroed,
})
```

The per-epoch channel ablation (zero one channel, measure accuracy) tracks how the model's reliance on each channel evolves during training. If the algebra channel's contribution grows over epochs, the model is increasingly using the cohomological information.

## Notes for Claude Code

- The output_proj dimension changes from 512 to 384. This means the checkpoint from aux_h128_seed1_diag CANNOT be loaded directly — the projection layer shape doesn't match. Train from scratch.
- The target encoder must also have the cohomology pathway (it's an EMA copy of the context encoder). Pass eq2_cohom to the target encoder in the no-grad block.
- VICReg applies to the full 512-dim raw_embedding (pre-L2-norm), which now includes both syntax and algebra channels. This is correct — VICReg should prevent collapse across ALL dimensions, not just the syntax ones.
- The aux heads still attach to the 256-dim pooled rep (from the GNN only, NOT including the cohomology). This is intentional — the aux heads anchor GNN features, the cohomology pathway is separate.
- The 87-dim cohomology input should be normalized (e.g., divide by p² for each prime's cells so all values are in [0, 1]) to keep the MLP's input scale consistent. Or use batch normalization on the input. Either works.
- For the 1,507 equations with all-null cohomology profiles, the input is all zeros. The MLP will learn to map all-zeros to a specific z_algebra vector. This is fine — it means "no linear magma satisfies this equation" which is itself algebraically meaningful.
- Commit the architecture change as a separate commit before running experiments.
- Run Test 1 first. Only proceed to Test 2 and Test 3 if Test 1 shows improvement.
