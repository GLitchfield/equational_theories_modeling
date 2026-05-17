# Skip Connection — Preserve Pooled Representation Features

## Problem

Probing showed the GNN's pooled representation captures cohomological features at higher accuracy than the projection + L2-norm output embedding does — most dramatically `count_p2` (0.98 pooled → 0.88 output, −10pp), but also `count_p3` (−4pp), `count_p5` (−4pp), and the rest of the cohom summaries (1-4pp). The projection layer, trained by contrastive loss alone, partly suppresses features it doesn't need for the majority of pairs. The cumulative effect across many features is the value the skip connection is recovering, not a single dramatic gain on any one feature.

## Methodology resolutions (locked in 2026-05-14)

After scrutiny, the following were locked in before implementation. Architecture stays as designed; LayerNorm is added to keep both halves on equal footing, success criteria sharpen, instrumentation adds per-channel rank.

1. **LayerNorm both halves before concat** (correctness, not optimization). Without normalization the skip channel dominates at init (LayerNorm'd hidden states have ~unit per-dim scale; the random `output_proj` outputs much smaller per-dim values). The contrastive loss would then have to "earn" the projection's way into the embedding over multiple epochs while the projection's information is largely discarded. Implementation:
    ```python
    self.proj_norm = nn.LayerNorm(proj_dim)
    self.skip_norm = nn.LayerNorm(pool_dim)
    # In forward:
    z_proj = self.proj_norm(self.output_proj(pooled))
    z_skip = self.skip_norm(pooled)
    raw_embedding = torch.cat([z_proj, z_skip], dim=-1)
    ```
    ~1.5K extra params total. Both halves start at unit-ish per-dim scale → L2-norm distributes equally → balanced contribution from step 0.
2. **Realistic gain expectation: 2-5pp on most cohom probes, not 10pp.** The 10pp headline was `count_p2` only; most cohom features dropped 1-4pp. The skip's value is cumulative across many features.
3. **Margin: keep at 1.5, monitor not pre-adjust.** Watch `dist_true_p50` / `dist_false_p50` in epochs 0-1; recalibrate only if the distributions look wrong.
4. **Aux + skip interaction is a feature, not a bug.** Since `pooled` is now literally half the embedding, aux heads anchoring features in `pooled` directly protect those features in the final embedding. Structural features get double protection (aux prevents suppression in the GNN backbone, skip prevents loss in the projection).
5. **Per-channel effective rank** alongside the full embedding's: log `val/eff_rank_proj`, `val/eff_rank_skip`, `val/eff_rank_full`. If one channel collapses while the other is healthy, that's important diagnostic information.
6. **Channel-ablation eval is informative for relative trends, not a perfectly clean comparison** (zeroing one half changes the L2 norm of the kept half, putting it on a different unit sphere). Useful to track per-epoch but interpret directionally.

## Fix

Add a skip connection that concatenates the raw pooled representation with the projected output before L2-normalization. The pooled rep's features survive to the embedding without passing through the projection bottleneck.

```
pooled (256-dim) ──┬── output_proj (256→512) ── z_proj (512-dim) ──┐
                   │                                                ├── concat ── raw (768) ── F.normalize ── embedding (768)
                   └──────────────── z_skip (256-dim) ─────────────┘
```

Total embedding widens from 512 to 768. The projection keeps its full 512-dim capacity (no regression from the 256→512 embed_dim gain). The skip adds 256 dims of preserved features on top.

## Architecture Change

The change is in `EquationEncoder.forward` only. No new modules, no new data, no new loss terms.

### Before

```python
def forward(self, x, edge_index, edge_attr, batch, return_pooled=False):
    h = ...  # GNN message passing
    
    mean_pool = global_mean_pool(h, batch)
    max_pool = global_max_pool(h, batch)
    pooled = torch.cat([mean_pool, max_pool], dim=1)  # [batch, 256]
    
    raw_embedding = self.output_proj(pooled)           # [batch, 512]
    embedding = F.normalize(raw_embedding, dim=-1)     # [batch, 512]
    
    if return_pooled:
        return embedding, pooled
    return embedding
```

### After

```python
def forward(self, x, edge_index, edge_attr, batch, return_pooled=False):
    h = ...  # GNN message passing (unchanged)
    
    mean_pool = global_mean_pool(h, batch)
    max_pool = global_max_pool(h, batch)
    pooled = torch.cat([mean_pool, max_pool], dim=1)  # [batch, 256]
    
    z_proj = self.output_proj(pooled)                  # [batch, 512]
    raw_embedding = torch.cat([z_proj, pooled], dim=1) # [batch, 768]  <-- NEW
    embedding = F.normalize(raw_embedding, dim=-1)     # [batch, 768]
    
    if return_pooled:
        return embedding, pooled
    return embedding
```

That's it. One line added, one line changed.

## Downstream Dimension Changes

The embedding widening from 512 to 768 propagates to:

### Predictor

Input and output dimensions change:

```python
# Before
self.predictor = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 512),
)

# After
self.predictor = nn.Sequential(
    nn.Linear(768, 128),   # input widens
    nn.ReLU(),
    nn.Linear(128, 768),   # output widens
)
```

Predictor hidden_dim stays at 128. Total predictor params increase from ~131K to ~197K (+50%).

### VICReg

Apply VICReg to z_proj (512-dim) only, NOT to the full raw_embedding (768-dim). The skip portion is a deterministic copy of the pooled rep — VICReg's decorrelation pressure would fight the natural covariance structure of the GNN's output.

```python
# VICReg on the projected portion only
vicreg_loss = vicreg(z_proj)  # [batch, 512], same as before
```

### Contrastive loss

Operates on the 768-dim normalized embedding. The margin (1.5) may need recalibration — on a 768-dim unit sphere, the expected distance between random points is different from a 512-dim sphere. Specifically, for unit vectors in d dimensions, the expected squared distance between random points is 2(1 - 0) = 2 regardless of dimension (they're uncorrelated on the sphere). So the margin doesn't need to change.

### Aux heads

Still attach to the 256-dim pooled rep. Unchanged.

### EMA target encoder

The target encoder is an EMA copy. It also produces 768-dim embeddings (same architecture). No special handling needed — the EMA update copies all parameters including the output_proj, and the skip connection is parameter-free.

## Config Changes

```yaml
encoder:
  hidden_dim: 128
  embed_dim: 768          # was 512 — widened for skip connection
  proj_dim: 512           # NEW: the output_proj target dim (was equal to embed_dim)
  num_layers: 4
  num_heads: 4
  skip_pooled: true       # NEW: enable skip connection

predictor:
  hidden_dim: 128
  # input/output dims derived from encoder.embed_dim (768)

# Everything else unchanged
loss:
  margin: 1.5
  true_weight: 1.0
  vicreg_var_weight: 0.1
  vicreg_cov_weight: 0.1
  aux_weight: 0.1
  aux_heads: structural
```

## Testing Plan

### Test 1: Skip connection at h=128

```yaml
encoder.hidden_dim: 128
encoder.embed_dim: 768
encoder.proj_dim: 512
encoder.skip_pooled: true
seed: 1
# everything else same as aux_h128_seed1_diag
```

Compare against: aux_h128_seed1_diag (96.21% bal_acc, 512-dim embedding, no skip).

**Primary success criterion:** Reprobe the new 768-dim embedding for cohomological features. Per-equation cohomology probing accuracy should be ≥ 95% on the output embedding (up from 88% without skip). This is the direct measurement of whether the skip preserved the features the projection was destroying.

**Secondary criterion:** bal_acc ≥ 96.0% (no regression from the architecture change). A gain of +0.3pp or more would be meaningful given the ~1pp total headroom, but expect seed noise to dominate.

**Ablation (free, no retraining):** After training, evaluate with the skip channel zeroed out (only z_proj contributes, like the old architecture). Compare accuracy. The drop tells you how much the model learned to rely on the skip channel.

### Test 2: Reproducibility (different seed)

Same config, seed=42. Confirm the result is recipe-driven, not seed-lucky.

### Test 3: Skip connection at h=256 (if Test 1 succeeds)

Same recipe at h=256. The skip connection doesn't interact with the h=256 instability mechanisms (VICReg regime, simplicity bias) in any obvious way — it's a passive concatenation, not a competing loss term. But worth testing to see if the preserved features change the h=256 dynamics.

## What to Monitor

Same diagnostic suite as previous runs:

- Per-step: total loss, contrastive, per-class split, gradient norms (encoder, predictor, pre/post clip), VICReg var/cov, aux loss
- Per-epoch: bal_acc, TRUE/FALSE recall, threshold, effective rank, distance percentiles, gap, overlap

Plus:

```python
# Per-epoch: ablation metrics
# Zero the skip channel, evaluate accuracy
z_proj_only = torch.cat([z_proj, torch.zeros_like(pooled)], dim=1)
z_proj_only = F.normalize(z_proj_only, dim=-1)
accuracy_proj_only = evaluate(z_proj_only, ...)

# Zero the projection channel, evaluate accuracy
z_skip_only = torch.cat([torch.zeros_like(z_proj), pooled], dim=1)
z_skip_only = F.normalize(z_skip_only, dim=-1)
accuracy_skip_only = evaluate(z_skip_only, ...)

wandb.log({
    "val/accuracy_proj_only": accuracy_proj_only,
    "val/accuracy_skip_only": accuracy_skip_only,
    "val/accuracy_full": accuracy_full,
})
```

Track these per epoch to see how the model's reliance on each channel evolves during training. If the skip channel's contribution grows over epochs, the model is increasingly using the preserved features.

## Post-Training Probing

After Test 1, run the full probing suite from `probing_experiments_plan.md` on the new 768-dim embeddings. Key comparisons vs the 512-dim baseline:

| Probe | 512-dim baseline | 768-dim skip (expected) |
|---|---|---|
| Exp 1: structural | 99-100% | 99-100% (unchanged) |
| Exp 2: cohom summaries | 80-88% | **95%+** (the target) |
| Exp 3: cohom cells | 75% mean | **85%+** |
| Exp 4: non-anchored | varies | same or better |
| Exp 5: pairwise obstruction | 69% | **75%+** |
| Exp 6: equiv class NN | 52% top-1 | same or better |

The cohomology probes (Exp 2, 3, 5) are the primary success metrics. If these improve substantially, the skip connection preserved the information the projection was destroying.

## Why This Before the Dual-Encoder

The skip connection tests a simpler hypothesis: "the GNN already learned the features, we just need to stop the projection from destroying them." No new data, no new training signal, no new encoder pathway. One line of code.

If the skip connection brings cohomology probing from 88% to 95%+ AND pairwise obstruction from 69% to 80%+, the dual-encoder (which adds external cohomology data) may not be necessary — the information was already in the model, just being discarded.

If the skip connection doesn't help (cohomology probing stays at 88%), the projection isn't the bottleneck — the GNN's pooled rep doesn't actually carry the cohomological information to the output in a way the contrastive task can use. Then the dual-encoder (adding external data) becomes the right move.

## Notes for Claude Code

- The architecture change is one line in the forward pass + dimension changes in the predictor and config. Small diff, easy to review.
- The checkpoint from aux_h128_seed1_diag CANNOT be loaded — embedding dim changed from 512 to 768, predictor dims changed. Train from scratch.
- VICReg must apply to z_proj (512-dim) ONLY, not to the full 768-dim raw_embedding. The skip portion is a deterministic copy — VICReg's decorrelation pressure would fight the GNN's natural output structure.
- The effective rank computation should be on the full 768-dim embedding. Expect it to be higher than the 512-dim baseline (130-145) because the skip adds 256 dims of non-collapsed, information-rich features.
- The per-epoch channel ablation (zero proj vs zero skip) is the key diagnostic. Log it every epoch, not just at the end.
- The L2-normalization distributes norm across 768 dims instead of 512. The skip channel dims have different scale from the projected dims (pooled rep values vs learned projection values). This might cause the skip channel to dominate or be dominated depending on relative magnitudes. Monitor the norm of z_proj vs the norm of pooled across a batch at epoch 0 to check — if one is 10x the other, the smaller one is effectively invisible after L2-norm. Add a learned scalar or layer norm to balance them if needed.
- Commit the architecture change separately before running experiments.
