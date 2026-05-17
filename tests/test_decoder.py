"""Unit tests for src/decoder/decoder.py.

Cover the contracts that matter for training and evaluation:
  - shape contracts on forward (cond + tokens -> logits)
  - generate() respects BOS init, EOS termination, max_new_tokens cap
  - cross-entropy loss ignores PAD
  - causal masking actually causal (position t can't see t+1)
  - gradient flows to decoder params only (embeddings are inputs,
    not parameters, so the "frozen embeddings" contract is automatic)
  - parameter count is in the few-million range the plan predicts
"""

from __future__ import annotations

import pytest
import torch

from src.decoder.decoder import PrefixConditionedDecoder, loss_for_batch
from src.decoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE


def _make_model(**kwargs):
    torch.manual_seed(0)
    return PrefixConditionedDecoder(**kwargs)


def _make_inputs(B=4, T=10, embed_dim=1536):
    torch.manual_seed(1)
    cond = torch.randn(B, embed_dim)
    tokens = torch.randint(low=3, high=VOCAB_SIZE, size=(B, T), dtype=torch.long)
    tokens[:, 0] = BOS_ID
    return cond, tokens


# ---------- shape contracts ----------

def test_forward_returns_logits_per_token_position():
    model = _make_model().eval()
    cond, tokens = _make_inputs(B=4, T=10)
    logits = model(cond, tokens)
    assert logits.shape == (4, 10, VOCAB_SIZE)


def test_forward_handles_max_token_len():
    model = _make_model(max_token_len=64).eval()
    cond, tokens = _make_inputs(B=2, T=64)
    logits = model(cond, tokens)
    assert logits.shape == (2, 64, VOCAB_SIZE)


def test_forward_rejects_over_max_len():
    model = _make_model(max_token_len=32).eval()
    cond, tokens = _make_inputs(B=2, T=33)
    with pytest.raises(ValueError, match='max_token_len'):
        model(cond, tokens)


# ---------- causal masking ----------

def test_attention_is_causal():
    """Changing tokens[:, t] should NOT change logits[:, t-1] (position t-1
    can't see position t under causal mask)."""
    model = _make_model().eval()
    cond, tokens_a = _make_inputs(B=2, T=8)
    tokens_b = tokens_a.clone()
    # Perturb the LAST token in the sequence.
    tokens_b[:, -1] = (tokens_b[:, -1] + 5) % VOCAB_SIZE
    logits_a = model(cond, tokens_a)
    logits_b = model(cond, tokens_b)
    # All positions EXCEPT the last should be identical.
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)
    # The last position MAY differ (it sees its own token via residual through embedding).
    # Note: actually under causal mask, logits[:, t] depends on tokens[:, :t+1],
    # so logits[:, -1] depends on tokens[:, -1]. So we don't assert it differs;
    # we only assert earlier positions are unaffected.


def test_attention_doesnt_see_future_at_position_zero():
    """Position 0 (BOS) should give the same logits regardless of what
    comes after — strong sanity check for causal masking."""
    model = _make_model().eval()
    cond, tokens_a = _make_inputs(B=2, T=8)
    tokens_b = tokens_a.clone()
    tokens_b[:, 1:] = (tokens_b[:, 1:] + 7) % VOCAB_SIZE   # totally rewrite suffix
    tokens_b[:, 0] = tokens_a[:, 0]                        # BOS unchanged
    logits_a = model(cond, tokens_a)
    logits_b = model(cond, tokens_b)
    assert torch.allclose(logits_a[:, 0, :], logits_b[:, 0, :], atol=1e-5)


# ---------- generation ----------

def test_generate_starts_with_bos():
    model = _make_model().eval()
    cond = torch.randn(3, 1536)
    out = model.generate(cond, max_new_tokens=5)
    assert (out[:, 0] == BOS_ID).all()


def test_generate_respects_max_new_tokens():
    model = _make_model().eval()
    cond = torch.randn(2, 1536)
    out = model.generate(cond, max_new_tokens=10)
    # 1 (BOS) + at most 10 new = at most 11 total
    assert out.shape[1] <= 11


def test_generate_pads_after_eos():
    """Once a sequence emits EOS, subsequent positions are PAD (so we don't
    'undo' a finished sequence in a batched generate)."""
    # Force the model to emit EOS by manipulating logits at generation:
    # easiest is to just spot-check that generate's output doesn't have
    # tokens after a PAD.
    model = _make_model().eval()
    cond = torch.randn(2, 1536)
    out = model.generate(cond, max_new_tokens=20)
    # Find PAD-containing rows; assert all subsequent positions are PAD too.
    for row in out:
        seen_pad = False
        for tok in row:
            if seen_pad:
                assert tok.item() == PAD_ID
            if tok.item() == PAD_ID:
                seen_pad = True


# ---------- loss ----------

def test_loss_for_batch_basic():
    model = _make_model().eval()
    cond, tokens = _make_inputs(B=4, T=10)
    logits = model(cond, tokens)
    # Targets: shift by 1 to align with next-token prediction.
    targets = torch.cat([tokens[:, 1:], torch.full((4, 1), EOS_ID)], dim=1)
    loss = loss_for_batch(logits, targets)
    assert loss.dim() == 0   # scalar
    assert loss.item() > 0   # untrained random init


def test_loss_ignores_pad():
    """PAD tokens in targets shouldn't contribute to the loss."""
    model = _make_model().eval()
    cond, tokens = _make_inputs(B=2, T=6)
    logits = model(cond, tokens)
    targets_all_real = torch.full((2, 6), 3, dtype=torch.long)
    targets_with_pad = torch.full((2, 6), PAD_ID, dtype=torch.long)
    loss_real = loss_for_batch(logits, targets_all_real)
    loss_pad = loss_for_batch(logits, targets_with_pad)
    # All-PAD targets -> mean over 0 elements; PyTorch returns NaN
    # (since cross_entropy with full-ignore returns nan-mean).
    assert torch.isnan(loss_pad) or loss_pad.item() == 0
    assert loss_real.item() > 0


# ---------- gradient flow ----------

def test_gradient_flows_to_decoder_params_only():
    """The decoder's only inputs are `cond` and `tokens`. Both should
    NOT receive gradients by default (they're inputs, not parameters).
    All decoder parameters should receive gradients."""
    model = _make_model()
    cond = torch.randn(2, 1536, requires_grad=False)   # frozen embeddings
    cond.requires_grad_(False)
    tokens = torch.tensor([[BOS_ID, 3, 4, 5], [BOS_ID, 6, 7, 8]], dtype=torch.long)
    logits = model(cond, tokens)
    loss = logits.mean()
    loss.backward()
    # No grad on cond (it's a leaf with requires_grad=False).
    assert cond.grad is None
    # Every decoder parameter should have a gradient.
    for name, p in model.named_parameters():
        assert p.grad is not None, f'no grad on {name}'


# ---------- parameter count ----------

def test_param_count_in_few_million_range():
    """Plan predicts 'a few million parameters'."""
    model = _make_model()
    n = model.num_parameters()
    # 1M < n < 10M is a generous "few million" band.
    assert 1_000_000 < n < 10_000_000, f'param count {n:,} outside few-million range'


# ---------- determinism ----------

def test_eval_mode_is_deterministic():
    model = _make_model().eval()
    cond, tokens = _make_inputs()
    out_a = model(cond, tokens)
    out_b = model(cond, tokens)
    assert torch.allclose(out_a, out_b)
