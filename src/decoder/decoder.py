"""Prefix-conditioned autoregressive decoder for the proof-decoder phase 1.

Architecture per `proof_decoder_plan.md`:

  - Input conditioning: concat(z_e1, z_e2) ∈ R^{2*768} = R^{1536}.
  - Linear projection: 1536 → K * d_model, reshaped to K prefix tokens.
  - K=4, d_model=256, num_layers=4, nhead=4, learned positional encodings.
  - Token embeddings for the 16-token vocab.
  - Causal self-attention over [prefix_1, ..., prefix_K, BOS, t_1, ..., t_T].
  - Output head: Linear(d_model → vocab_size).
  - Pure embeddings-only conditioning (no arity input, per Methodology
    Resolution 2). Failure attribution via the nested metric stack at
    eval time.

Embeddings are frozen — only the conditioning projection and the
transformer/head train. (Frozen-ness is enforced at training time by
not passing the JEPA encoder's params to the optimizer; this module
takes the embeddings as inputs.)

Train: pass (cond, tokens) where tokens is `[BOS, t_1, ..., t_T]`;
output logits at positions 1..T predict tokens t_1..t_T+1 (with EOS
as the last target). Loss is token-level cross-entropy with
`ignore_index=PAD_ID`.

Generate: see ``PrefixConditionedDecoder.generate``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.decoder.tokenizer import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE


class PrefixConditionedDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int | None = None,
        num_prefix: int = 4,
        embed_dim: int = 1536,
        dropout: float = 0.1,
        max_token_len: int = 64,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_prefix = num_prefix
        self.max_token_len = max_token_len
        self.max_total_len = num_prefix + max_token_len

        # Conditioning: project [z_e1; z_e2] -> K * d_model -> K prefix tokens.
        self.conditioning = nn.Linear(embed_dim, num_prefix * d_model)

        # Token embeddings.
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        # Learned positional encodings cover both prefix and token positions.
        self.pos_emb = nn.Embedding(self.max_total_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Use TransformerEncoderLayer with a causal mask — equivalent to a
        # GPT-style decoder. norm_first=True for stability.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Output head.
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, total_len: int, device: torch.device) -> torch.Tensor:
        """Standard causal (upper-triangular True) attention mask of shape
        (L, L) where True positions are masked out."""
        return torch.triu(
            torch.ones(total_len, total_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, cond: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Compute logits for next-token prediction at every token position.

        Args:
            cond: ``[B, embed_dim]`` conditioning vector (concat z_e1, z_e2).
            tokens: ``[B, T]`` long tensor of input token IDs. Expected to
                start with BOS at position 0.

        Returns:
            ``[B, T, vocab_size]`` logits. Position t predicts what comes
            AFTER tokens[:, t] — so loss is between
            ``logits[:, t]`` and ``tokens[:, t + 1]`` (with EOS at the last
            target position). The caller handles teacher forcing / loss.
        """
        B, T = tokens.shape
        K = self.num_prefix
        if T > self.max_token_len:
            raise ValueError(f'tokens length {T} > max_token_len {self.max_token_len}')

        # Prefix tokens from conditioning.
        prefix = self.conditioning(cond).view(B, K, self.d_model)

        # Token embeddings (scaled by sqrt(d_model) per standard transformer).
        tok = self.token_emb(tokens)

        # Concat prefix + tokens.
        x = torch.cat([prefix, tok], dim=1)              # [B, K+T, d_model]
        L = K + T

        # Positional encoding.
        positions = torch.arange(L, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        x = self.dropout(x)

        # Causal self-attention.
        mask = self._causal_mask(L, x.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        # Drop the K prefix positions; only token positions get a softmax.
        x = x[:, K:, :]                                  # [B, T, d_model]

        return self.head(x)                              # [B, T, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        max_new_tokens: int = 32,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate a token sequence given the conditioning.

        Starts from BOS and rolls out until EOS or max_new_tokens.

        Args:
            cond: ``[B, embed_dim]``
            max_new_tokens: cap on generated length (not counting BOS).
            greedy: if True, take argmax; else sample from softmax.

        Returns:
            ``[B, T]`` long tensor where T = 1 + actual generated length
            (includes the leading BOS).
        """
        self.eval()
        B = cond.shape[0]
        device = cond.device
        tokens = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            logits = self.forward(cond, tokens)         # [B, T, vocab]
            next_logits = logits[:, -1, :]              # [B, vocab]
            if greedy:
                nxt = next_logits.argmax(dim=-1)        # [B]
            else:
                probs = F.softmax(next_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Keep PAD for already-finished sequences so we don't extend them.
            nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
            tokens = torch.cat([tokens, nxt.unsqueeze(-1)], dim=-1)
            finished = finished | (nxt == EOS_ID)
            if finished.all():
                break
        return tokens

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def loss_for_batch(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Given ``logits[:, t]`` predicting ``target_tokens[:, t]`` (callers
    should align by passing inputs = tokens[:, :-1] and targets =
    tokens[:, 1:]), compute mean cross-entropy with PAD ignored.
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, V),
        target_tokens.reshape(B * T),
        ignore_index=PAD_ID,
    )
