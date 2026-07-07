# Multi-Head Attention in the Transformer Block — A Deep Dive

*Compiled by D. Gueorguiev with Claude Opus 4.7 — May 11, 2026*

---

## Table of Contents

- [1. Scope and setup](#1-scope-and-setup)
- [2. The MHA pipeline at a glance](#2-the-mha-pipeline-at-a-glance)
- [3. Step-by-step deep dive](#3-step-by-step-deep-dive)
  - [3.1 Linear projections — producing Q, K, V](#31-linear-projections--producing-q-k-v)
  - [3.2 Scaled dot-product scores](#32-scaled-dot-product-scores)
  - [3.3 Causal masking (decoder / autoregressive only) — deep dive](#33-causal-masking-decoder--autoregressive-only--deep-dive)
  - [3.4 Row-wise softmax — the attention pattern](#34-row-wise-softmax--the-attention-pattern)
  - [3.5 Weighted average of values](#35-weighted-average-of-values)
  - [3.6 Concatenating heads](#36-concatenating-heads)
  - [3.7 Output projection — mixing the heads](#37-output-projection--mixing-the-heads)
  - [3.8 The Add & Norm wrapper](#38-the-add--norm-wrapper)
- [4. The MHA equation in one line](#4-the-mha-equation-in-one-line)
- [5. A minimal reference implementation](#5-a-minimal-reference-implementation)
- [6. The dynamical-systems perspective](#6-the-dynamical-systems-perspective)
- [7. From O to the next predicted token](#7-from-o-to-the-next-predicted-token)
  - [7.1 The nature of the MHA output O](#71-the-nature-of-the-mha-output-o)
  - [7.2 The seven-step journey from O to a predicted token](#72-the-seven-step-journey-from-o-to-a-predicted-token)
- [8. The residual stream and the residual-stream space](#8-the-residual-stream-and-the-residual-stream-space)
  - [8.1 The basic idea](#81-the-basic-idea)
  - [8.2 The residual-stream space](#82-the-residual-stream-space)
  - [8.3 The read/write asymmetry](#83-the-readwrite-asymmetry)
  - [8.4 The "stream" metaphor](#84-the-stream-metaphor)
  - [8.5 The four roles of weight matrices](#85-the-four-roles-of-weight-matrices)
  - [8.6 Consequences of the residual-stream reframing](#86-consequences-of-the-residual-stream-reframing)
  - [8.7 A subtle implementation point](#87-a-subtle-implementation-point)
- [9. The unembedding matrix W_U](#9-the-unembedding-matrix-w_u)
  - [9.1 Definition and shape](#91-definition-and-shape)
  - [9.2 Parameter scale](#92-parameter-scale)
  - [9.3 Weight tying: W_U and W_E as one matrix](#93-weight-tying-w_u-and-w_e-as-one-matrix)
  - [9.4 Geometric interpretation: dot products with token directions](#94-geometric-interpretation-dot-products-with-token-directions)
  - [9.5 W_U is not an isometry](#95-w_u-is-not-an-isometry)
  - [9.6 Over-complete frame structure](#96-over-complete-frame-structure)
  - [9.7 What the unembedding does, mechanistically](#97-what-the-unembedding-does-mechanistically)
  - [9.8 How W_U is trained](#98-how-w_u-is-trained)
  - [9.9 Summary of W_U properties](#99-summary-of-w_u-properties)
- [10. Training dynamics: the autograd view](#10-training-dynamics-the-autograd-view)
  - [10.1 What is learnable in a transformer](#101-what-is-learnable-in-a-transformer)
  - [10.2 Two meanings of "embedding" — a crucial distinction](#102-two-meanings-of-embedding--a-crucial-distinction)
  - [10.3 The autograd graph built by the forward pass](#103-the-autograd-graph-built-by-the-forward-pass)
  - [10.4 The backward pass through MHA](#104-the-backward-pass-through-mha)
  - [10.5 The optimizer step](#105-the-optimizer-step)
  - [10.6 One training iteration end-to-end](#106-one-training-iteration-end-to-end)
  - [10.7 The random-initialization trajectory](#107-the-random-initialization-trajectory)
  - [10.8 Practical training concerns](#108-practical-training-concerns)
- [11. Summary table](#11-summary-table)
- [Appendix A — Fused multi-head projections in PyTorch](#appendix-a--fused-multi-head-projections-in-pytorch)
  - [A.1 Recap: why this fusion works](#a1-recap-why-this-fusion-works)
  - [A.2 Equivalence demonstration — naive per-head vs. fused matmul](#a2-equivalence-demonstration--naive-per-head-vs-fused-matmul)
  - [A.3 A production nn.Module](#a3-a-production-nnmodule)
  - [A.4 One step further — the fused QKV matmul](#a4-one-step-further--the-fused-qkv-matmul)
  - [A.5 Cost summary](#a5-cost-summary)
- [12. References](#12-references)

---

## 1. Scope and setup

This document deconstructs the **Multi-Head Attention (MHA)** sub-layer of a single transformer block, traces exactly how it transforms an input matrix $H \in \mathbb{R}^{T \times d}$ into an output of the same shape, shows how the surrounding residual + LayerNorm wrapper closes the loop, and follows the MHA output through the rest of the network to the final next-token prediction. Along the way it develops the **residual-stream view** of transformer architecture, examines the **unembedding matrix** $W_U$ as the bridge from residual-stream space to vocabulary space, and closes with a **training-dynamics** section that walks the PyTorch autograd graph backward through MHA to show how every weight is learned end-to-end. The treatment is post-LN (the original Vaswani et al. convention); the pre-LN variant used in GPT-2 is noted where it differs.

We fix notation as follows:

| Symbol | Meaning | Typical value |
|---|---|---|
| $T$ | sequence length (number of token positions) | up to 1024 / 2048 / 8192 |
| $d$ | model dimension (a.k.a. $d_{\text{model}}$) | 512 (Vaswani), 768 (GPT-2 small) |
| $h$ | number of attention heads | 8 (Vaswani), 12 (GPT-2 small) |
| $d_k$ | per-head dimension, $= d / h$ | 64 |
| $H^{(\ell)}$ | residual-stream tensor at layer $\ell$, shape $T \times d$ | — |
| $\ell$ | block (depth) index, $\ell = 1, \dots, L$ | $L = 12$ for GPT-2 small |

For the **first block** the input is $H^{(0)}$ = token embeddings + positional encoding. For subsequent blocks the input is the output of the previous block.

---

## 2. The MHA pipeline at a glance

```mermaid
flowchart TD
    H["H in R^(TxD)"] --> P{{"Linear projections"}}
    P -->|"H*W_Q"| Q["Q in R^(TxD)"]
    P -->|"H*W_K"| K["K in R^(TxD)"]
    P -->|"H*W_V"| V["V in R^(TxD)"]
    Q --> RS["Reshape -> (T, h, d_k) -> transpose -> (h, T, d_k)"]
    K --> RS
    V --> RS
    RS --> S["Scores: S = QK^T / sqrt(d_k), shape (h, T, T)"]
    S --> M["Optional causal mask"]
    M --> A["A = softmax_row(S)"]
    A --> O["Per-head output: O = A*V, shape (h, T, d_k)"]
    O --> C["Concat heads -> (T, d)"]
    C --> WO["Output projection: *W_O"]
    WO --> Out["MHA(H) in R^(TxD)"]
    Out --> R["+ residual H"]
    R --> LN["LayerNorm"]
    LN --> Z["Z in R^(TxD) -- sub-layer output"]
```

The flow has seven conceptual stages: **(1)** linear projections to Q/K/V, **(2)** scaled dot-product scoring, **(2.5)** optional causal mask, **(3)** row-wise softmax, **(4)** value averaging, **(5)** head concatenation, **(6)** output projection, **(7)** Add & Norm. Each is examined below.

---

## 3. Step-by-step deep dive

### 3.1 Linear projections — producing Q, K, V

**Learnable parameters per head $i \in \{1, \dots, h\}$:**

$$W_Q^{(i)},  W_K^{(i)},  W_V^{(i)} \in \mathbb{R}^{d \times d_k}$$

**Plus a single output-mixing matrix:**

$$W_O \in \mathbb{R}^{d \times d}$$

In practice, the $h$ per-head matrices are stacked side-by-side into single $d \times d$ matrices $W_Q$, $W_K$, $W_V$ so that one fused matmul produces all heads at once. The "split into heads" then becomes a tensor reshape, not a separate computation.

**Per-head projection:**

$$Q^{(i)} = H  W_Q^{(i)}, \qquad K^{(i)} = H  W_K^{(i)}, \qquad V^{(i)} = H  W_V^{(i)}, \qquad Q^{(i)}, K^{(i)}, V^{(i)} \in \mathbb{R}^{T \times d_k}$$

**Row-by-row, for token $t$:**

$$q_t^{(i)} = h_t W_Q^{(i)}, \qquad k_t^{(i)} = h_t W_K^{(i)}, \qquad v_t^{(i)} = h_t W_V^{(i)}, \qquad \in \mathbb{R}^{d_k}$$

The canonical mechanistic reading:

- $q_t^{(i)}$ — **query**: encodes what token $t$ is looking for in the context.
- $k_t^{(i)}$ — **key**: encodes what token $t$ advertises to other positions.
- $v_t^{(i)}$ — **value**: encodes what token $t$ contributes when attended to.

$Q$ and $K$ live in the same $d_k$-dimensional *matching space*; $V$ lives in its own $d_k$-dimensional *content space*. There is no a priori reason for these to coincide, and indeed $W_Q, W_K, W_V$ are independent learnable parameters.

```python
# PyTorch — fused projection, then reshape into heads
B, T, d = x.shape                      # batch, seq, model dim
h, d_k = num_heads, d // num_heads

q = self.W_q(x)                        # (B, T, d)
k = self.W_k(x)                        # (B, T, d)
v = self.W_v(x)                        # (B, T, d)

# Split d → (h, d_k) and bring heads to the leading position
q = q.view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
k = k.view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
v = v.view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
```

### 3.2 Scaled dot-product scores

For each head $i$, compute a $T \times T$ matrix of pairwise affinities:

$$S^{(i)} = \frac{Q^{(i)} \big(K^{(i)}\big)^{\top}}{\sqrt{d_k}}, \qquad S^{(i)} \in \mathbb{R}^{T \times T}$$

Entry-wise:

$$S^{(i)}_{t,t'} = \frac{q_t^{(i)} \cdot k_{t'}^{(i)}}{\sqrt{d_k}}$$

This is the **affinity between token $t$'s query and token $t'$'s key**, in head $i$. Large positive values mean strong attraction; large negative values, strong repulsion.

#### Motivation for the $\sqrt{d_k}$ scaling factor

Suppose the components of $q_t^{(i)}$ and $k_{t'}^{(i)}$ are approximately independent zero-mean unit-variance random variables. Then their dot product

$$q_t^{(i)} \cdot k_{t'}^{(i)} = \sum_{j=1}^{d_k} q_{t,j}^{(i)} k_{t',j}^{(i)}$$

has

$$\mathbb{E}\big[q_t^{(i)} \cdot k_{t'}^{(i)}\big] = 0, \qquad \mathrm{Var}\big[q_t^{(i)} \cdot k_{t'}^{(i)}\big] = d_k$$

Without the $1/\sqrt{d_k}$ factor, the variance of the pre-softmax scores grows linearly with $d_k$. The softmax then saturates — one entry approaches 1, all others approach 0 — and the Jacobian collapses, killing gradients. Dividing by $\sqrt{d_k}$ restores unit variance and keeps the softmax in its responsive regime regardless of $d_k$.

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # (B, h, T, T)
```

### 3.3 Causal masking (decoder / autoregressive only) — deep dive

For causal models (GPT-2, decoder blocks in Vaswani), add a mask $M \in \mathbb{R}^{T \times T}$ where

$$
M_{t,t'} = \begin{cases}
0 & \text{if } t' \leq t \quad \text{(present or past -- allowed)} \\\\
-\infty & \text{if } t' \gt t \quad \text{(future -- forbidden)}
\end{cases}
$$

so that

$$S'^{(i)} = S^{(i)} + M$$

For **encoder blocks** (BERT, the encoder half of Vaswani) there is no mask — every position attends to every position.

#### How the mask is applied

Concretely, $M$ is upper-triangular with $-\infty$ above the main diagonal and $0$ on and below it:

$$
M = \begin{pmatrix}
0 & -\infty & -\infty & \cdots & -\infty \\\\
0 & 0 & -\infty & \cdots & -\infty \\\\
0 & 0 & 0 & \cdots & -\infty \\\\
\vdots & \vdots & \vdots & \ddots & \vdots \\\\
0 & 0 & 0 & \cdots & 0
\end{pmatrix}
$$

The mask is added **before** the softmax, not after. The resulting masked softmax is

$$A^{(i)}_{t,t'} = \frac{\exp\big(S^{(i)}_{t,t'} + M_{t,t'}\big)}{\sum_{u=1}^{T} \exp\big(S^{(i)}_{t,u} + M_{t,u}\big)}$$

#### Pre-softmax additive masking vs. post-softmax zeroing

The key identity is

$$\exp\big(S^{(i)}_{t,t'} + (-\infty)\big) = \exp(-\infty) = 0$$

So for any forbidden $(t, t')$ pair, the *numerator* in the softmax is exactly zero, and that entry contributes nothing to the *denominator* either. The allowed entries are then renormalized cleanly — they sum to 1 over only the past-and-present positions.

If you instead applied the mask **after** the softmax — zeroing out the forbidden entries post-hoc — two things would go wrong:

1. **The row would no longer sum to 1**, forcing a manual renormalization step.
2. **The gradients would leak information about future tokens during training**, because the masked entries would still have non-zero gradients through the unmasked softmax. The pre-softmax additive mask gets the math right in a single step.

#### Numerical detail: implementations use a large finite negative number

In production code the "$-\infty$" is realized as a large finite negative value (typically $-10^9$, or `float('-inf')` via masked-fill) for two reasons:

- `exp(-inf) = 0` is well-defined in IEEE 754, but `inf - inf = nan`, which can arise in rare edge cases (e.g., an entire row masked out due to padding combined with causal masking).
- A value like $-10^4$ is already enough: $\exp(-10^4)$ underflows to zero in float32 anyway.

PyTorch's standard `masked_fill(mask, float('-inf'))` works correctly for the canonical causal case because every row has at least one allowed entry (the diagonal), so the denominator is never zero. The `-1e9` convention is a defensive choice that survives pathological cases.

#### Worked example, $T = 4$

Suppose the raw scores for a single head are

$$
S = \begin{pmatrix}
2.0 & 1.5 & 0.3 & 0.8 \\\\
1.0 & 2.5 & 1.8 & 0.4 \\\\
0.5 & 1.2 & 3.0 & 1.1 \\\\
0.7 & 0.9 & 1.4 & 2.2
\end{pmatrix}
$$

After adding $M$:

$$
S' = S + M = \begin{pmatrix}
2.0 & -\infty & -\infty & -\infty \\\\
1.0 & 2.5 & -\infty & -\infty \\\\
0.5 & 1.2 & 3.0 & -\infty \\\\
0.7 & 0.9 & 1.4 & 2.2
\end{pmatrix}
$$

After row-wise softmax:

$$
A = \begin{pmatrix}
1.000 & 0.000 & 0.000 & 0.000 \\\\
0.182 & 0.818 & 0.000 & 0.000 \\\\
0.057 & 0.115 & 0.828 & 0.000 \\\\
0.106 & 0.130 & 0.214 & 0.550
\end{pmatrix}
$$

Observations:

- **Row 1** (token 1) attends only to itself — the only allowed position. $A_{1,1} = 1$ regardless of the original score values.
- **Row 2** (token 2) attends to tokens 1 and 2, with weights depending only on $S_{2,1}$ and $S_{2,2}$.
- **Each row sums to exactly 1**, normalized over only the allowed (past-and-present) positions.
- **The lower-triangular structure** of $A$ is the visible signature of the causal constraint.

#### PyTorch implementation

```python
# Build the causal mask once (T × T boolean, True = forbidden)
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool, device=x.device),
    diagonal=1,
)
# diagonal=1 → strictly above the main diagonal is True;
# the diagonal itself is False because token t IS allowed to attend to itself.

# Apply to scores; broadcasts across (B, h)
scores = scores.masked_fill(mask, float('-inf'))

# Row-wise softmax now produces the lower-triangular attention pattern
attn = F.softmax(scores, dim=-1)
```

In efficient implementations (Flash Attention, xformers, PyTorch's `F.scaled_dot_product_attention(..., is_causal=True)`), the mask is **never materialized** as a $T \times T$ tensor. The kernel simply skips the upper-triangular work entirely, saving both memory ($O(T^2)$ for the mask) and roughly halving the attention FLOPs. Materializing the mask is fine for small $T$ but becomes a real bottleneck at long context lengths.

#### Architectural significance

The causal mask is what makes a transformer **autoregressive at the level of the loss**: during training, the model sees the entire sequence in one forward pass, but the mask guarantees that the prediction for token $t+1$ depends only on tokens $1, \dots, t$. This is the source of the *training parallelism* that makes transformers efficient — every position's loss is computed simultaneously, but each position's computation respects the autoregressive ordering.

At inference time, the mask is what makes **KV-caching** work: once the keys and values for tokens $1, \dots, t$ have been computed, they are frozen — no future token can affect them — so they can be cached, and only token $t+1$ requires fresh work. Without the mask, every token's representation would depend on every other token's, and incremental decoding would be impossible.

### 3.4 Row-wise softmax — the attention pattern

$$A^{(i)} = \mathrm{softmax}_{\text{row}}\big(S^{(i)}\big), \qquad A^{(i)} \in \mathbb{R}^{T \times T}$$

Given the prior node, $S^{(i)}$ has shape $(T, T)$ with $S^{(i)} = Q^{(i)}(K^{(i)})^\top / \sqrt{d_k}$. The `_row` qualifier means softmax is applied along the **last axis** (the key/column index) independently for each query row. So for a single head $i$, with query index $t$ and key index $t'$:

$$A^{(i)}_{t,t'} = \mathrm{softmax}_{\text{row}}(S^{(i)})_{t,t'} = \frac{\exp\big(S^{(i)}_{t,t'}\big)}{\sum_{u=1}^{T} \exp\big(S^{(i)}_{t,u}\big)}$$

Each row $t$ is normalized over $u = 1, \dots, T$, so every row of $A^{(i)}$ sums to 1 and $A^{(i)}$ keeps shape $(T, T)$. In practice it is the numerically stable form with the per-row max subtracted:

$$A^{(i)}_{t,t'} = \frac{\exp\big(S^{(i)}_{t,t'} - m_t\big)}{\sum_{u=1}^{T} \exp\big(S^{(i)}_{t,u} - m_t\big)}, \qquad m_t = \max_{u} S^{(i)}_{t,u}$$

which is mathematically identical but avoids overflow.

One thing worth flagging given the diagram ordering: the **optional causal mask** sits *before* this node, so the masking is applied to $S^{(i)}$ (setting $S^{(i)}_{t,t'} = -\infty$ for $t' \gt t$) prior to the exp. Those entries then map to exactly 0 in $A^{(i)}$, and the normalization denominator only runs over the unmasked keys $u \le t$. The formula above defines `softmax_row` with the sum over all $u = 1, \dots, T$ — this is consistent only because the masked logits have already been driven to $-\infty$ upstream, so their $\exp$ terms contribute exactly zero to both numerator and denominator.

Each row of $A^{(i)}$ is a probability distribution over the $T$ key positions. This matrix is the **attention pattern** — the object interpretability researchers stare at to understand what a head is doing. Recurring motifs include:

- **previous-token heads** ($A_{t, t-1} \approx 1$, all other entries near zero)
- **first-token heads** ($A_{t, 1} \approx 1$, anchoring on `<bos>`)
- **induction heads** (matching repeated bigrams; Olsson et al. 2022)
- **subject-of-verb heads** (long-range syntactic linkage)

```python
attn = F.softmax(scores, dim=-1)   # (B, h, T, T), rows sum to 1
```

### 3.5 Weighted average of values

$$O^{(i)} = A^{(i)} V^{(i)}, \qquad O^{(i)} \in \mathbb{R}^{T \times d_k}$$

Row $t$:

$$o_t^{(i)} = \sum_{t'=1}^{T} A^{(i)}_{t,t'}  v_{t'}^{(i)}$$

This is a **convex combination** of the value vectors weighted by attention. For each token $t$, head $i$ produces a $d_k$-dimensional vector summarizing the information pulled from the rest of the sequence.

```python
out_per_head = attn @ v   # (B, h, T, d_k)
```

### 3.6 Concatenating heads

Stack the $h$ per-head outputs along the channel axis:

$$O = \mathrm{Concat}\big(O^{(1)}, O^{(2)}, \dots, O^{(h)}\big) \in \mathbb{R}^{T \times d}$$

Since $h \cdot d_k = d$ by construction, the concatenation brings us back to the model dimension.

```python
# (B, h, T, d_k) → (B, T, h, d_k) → (B, T, d)
out_concat = out_per_head.transpose(1, 2).contiguous().view(B, T, d)
```

### 3.7 Output projection — mixing the heads

$$\mathrm{MHA}(H) = O  W_O, \qquad W_O \in \mathbb{R}^{d \times d}$$

Without $W_O$, the $h$ heads would write into **disjoint** $d_k$-dimensional channel slices of the output. $W_O$ is the linear map that allows them to recombine and share information across the channel axis. It is essential — removing it severely degrades the model. In mechanistic interpretability, $W_V W_O$ is often analyzed as a single "OV circuit" governing what gets written to the residual stream, while $W_Q W_K^\top$ is the "QK circuit" governing what gets attended to.

```python
output = self.W_o(out_concat)   # (B, T, d)
```

### 3.8 The Add & Norm wrapper

The MHA box in the Vaswani diagram is followed by an "Add & Norm" box. In the **post-LN** convention:

$$Z = \mathrm{LN}\big(H + \mathrm{MHA}(H)\big)$$

where LayerNorm operates **per token** (over the $d$ channel dimension):

$$\mathrm{LN}(x)_j = \gamma_j \cdot \frac{x_j - \mu(x)}{\sqrt{\sigma^2(x) + \epsilon}} + \beta_j$$

with

$$\mu(x) = \frac{1}{d} \sum_{j=1}^{d} x_j, \qquad \sigma^2(x) = \frac{1}{d} \sum_{j=1}^{d} \big(x_j - \mu(x)\big)^2$$

and $\gamma, \beta \in \mathbb{R}^d$ learnable scale and shift. The $\epsilon$ (typically $10^{-5}$) prevents division by zero.

In the **pre-LN** convention used by GPT-2 and most modern decoder-only models, the order is flipped:

$$Z = H + \mathrm{MHA}\big(\mathrm{LN}(H)\big)$$

This is more than a stylistic choice — it has substantive consequences:

- **Post-LN**: the residual stream is renormalized after every sub-layer; signals are clipped, training requires careful warmup, but representations stay bounded.
- **Pre-LN**: the residual stream accumulates unnormalized contributions; LN is applied only on the *read* side (into attention / FFN / unembedding). Training is more stable and the model is more amenable to depth scaling, but the residual stream's norm grows with depth.

---

## 4. The MHA equation in one line

Pulling everything together, the multi-head attention transformation $H \mapsto \mathrm{MHA}(H)$ is

$$\boxed{ \mathrm{MHA}(H) = \mathrm{Concat}_{i=1}^{h}\Big[\mathrm{softmax}\Big(\frac{(H W_Q^{(i)})(H W_K^{(i)})^{\top}}{\sqrt{d_k}}\Big) H W_V^{(i)}\Big] W_O }$$

with the surrounding sub-layer being either $\mathrm{LN}(H + \mathrm{MHA}(H))$ (post-LN) or $H + \mathrm{MHA}(\mathrm{LN}(H))$ (pre-LN).

---

## 5. A minimal reference implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention as described in Vaswani et al. (2017).

    Input/output shape: (B, T, d) where d = num_heads * d_k.
    Causal flag enables the autoregressive mask used in GPT-style models.
    """
    def __init__(self, d_model: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        # Fused per-head projections — each is d_model → d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape

        # 1. Project to Q, K, V and split into heads
        q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        k = self.W_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)      # (B, h, T, T)

        # 2.5. Causal mask
        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        # 3. Row-wise softmax → attention pattern A
        attn = F.softmax(scores, dim=-1)                              # (B, h, T, T)

        # 4. Weighted average of values
        out = attn @ v                                                # (B, h, T, d_k)

        # 5. Concatenate heads back into (B, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 6. Output projection
        return self.W_o(out)                                          # (B, T, d)


class TransformerBlockPreLN(nn.Module):
    """
    A single pre-LN decoder-only block (GPT-2 style).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, causal=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Pre-LN: H ← H + Sublayer(LN(H))
        h = h + self.attn(self.ln1(h))
        h = h + self.ffn(self.ln2(h))
        return h
```

---

## 6. The dynamical-systems perspective

Because each block is a **shape-preserving** map $\mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$, the depth-indexed sequence $H^{(0)}, H^{(1)}, \dots, H^{(L)}$ is a discrete trajectory in a single fixed state space. Decomposing the block update (pre-LN):

$$h_t^{(\ell+1)} - h_t^{(\ell)} = \underbrace{\mathrm{MHA}\big(\mathrm{LN}(H^{(\ell)})\big)_t}_{\text{non-local in } t} + \underbrace{\mathrm{FFN}\big(\mathrm{LN}(\tilde H^{(\ell)})\big)_t}_{\text{local in } t}$$

where $\tilde H^{(\ell)} = H^{(\ell)} + \mathrm{MHA}(\mathrm{LN}(H^{(\ell)}))$ is the post-attention residual stream.

```mermaid
flowchart LR
    H["H^(l)"] --> LN1["LN"]
    LN1 --> MHA["MHA<br>(non-local in t)"]
    MHA --> Add1["+"]
    H --> Add1
    Add1 --> Htilde["H_tilde^(l)"]
    Htilde --> LN2["LN"]
    LN2 --> FFN["FFN<br>(local in t)"]
    FFN --> Add2["+"]
    Htilde --> Add2
    Add2 --> Hnext["H^(l+1)"]
```

Two structural facts that matter for transformer analysis:

1. **The FFN sub-layer is strictly local in the position index $t$**. It processes each token's vector independently, contributing a per-token term to the update at each layer.

2. **The MHA sub-layer is the only non-local coupling between positions**. Its "interaction kernel" $A^{(i)}_{t,t'}$ is itself **state-dependent**, because $A$ depends on $h_t$ and $h_{t'}$ through $Q$ and $K$. This is a many-body interaction whose coupling strength is set by the configuration itself — structurally more like a self-consistent mean-field term than a fixed two-body potential.

---

## 7. From $O$ to the next predicted token

The MHA output $O$ is **not** the predicted next token. The gap between $O$ and an actual token prediction spans the rest of the entire transformer plus the unembedding step. This section traces that gap explicitly.

### 7.1 The nature of the MHA output $O$

After concatenation and the output projection, MHA produces

$$\mathrm{MHA}(H) = OW_O \in \mathbb{R}^{T \times d}$$

This is still a matrix in the **residual-stream space** $\mathbb{R}^{T \times d}$. Each row is a $d$-dimensional vector — for GPT-2 small, $d = 768$. It is:

- **Not** an integer index into a vocabulary
- **Not** a probability distribution over tokens
- **Not** a prediction

It is one **contribution** to the per-position hidden state. The role of MHA at layer $\ell$ is to compute the additive update that attention applies to each token's representation at this layer, and $OW_O$ is exactly that update.

A predicted token, by contrast, is either an integer in $\{1, \dots, V\}$ (for argmax decoding) or a probability vector in $\mathbb{R}^V$ where $V \approx 50{,}000$. Neither shape matches $O$.

### 7.2 The seven-step journey from $O$ to a predicted token

Suppose $OW_O$ is the MHA output at layer $\ell$. To convert it into a next-token prediction at the *last* position requires the following steps:

```mermaid
flowchart TD
    O["MHA(H) at layer l<br>shape (T, d)"] --> S1["1. Add residual + LN<br>-> Z, shape (T, d)"]
    S1 --> S2["2. FFN sub-layer of block l<br>-> H^(l), shape (T, d)"]
    S2 --> S3["3. Pass through blocks l+1, ..., L<br>-> H^(L), shape (T, d)"]
    S3 --> S4["4. Final LayerNorm (pre-LN models)<br>-> shape (T, d)"]
    S4 --> S5["5. Unembed with W_U<br>logits = H * W_U, shape (T, V)"]
    S5 --> S6["6. Row-wise softmax<br>-> P, shape (T, V)"]
    S6 --> S7["7. Take row t=T, sample or argmax<br>-> next token x_(T+1)"]
```

Explicitly:

1. **Finish the attention sub-layer:** $Z = \mathrm{LN}(H + \mathrm{MHA}(H))$ (post-LN) or $Z = H + \mathrm{MHA}(\mathrm{LN}(H))$ (pre-LN).
2. **Apply the FFN sub-layer** of the same block: $H^{(\ell)} = \mathrm{LN}(Z + \mathrm{FFN}(Z))$ (post-LN equivalent).
3. **Propagate through all remaining blocks** $\ell+1, \ell+2, \dots, L$.
4. **Apply the final LayerNorm** (in pre-LN models like GPT-2): $\tilde H^{(L)} = \mathrm{LN}_{\text{final}}(H^{(L)})$.
5. **Unembed:** $\text{logits} = \tilde H^{(L)}W_U \in \mathbb{R}^{T \times V}$.
6. **Row-wise softmax:** $P_t = \mathrm{softmax}(\text{logits}_t) \in \mathbb{R}^{V}$.
7. **Take row $t = T$** (the last position) and sample or take argmax to get the actual next token.

If $\ell = 1$ (the very first block), there are $L-1$ more blocks plus the unembedding step between $O$ and a predicted token. Even at $\ell = L$ (the final block), $O$ is still a hidden vector in $\mathbb{R}^d$ — only after the final LN and $W_U$ does it become a vocabulary-space quantity.

---

## 8. The residual stream and the residual-stream space

The MHA output $O$ does not stand alone — it joins a running quantity called the **residual stream**, a reframing of the transformer architecture that became standard after Elhage et al.'s *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021). This section makes the residual-stream view explicit.

### 8.1 The basic idea

The residual stream is the **per-position state vector that flows through the entire transformer from input to output, getting additively updated by each sub-layer along the way**. Instead of viewing a transformer as a stack of nonlinear functions $f_L \circ \cdots \circ f_1$, view it as a single shared workspace — one $d$-dimensional vector per token position — into which every sub-layer reads, computes something, and writes its contribution back via addition.

The structural fact that enables this view: **every sub-layer is wrapped in a residual connection** ($H \mapsto H + \text{stuff}$). Unrolling the $L$ pre-LN blocks gives

$$H^{(L)} = H^{(0)} + \sum_{\ell=1}^{L} \mathrm{MHA}^{(\ell)}\big(\mathrm{LN}_1(H^{(\ell-1)})\big) + \sum_{\ell=1}^{L} \mathrm{FFN}^{(\ell)}\big(\mathrm{LN}_2(\tilde H^{(\ell)})\big)$$

The network output is the *sum* of the input embedding and $2L$ additive contributions — one per sub-layer. Each sub-layer reads from the current stream value, computes a contribution, and writes it back. The stream itself never gets multiplied by anything; it just accumulates terms.

### 8.2 The residual-stream space

The space the stream lives in is

$$\mathcal{R} = \mathbb{R}^{T \times d}$$

For a single token position $t$, the per-position residual stream is $\mathcal{R}_t = \mathbb{R}^d$. The dimensionality $d$ is the **bandwidth** of the stream: everything the model wants to communicate from one layer to another — features, syntactic information, positional signals, abstractions — must fit into these $d$ channels. This is the structural origin of *superposition*: there are typically far more concepts the model wants to represent ($\gg d$) than there are dimensions, so concepts get encoded as overlapping directions.

### 8.3 The read/write asymmetry

A key structural feature of pre-LN architectures: **sub-layers read from the stream through a LayerNorm, but write to the stream directly.**

- **Reading**: the sub-layer sees $\mathrm{LN}(H)$, which strips away any uniform scaling that has accumulated in the stream. Sub-layers can only access the *direction* of the stream, not its absolute magnitude.
- **Writing**: the sub-layer's output is added directly to the unnormalized stream. No LN on the way out.

This asymmetry has a striking consequence: the stream's *norm* grows monotonically with depth (each contribution is added without renormalization), but its *direction* is what each sub-layer actually acts on. Empirically the norm of the residual stream in GPT-2 grows by a factor of roughly $10\text{–}100\times$ from input to output.

### 8.4 The "stream" metaphor

```mermaid
flowchart LR
    E["Embedding + PE<br>H^(0)"] --> S0("Residual stream")
    S0 --> MHA1["MHA-1"]
    MHA1 -->|write| S0
    S0 --> FFN1["FFN-1"]
    FFN1 -->|write| S0
    S0 --> MHA2["MHA-2"]
    MHA2 -->|write| S0
    S0 --> FFN2["FFN-2"]
    FFN2 -->|write| S0
    S0 -.->|"... L blocks ..."| SL("H^(L)")
    SL --> Unembed["W_U"]
    Unembed --> Logits["logits"]
```

Think of a stream of water flowing through the network from input embeddings at $\ell = 0$ to the unembedding at $\ell = L$. Sub-layers are tributaries pouring their output into the stream. Once written, a contribution is carried forward — it can only be cancelled by a downstream sub-layer writing an opposing contribution. This maps cleanly onto interpretability findings:

- **Information accumulates rather than transforms.** An attention head can write a "subject–verb agreement signal" at layer 5, and that signal is still present (possibly modified by later additions) at layer 11 where another head reads and uses it.
- **Sub-layers communicate via shared addresses.** Two heads in different layers communicate by writing and reading the same subspace of $\mathbb{R}^d$. This is how multi-layer circuits — induction heads (Olsson et al. 2022), IOI circuits (Wang et al. 2023) — work: head $A$ in layer $\ell$ writes to a subspace, head $B$ in layer $\ell'$ reads from it via its $Q$ or $K$ projection.
- **No bottlenecks.** Because everything is additive, there is no point where information must be re-encoded or compressed. The full $d$-dimensional channel is available at every layer.

### 8.5 The four roles of weight matrices

Every interaction with the stream factors into one of four roles, determined by which weight matrix is acting:

| Role | Mechanism | Function |
|---|---|---|
| **Read for routing** | $W_Q^{(\ell, i)}, W_K^{(\ell, i)}$ | Decide *what to attend to* based on the current stream content |
| **Read for content** | $W_V^{(\ell, i)}$ (attention), $W_1^{(\ell)}$ (FFN) | Extract information from the stream to compute the contribution |
| **Write** | $W_O^{(\ell, i)}$ (attention), $W_2^{(\ell)}$ (FFN) | Project the computed contribution back into the stream |
| **Read for prediction** | $W_U$ (unembedding) | Project the final stream into vocabulary space |

The QK matrices and the unembedding are pure **readers**. The $W_O$ projection in attention and the $W_2$ projection in FFN are the **writers**. The OV circuit ($W_V W_O$) and the FFN's read–write composition are the two pathways by which content actually moves into the stream.

### 8.6 Consequences of the residual-stream reframing

Three substantial consequences:

1. **Depth is a budget, not a transformation chain.** There are $2L$ "writes" available in a depth-$L$ model. The model's job is to allocate these writes to perform useful computation, with each head able to specialize in writing one kind of information at one layer.

2. **Linearity dominates the stream.** Because the stream is built by addition, and LN is approximately linear in its operating regime, much of what flows through the stream is *linearly decomposable*. One can talk about "the direction encoding indirect-object identity" or "the subspace carrying syntactic gender" as if these were independent additive contributions — and this is approximately true. The nonlinearities (softmax, GeLU) live inside sub-layers, not on the stream itself.

3. **The architecture is communication-bound.** The bottleneck is not depth or width per se, but how many *useful directions* in $\mathbb{R}^d$ the model can allocate to distinct concepts. This is why $d$ scales aggressively with model size (GPT-2 small: 768; GPT-3: 12{,}288; frontier models: $\sim 16{,}000$).

### 8.7 A subtle implementation point

The **residual stream is not literally a tensor stored anywhere**. It is a conceptual aggregation. In actual code, the variable `x` (or `hidden_states`) that gets passed from sub-layer to sub-layer *is* the residual stream at each layer. There is no separate buffer accumulating contributions; the stream just *is* the running variable that each sub-layer updates.

The reframing is in *how the code is read*, not in what it does. The same forward pass that a 2017-era reader would describe as "applying twelve transformer blocks" is now described as "twenty-four sub-layer contributions accumulated into a $T \times d$ residual stream." Both descriptions compute the same thing; the second is far more useful for asking what the network is actually doing.

---

## 9. The unembedding matrix $W_U$

The residual stream is read out into vocabulary space by a single matrix at the very top of the network. This section examines its structure, geometry, and consequences.

### 9.1 Definition and shape

The unembedding matrix is

$$W_U \in \mathbb{R}^{d \times V}$$

where $V$ is the vocabulary size. Its job is to convert a hidden-state vector into a vector of logits over the vocabulary. The final transformation in a forward pass is:

$$\text{logits} = H^{(L)}  W_U \in \mathbb{R}^{T \times V}$$

For one token position:

$$\text{logits}_t = h_t^{(L)}  W_U \in \mathbb{R}^{V}$$

followed by a softmax:

$$P(x_{t+1} \mid x_{\leq t}) = \mathrm{softmax}(\text{logits}_t)$$

In pre-LN models there is a final LayerNorm before $W_U$: $\text{logits} = \mathrm{LN}_{\text{final}}(H^{(L)})W_U$. That is the entire readout: one matrix multiplication, possibly preceded by an LN, with no nonlinearity and (usually) no bias.

### 9.2 Parameter scale

$W_U$ is one of the **two largest matrices in any modern transformer** (the other being the embedding $W_E$).

| Model | $d$ | $V$ | $W_U$ params |
|---|---|---|---|
| GPT-2 small | 768 | 50,257 | ~38.6M |
| GPT-2 large | 1,280 | 50,257 | ~64.3M |
| Llama 3 8B | 4,096 | 128,256 | ~525M |
| Llama 3 70B | 8,192 | 128,256 | ~1.05B |

For GPT-2 small the 38.6M parameters in $W_U$ are about 30% of the model's total $\sim 124$M parameters.

### 9.3 Weight tying: $W_U$ and $W_E$ as one matrix

The **embedding matrix** $W_E \in \mathbb{R}^{V \times d}$ maps a one-hot token vector to a $d$-dimensional vector at the input. It has the same shape (up to transpose) as $W_U$. In many models — including GPT-2, GPT-3, and Pythia — the weights are **tied**: $W_U = W_E^\top$. The same matrix is used at the bottom (to embed tokens) and at the top (to unembed). This is *weight tying* (Press & Wolf, 2017).

Motivation:

1. **Parameter savings.** Without tying, both matrices cost $2Vd$ parameters; with tying, $Vd$. For GPT-2 small, that is the difference between $\sim$77M and $\sim$39M parameters.
2. **Conceptual symmetry.** The embedding maps token → direction; the unembedding finds the closest token direction. Tying enforces a single per-token representation shared between input encoding and output decoding.

The cost: tying constrains the model. The embedding's job (write a token-identity signal into the stream) and the unembedding's job (read out a next-token prediction from a heavily-processed stream) are genuinely different tasks. **Modern frontier models increasingly untie**: Llama 3, Gemma, and most ~2023+ models use separate $W_U$ and $W_E$. The parameter cost is judged worth the added flexibility.

For dynamics work on GPT-2 and Pythia: weights are tied exactly. This affects the geometric interpretation in §9.4.

### 9.4 Geometric interpretation: dot products with token directions

Write the columns of $W_U$ as $u_1, u_2, \dots, u_V \in \mathbb{R}^d$, one per vocabulary token. Then:

$$\text{logits}_t[v] = h_t^{(L)} \cdot u_v$$

The logit for token $v$ is the **dot product** between the final hidden state and token $v$'s column in $W_U$. The argmax prediction is

$$\mathrm{argmax}_v  \text{logits}_t[v] = \mathrm{argmax}_v  h_t^{(L)} \cdot u_v$$

— the token whose unembedding direction $u_v$ is most aligned with the residual stream at depth $L$. This is **nearest-direction lookup** in $\mathbb{R}^d$, with $V$ candidates.

Under weight tying, $u_v$ is literally token $v$'s embedding. The residual stream, in this view, is a vector that *points toward the next token's embedding*, and the transformer's job during the $L$ blocks of computation is to rotate the stream to point in the right direction.

This geometry is the basis of the **logit lens** (nostalgebraist, 2020), which applies $W_U$ to *intermediate* hidden states $h_t^{(\ell)}$ for $\ell < L$ to see what the model "would predict" if it stopped at layer $\ell$. It works because $W_U$ is a fixed linear readout defined independently of which layer feeds it. Predictions typically get sharper and more accurate as $\ell$ grows — direct evidence that the residual stream is *progressively refined toward the answer* rather than being transformed arbitrarily.

### 9.5 $W_U$ is not an isometry

Since $W_U$ maps $\mathbb{R}^d \to \mathbb{R}^V$ with $V \gg d$, its image is a $d$-dimensional subspace of $\mathbb{R}^V$ (assuming full column rank, which holds in practice). Two consequences:

1. **Many directions in logit space are unreachable.** The model can only produce logits of the form $hW_U$ for some $h \in \mathbb{R}^d$. The reachable logits form a $d$-dimensional subspace of $V$-dimensional logit space.
2. **$W_U$ deforms the geometry of residual-stream space.** If two hidden states $h$ and $h'$ differ by something in the **near-kernel** of $W_U$ (a direction with small singular value), they produce nearly identical logits. The unembedding is *not* an isometry — some directions in residual-stream space get amplified, others attenuated.

This non-isometry has direct consequences for analysis: distances in residual-stream space are *not* the same as distances in logit (or probability) space. A small $L^2$ change in $h_t^{(L)}$ along a high-singular-value direction of $W_U$ can produce a large prediction change; the same-sized change along a low-singular-value direction can produce essentially none.

### 9.6 Over-complete frame structure

The $V$ columns of $W_U$ are $V$ vectors in $\mathbb{R}^d$. Since $V \gg d$, they cannot be orthogonal — there is not enough room in $d$ dimensions for $V$ mutually orthogonal vectors. They form an **over-complete frame**: $V$ vectors spanning a $d$-dimensional space, with mandatory inter-vector correlations.

For GPT-2 small, this is 50{,}257 vectors crammed into 768 dimensions. Semantically similar tokens ("cat", "kitten", "feline") tend to have positively-correlated $u_v$'s; dissimilar tokens tend toward smaller (but rarely zero) inner products.

This is one place where the *linear representation hypothesis* enters: if "concepts" are linear directions in $\mathbb{R}^d$, then the unembedding columns provide a natural (overcomplete, but interpretable) basis. Probing work that finds directions encoding truth, sentiment, refusal, etc., implicitly leverages this structure.

### 9.7 What the unembedding does, mechanistically

Putting it together, the unembedding step does three things at once:

1. **Project onto token directions** — compute the inner product between $h_t^{(L)}$ and each $u_v$.
2. **Rank tokens by similarity** — order the vocabulary by alignment with the current stream state.
3. **Convert to probabilities via softmax** — exponentiate and normalize.

The third step is worth pausing on: softmax depends on *differences* of logits (it's translation-invariant), and since $h \cdot u_v$ scales with $\|h\|$, **the temperature of the output distribution is governed by the norm of the residual stream**. Larger $\|h_t^{(L)}\|$ produces sharper (more confident) predictions; smaller $\|h_t^{(L)}\|$ produces flatter ones. This is why the $10\text{–}100\times$ growth of residual-stream norm through depth (§8.3) is significant: it controls the model's output confidence.

### 9.8 How $W_U$ is trained

$W_U$ is trained via cross-entropy on next-token prediction. The gradient has a clean interpretation:

$$\nabla_{W_U} \mathcal{L}_{\text{NTP}} = h_t^{(L)} \otimes \big(P_t - \delta_{x_{t+1}^*}\big)$$

where $P_t$ is the predicted distribution and $\delta_{x_{t+1}^*}$ is the one-hot target. At each training step:

- The column $u_v$ for the **correct** token $v = x_{t+1}^*$ is **pulled toward** the current $h_t^{(L)}$.
- All other columns are **pushed away** by amounts proportional to their predicted probabilities.

Over many steps this produces a contrastive organization of token directions — tokens that occur in similar contexts end up with similar $u_v$. Under weight tying, the same update also reshapes $W_E$, so every backward pass through $W_U$ also alters how tokens are initially embedded.

### 9.9 Summary of $W_U$ properties

| Property | Description |
|---|---|
| Definition | A linear map $\mathbb{R}^d \to \mathbb{R}^V$ from residual-stream space to logits |
| Shape | $d \times V$ |
| Function | Dot products between $h_t^{(L)}$ and per-token "unembedding directions" |
| Location | Top of the network, after the final block and (in pre-LN) the final LayerNorm |
| Weight tying | In GPT-2, Pythia: $W_U = W_E^\top$. In Llama 3, Gemma, modern frontier: typically untied. |
| Geometric picture | $V$ vectors in $\mathbb{R}^d$ forming an over-complete frame |
| Parameter count | $Vd$ — typically the largest single matrix in the model |
| Logit lens connection | The lens applies $W_U$ to intermediate $h_t^{(\ell)}$, not just $h_t^{(L)}$ |
| Isometry | No — the singular spectrum amplifies some directions and attenuates others |

In summary, $W_U$ is the **fixed linear readout** that translates the residual stream's final state into a vocabulary distribution, and its column geometry encodes the model's mapping between internal representations (directions in $\mathbb{R}^d$) and tokens (indices in $V$).

---

## 10. Training dynamics: the autograd view

Sections 1–9 describe the transformer as a *static* computation graph: given weights, how does input flow to output? This section is about how those weights *come to be what they are*. The story is one of PyTorch's autograd engine building a graph on the forward pass, walking it backward to compute gradients through every matrix in every block, and an optimizer nudging every parameter simultaneously so that next-token prediction improves.

The MHA block is central here because it contains four of the largest learnable matrices per layer ($W_Q, W_K, W_V, W_O$) and because its non-local coupling structure (§6) means gradients from a loss at position $t$ flow back through every other position via the attention pattern.

### 10.1 What is learnable in a transformer

Before tracing how learning happens, it helps to inventory *what* is being learned. Every parameter listed below is a `torch.nn.Parameter` with `requires_grad=True`, meaning autograd tracks its contribution to every downstream computation and can compute gradients with respect to it.

| Location | Matrix | Shape | Role |
|---|---|---|---|
| Input embedding | $W_E$ | $V \times d$ | Token id → residual-stream direction |
| Positional encoding | $W_P$ (learned models) | $T_{\max} \times d$ | Position id → position-encoding direction |
| MHA, per block, per head | $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ | $d \times d_k$ each | Read for routing / read for content |
| MHA, per block | $W_O$ | $d \times d$ | Write attention output to residual stream |
| FFN, per block | $W_1, W_2$ | $d \times d_{ff}$, $d_{ff} \times d$ | Read → nonlinearity → write |
| LayerNorm, per sub-layer | $\gamma, \beta$ | $d$ each | Learnable scale and shift |
| Output projection | $W_U$ | $d \times V$ | Residual stream → logits (often tied to $W_E^\top$) |

A GPT-2-small forward pass touches roughly 124M such scalars. **None of the intermediate tensors** — $Q^{(i)}, K^{(i)}, V^{(i)}, S^{(i)}, A^{(i)}, O^{(i)}, H^{(\ell)}$ — are parameters. They are *activations*: values computed from the inputs and the parameters, held in memory only as long as autograd needs them for the backward pass, then freed.

### 10.2 Two meanings of "embedding" — a crucial distinction

The word "embedding" is used in the literature for two very different objects, and confusion between them is the single most common source of misunderstanding about how training works.

**Static embeddings (parameters).** The rows of $W_E \in \mathbb{R}^{V \times d}$. There are exactly $V$ of them (one per vocabulary token). They are *look-up entries*: given a token id $x$, the embedding is `W_E[x]`. These are learnable parameters — during training their values change.

**Dynamic embeddings (activations, a.k.a. hidden states).** The per-token vectors $h_t^{(\ell)} \in \mathbb{R}^d$ that flow through the residual stream at each layer. These are *not* stored parameters. They are computed fresh on every forward pass from the current values of *all* the weight matrices ($W_E$, all block matrices, LayerNorm parameters). They live in memory only during the forward+backward pass and are then discarded.

The final hidden state $h_t^{(L)}$ — the vector fed to $W_U$ to produce logits — is a dynamic embedding. So is every intermediate $h_t^{(\ell)}$ for $\ell = 1, \dots, L - 1$. Even $h_t^{(0)}$, the input to the first block, is a dynamic embedding (it equals $W_E[x_t] + W_P[t]$, i.e., a *sum* of one static-embedding lookup and one positional-encoding lookup).

**Why this matters for training.** When we say "the embeddings update at each step", we can mean either:

1. **The static lookup table $W_E$ updates.** Yes — its gradient is computed and the optimizer moves it, so the next time the same token id is looked up, its embedding is slightly different.
2. **The dynamic hidden states $h_t^{(\ell)}$ change between training steps.** Yes — because they are recomputed on every forward pass, and every parameter that contributes to them (every $W_Q$, $W_K$, $W_V$, $W_O$, LayerNorm, etc.) has just been nudged. The same input sentence produces different $h_t^{(\ell)}$ this step vs. the previous step.

Both interpretations are correct and both happen simultaneously. Downstream contrastive-training discussions — including the InfoNCE story in the sibling bi-encoder document — usually refer to the *dynamic* embeddings ($\phi_q(q), \phi_d(d)$ are outputs of the whole transformer, not rows of $W_E$). The comment "embeddings are still random and all similarities cluster near zero" refers to the *dynamic* embeddings at initialization, before the accumulated updates to the full weight stack have taught the model to produce semantically meaningful $h_t^{(L)}$.

### 10.3 The autograd graph built by the forward pass

Every PyTorch operation with at least one `requires_grad=True` input records itself in a **directed acyclic graph** (DAG) whose nodes are tensors and whose edges are backward functions. Running the forward pass therefore does two things at once: it computes the loss, and it builds the graph that will be used to compute gradients.

For a single MHA layer, the relevant portion of the graph looks like:

```mermaid
flowchart TD
    H["H (activation)"] --> MMQ["matmul with W_Q<br>(parameter)"]
    H --> MMK["matmul with W_K<br>(parameter)"]
    H --> MMV["matmul with W_V<br>(parameter)"]
    MMQ --> Q["Q"]
    MMK --> K["K"]
    MMV --> V["V"]
    Q --> SC["scaled dot product"]
    K --> SC
    SC --> Sc["scores S"]
    Sc --> SM["softmax"]
    SM --> A["attention A"]
    A --> AV["matmul with V"]
    V --> AV
    AV --> Oh["per-head output O_i"]
    Oh --> CAT["concat + matmul with W_O"]
    CAT --> MHAout["MHA(H)"]
```

Nodes drawn as *parameters* ($W_Q, W_K, W_V, W_O$ and the LayerNorm $\gamma, \beta$ not shown) are leaves of the DAG — gradients accumulate into their `.grad` attribute during the backward pass. Nodes drawn as *activations* ($Q, K, V, S, A, O_i, \mathrm{MHA}(H)$) are internal nodes — they carry a gradient during backward but they don't persist. If `retain_graph=False` (the default), all internal tensors and their backward-function pointers are freed as soon as `loss.backward()` completes.

The full graph for a transformer is this MHA sub-graph, repeated $L$ times, sandwiched with FFN sub-graphs of a similar shape, terminated by $W_U$ and the softmax-cross-entropy loss. For a 12-layer 12-head model, a single forward pass builds a graph with hundreds of thousands of nodes.

### 10.4 The backward pass through MHA

Given the loss $\mathcal{L}$, `.backward()` walks the DAG in reverse-topological order, applying the chain rule at each node. Every parameter's gradient tensor has the same shape as the parameter itself.

For MHA, the backward pass produces:

$$\frac{\partial \mathcal{L}}{\partial W_Q^{(i)}}, \quad \frac{\partial \mathcal{L}}{\partial W_K^{(i)}}, \quad \frac{\partial \mathcal{L}}{\partial W_V^{(i)}} \in \mathbb{R}^{d \times d_k}, \qquad \frac{\partial \mathcal{L}}{\partial W_O} \in \mathbb{R}^{d \times d}$$

These are obtained by chaining seven intermediate Jacobians backward through the seven stages of §3. The gradient flow, from loss to $W_Q$, looks like:

$$
\frac{\partial \mathcal{L}}{\partial W_Q^{(i)}}
= \underbrace{H^{\top}}_{\text{from } Q = H W_Q}
\cdot \frac{\partial \mathcal{L}}{\partial Q^{(i)}}
$$

and the tricky term $\partial \mathcal{L} / \partial Q^{(i)}$ itself decomposes via the softmax Jacobian (which couples every $S_{t, u}^{(i)}$ to every $A_{t, u'}^{(i)}$ in the same row) and the value-weighting step. A full symbolic derivation is straightforward but bulky; the important structural facts are:

1. **Every gradient into an MHA parameter is a $d \times d$ (or $d \times d_k$) tensor** — the same shape as the parameter.
2. **The softmax Jacobian is not diagonal.** A perturbation of one score $S_{t, u}^{(i)}$ changes *every* attention weight $A_{t, u'}^{(i)}$ in the same row via the softmax's coupling. Consequently, the gradient of $\mathcal{L}$ with respect to $S_{t, u}^{(i)}$ depends on the *entire row* of $A$, not just its own entry.
3. **Attention couples positions during backward too.** Because $S^{(i)}_{t, t'}$ mixes token $t$ and token $t'$, the gradient into $h_{t'}$ depends on losses arising at token $t$. This is what allows a next-token prediction error at position $T$ to update representations at position $1$ — the mechanism by which the model learns long-range dependencies.
4. **The causal mask carries into the backward pass.** Because masked scores are set to $-\infty$ before the softmax, their $A_{t, t'}$ is exactly zero, and their gradient is exactly zero. No gradient signal from the loss at token $t$ can reach any key/value at position $t' > t$ — the same property that makes causal masking work forward makes it work backward.

The output projection $W_O$ gets a simpler gradient:

$$\frac{\partial \mathcal{L}}{\partial W_O} = \mathrm{Concat}(O^{(1)}, \dots, O^{(h)})^{\top} \cdot \frac{\partial \mathcal{L}}{\partial \mathrm{MHA}(H)}$$

Because $W_O$ sits at the end of the sub-layer, its gradient sees only the upstream gradient (from the residual and LayerNorm above) and the concatenated per-head outputs — no softmax coupling.

### 10.5 The optimizer step

Once gradients populate every `.grad` attribute, the **optimizer** applies them. AdamW (the standard choice for transformer training) maintains two exponential moving averages per parameter — first-moment $m$ and second-moment $v$ — and updates each parameter $\theta$ via:

$$m \leftarrow \beta_1 m + (1 - \beta_1) g$$

$$v \leftarrow \beta_2 v + (1 - \beta_2) g^{2}$$

$$\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} - \eta \lambda \theta$$

where $g = \partial \mathcal{L} / \partial \theta$, $\hat{m}$ and $\hat{v}$ are bias-corrected moments, $\eta$ is the learning rate, and $\lambda$ is the weight-decay coefficient. Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\eta \in [10^{-4}, 10^{-3}]$, $\lambda \in [0.01, 0.1]$.

Because AdamW is per-parameter, *every* scalar of every MHA matrix gets its own $m$ and $v$ moments. For GPT-2 small this is $\approx 3 \times 124\text{M} = 372\text{M}$ scalars of optimizer state (weights + $m$ + $v$), all in GPU memory — the reason optimizer state is often the largest single memory user during training.

### 10.6 One training iteration end-to-end

Putting the pieces together, a single training step looks like this in code:

```python
# ---- Forward pass: builds the autograd graph -----------------------------
outputs = model(input_ids)                        # (B, T, V) logits
loss = F.cross_entropy(
    outputs[:, :-1, :].reshape(-1, V),            # predictions for x_2, ..., x_T
    input_ids[:, 1:].reshape(-1),                 # targets: shifted-by-one input
)

# ---- Backward pass: gradients populate .grad on every parameter ----------
optimizer.zero_grad(set_to_none=True)             # clear any leftover .grad
loss.backward()                                   # traverses the entire DAG

# Every W_Q, W_K, W_V, W_O in every layer now has a .grad tensor of the same
# shape as itself. So do W_E, W_P, every LayerNorm gamma/beta, and W_U.

# ---- Optimizer step: update all parameters -------------------------------
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # stability
optimizer.step()                                                    # AdamW update
scheduler.step()                                                    # LR schedule
```

After `optimizer.step()`, every parameter in the model has moved a small step in the direction that (locally) reduces $\mathcal{L}$. On the *next* forward pass with the *same* input, every activation is different — because every weight feeding into every activation is different. The dynamic embeddings $h_t^{(\ell)}$ from step $n+1$ are not the same tensors as $h_t^{(\ell)}$ from step $n$, even for the same input tokens.

The training loop repeats this for millions of steps.

### 10.7 The random-initialization trajectory

At step 0, the parameters have their initial values:

- $W_E$, $W_P$: sampled from $\mathcal{N}(0, \sigma^2)$ with $\sigma \approx 0.02$ (GPT-2 convention).
- $W_Q, W_K, W_V, W_O$: sampled from $\mathcal{N}(0, \sigma^2)$ or Xavier/Kaiming, with $\sigma$ scaled by $1/\sqrt{d}$ or $1/\sqrt{L}$ depending on the initialization scheme.
- $\gamma$: initialized to 1; $\beta$: initialized to 0 (so LayerNorm starts as identity plus $\epsilon$ noise).

At this point:

- Every $q_t^{(i)}, k_t^{(i)}, v_t^{(i)}$ is a random $d_k$-dim vector.
- Every score $S_{t, t'}^{(i)} = q_t^{(i)} \cdot k_{t'}^{(i)} / \sqrt{d_k}$ is a mean-zero random variable with variance approximately 1 (by the argument in §3.2).
- Every row of the attention pattern $A^{(i)}$ is close to uniform ($\approx 1/T$ per entry) because the softmax of near-zero scores is nearly flat.
- Every hidden state $h_t^{(\ell)}$ is a random accumulation of random contributions — a random $d$-dim vector.
- **The logits are near-uniform**, and the model predicts every token with probability $\approx 1/V$. The loss starts at approximately $\log V \approx 10.8$ for a 50k vocabulary.

As training proceeds, gradients flow back through the whole stack; every $W$ matrix acquires structure; the attention patterns sharpen from uniform to selective (previous-token, first-token, induction, etc.); the residual stream begins to encode semantically meaningful directions; and the dynamic embeddings $h_t^{(L)}$ start to point toward the correct $u_v$ direction in unembedding space (§9.4). This is the trajectory the InfoNCE-style comment about *"embeddings are still random and all similarities cluster near zero"* describes: at initialization the dynamic embeddings are the outputs of a random transformer, and it takes many gradient steps of updating every matrix in every block before those outputs start to carry meaning.

### 10.8 Practical training concerns

A few implementation realities that are useful to know but tangential to the theory:

- **Gradient checkpointing.** Storing every intermediate activation for backward is expensive ($O(L \cdot T \cdot d)$ memory). Gradient checkpointing trades compute for memory: only every $k$-th layer's activations are stored on forward; the rest are recomputed during backward. This costs ~30% extra forward time but reduces activation memory by roughly a factor of $\sqrt{L}$.

- **Mixed precision (`bfloat16` / `float16`).** Forward and backward run in reduced precision; weights, optimizer state, and gradient accumulation buffers stay in `float32`. This roughly halves activation memory and doubles arithmetic throughput on modern accelerators.

- **Gradient clipping.** Long-range dependencies can produce very large gradients when the softmax attention pattern is near-degenerate. Clipping the global gradient norm to $\le 1.0$ before the optimizer step is standard and cheap insurance.

- **Warmup + cosine decay.** Learning-rate schedules typically ramp $\eta$ linearly from 0 to its peak over the first $\sim 2000$ steps, then decay via cosine. The warmup phase is important because early gradients (when everything is random) are noisy; a large learning rate at that stage can push weights into pathological regimes.

- **Freezing.** In many fine-tuning setups only a subset of parameters have `requires_grad=True`. Everything else has zero-gradient (autograd still walks past it but skips the `.grad` accumulation and the optimizer step). LoRA (§ not covered here) is one popular variant where only small rank-$r$ update matrices are trainable, dramatically reducing optimizer memory.

**The overall picture.** The forward pass computes the loss and, as a side effect, records how the loss depends on every parameter. The backward pass distributes credit — for each scalar of $W_Q, W_K, W_V, W_O$ in every layer, how much did *this scalar* contribute to today's mistake? The optimizer applies a small correction in the direction that would have reduced the mistake. Repeat. The static architecture (§1–9) is what allows all of this to be one differentiable graph; the training loop (§10) is what turns architecture into a working language model.

---

## 11. Summary table

| Stage | Operation | Input shape | Output shape | Parameters |
|---|---|---|---|---|
| 1 | $Q, K, V = HW_Q, HW_K, HW_V$ | $(T, d)$ | $3 \times (T, d)$ | $3d^2$ |
| | Reshape into heads | $(T, d)$ | $(h, T, d_k)$ | — |
| 2 | $S = QK^\top / \sqrt{d_k}$ | $(h, T, d_k)$ | $(h, T, T)$ | — |
| 2.5 | Causal mask (optional) | $(h, T, T)$ | $(h, T, T)$ | — |
| 3 | $A = \mathrm{softmax}_{\text{row}}(S)$ | $(h, T, T)$ | $(h, T, T)$ | — |
| 4 | $O^{(i)} = A^{(i)} V^{(i)}$ | $(h, T, T), (h, T, d_k)$ | $(h, T, d_k)$ | — |
| 5 | Concat heads | $(h, T, d_k)$ | $(T, d)$ | — |
| 6 | Output projection $OW_O$ | $(T, d)$ | $(T, d)$ | $d^2$ |
| 7 | Add & Norm | $(T, d)$ | $(T, d)$ | $2d$ (LN $\gamma, \beta$) |

**Total parameter count for MHA + LN**: $4d^2 + 2d$ per block (ignoring biases). For GPT-2 small ($d = 768$): $\approx 2.36$M parameters in MHA + LN per layer.

---

## Appendix A — Fused multi-head projections in PyTorch

The remark in §3.1 that "the $h$ per-head matrices are stacked side-by-side into single $d \times d$ matrices $W_Q, W_K, W_V$ so that one fused matmul produces all heads at once, and the 'split into heads' becomes a tensor reshape, not a separate computation" hides a small but load-bearing implementation trick. This appendix makes it concrete: it derives the algebraic identity that justifies the fusion, verifies numerical equivalence against a naive per-head implementation, and presents the production `nn.Module` idioms used by nanoGPT, HuggingFace `transformers`, and every serious modern transformer library.

### A.1 Recap: why this fusion works

Recall the per-head projection from §3.1 for head $i \in \{1, \ldots, h\}$:

$$
Q^{(i)} = H W_Q^{(i)}, \quad K^{(i)} = H W_K^{(i)}, \quad V^{(i)} = H W_V^{(i)},
$$

with $H \in \mathbb{R}^{T \times d}$, each $W_{\bullet}^{(i)} \in \mathbb{R}^{d \times d_k}$, and $d = h \cdot d_k$.

Stack the $h$ per-head query matrices side-by-side along the output dimension:

$$
W_Q = [W_Q^{(1)} \mid W_Q^{(2)} \mid \cdots \mid W_Q^{(h)}] \in \mathbb{R}^{d \times d}.
$$

Because matrix multiplication distributes across horizontal concatenation of the right operand,

$$
H \cdot [W_Q^{(1)} \mid \cdots \mid W_Q^{(h)}] = [H W_Q^{(1)} \mid \cdots \mid H W_Q^{(h)}] \in \mathbb{R}^{T \times d},
$$

so the fused $(T, d) \cdot (d, d)$ matmul emits every per-head output *side by side* in a single shot. Retrieving each head is then a stride reinterpretation of the last axis — mathematically a no-op, computationally free.

### A.2 Equivalence demonstration — naive per-head vs. fused matmul

The following self-contained script constructs $h$ independent per-head projection matrices, applies them the naive way, and then stacks them into a fused $(d, d)$ matrix that reproduces the same result via one matmul plus a view.

```python
import torch

torch.manual_seed(0)

B, T = 2, 5           # batch, sequence length
d     = 32            # model hidden size
h     = 4             # number of heads
d_k   = d // h        # per-head width (d must be divisible by h)

H = torch.randn(B, T, d)

W_Q_per_head = [torch.randn(d, d_k) for _ in range(h)]

Q_naive = torch.stack(
    [H @ W_Q_per_head[i] for i in range(h)],
    dim=1,
)

W_Q_fused = torch.cat(W_Q_per_head, dim=1)

Q_fused = H @ W_Q_fused
Q_fused = Q_fused.view(B, T, h, d_k)
Q_fused = Q_fused.transpose(1, 2)

assert torch.allclose(Q_naive, Q_fused, atol=1e-6)
```

Neither `.view()` nor `.transpose()` performs any floating-point work: `.view` reinterprets which stride corresponds to which logical axis, and `.transpose` swaps two entries of the strides tuple. All arithmetic is inside the single fused matmul.

### A.3 A production nn.Module

In practice, the three projections $W_Q$, $W_K$, $W_V$ are wrapped as ordinary `nn.Linear(d, d)` layers, and the per-head split happens inside the module's `forward`:

```python
import torch
import torch.nn as nn


class MultiHeadProjection(nn.Module):
    """Fused Q, K, V projections for multi-head attention.

    Stores each of W_Q, W_K, W_V as a single (d, d) matrix (nn.Linear),
    where the per-head sub-matrices sit side-by-side along the output
    dimension. The 'split into heads' is a reshape + transpose, not a matmul.
    """

    def __init__(self, d: int, h: int):
        super().__init__()
        assert d % h == 0, "d must be divisible by h"
        self.d   = d
        self.h   = h
        self.d_k = d // h

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)

    def forward(self, H: torch.Tensor):
        """
        H : (B, T, d) hidden states
        returns Q, K, V each of shape (B, h, T, d_k)
        """
        B, T, _ = H.shape

        Q = self.W_Q(H)
        K = self.W_K(H)
        V = self.W_V(H)

        Q = Q.view(B, T, self.h, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.h, self.d_k).transpose(1, 2)

        return Q, K, V


mha_proj = MultiHeadProjection(d=32, h=4)
Q, K, V  = mha_proj(torch.randn(2, 5, 32))
assert Q.shape == (2, 4, 5, 8)
```

Two implementation subtleties worth flagging:

- `.view(B, T, h, d_k)` requires contiguous strides along the last axis. That is always satisfied here because `nn.Linear` returns a contiguous tensor.
- The subsequent `.transpose(1, 2)` produces a **non-contiguous** view. This is fine for the downstream $Q K^\top$ matmul (PyTorch's `matmul` handles arbitrary strides), but some fused attention kernels — FlashAttention-style CUDA kernels, in particular — demand contiguous inputs. Add `.contiguous()` in that case.

### A.4 One step further — the fused QKV matmul

The same "stack side-by-side" idea can be pushed one more level: concatenate $W_Q$, $W_K$, and $W_V$ themselves along the output dimension into a single $(d, 3d)$ matrix. One matmul now emits every head's $Q$, $K$, and $V$ simultaneously:

```python
class FusedQKVProjection(nn.Module):
    """Single (d, 3*d) matmul emitting Q, K, V for all heads at once.

    This is what nanoGPT and HuggingFace's GPT-2 c_attn use. Compared to
    MultiHeadProjection above, it replaces three (d, d) matmuls with one
    (d, 3d) matmul: same total FLOPs, higher arithmetic intensity, better
    hardware utilisation.
    """

    def __init__(self, d: int, h: int):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.d_k = d, h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)

    def forward(self, H: torch.Tensor):
        B, T, _ = H.shape
        qkv = self.qkv(H)

        qkv = qkv.view(B, T, 3, self.h, self.d_k)
        Q, K, V = qkv.unbind(dim=2)

        return Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)


fused    = FusedQKVProjection(d=32, h=4)
Q, K, V  = fused(torch.randn(2, 5, 32))
assert Q.shape == (2, 4, 5, 8)
```

The `qkv.view(B, T, 3, h, d_k)` compactly expresses the observation from §3.1: the split into Q/K/V *and* into heads is a single reshape, with the fused matmul carrying all the arithmetic. This is the pattern actually deployed in every modern high-throughput transformer implementation.

### A.5 Cost summary

| Step | What runs | FLOPs | Wall-clock |
|---|---|---|---|
| `nn.Linear(d, d)(H)` — one projection | Dense matmul $(BT, d) \cdot (d, d)$ | $O(B T d^2)$ | Dominant |
| `nn.Linear(d, 3d)(H)` — fused QKV | Dense matmul $(BT, d) \cdot (d, 3d)$ | $O(3 B T d^2)$ | Faster than three separate $(d, d)$ matmuls despite the same total FLOPs, because arithmetic intensity is higher |
| `.view(B, T, h, d_k)` | Stride metadata update | 0 | $\approx 0$ |
| `.transpose(1, 2)` | Two entries of the strides tuple swapped | 0 | $\approx 0$ |
| `.unbind(dim=2)` on a $(B, T, 3, h, d_k)$ tensor | Three views over the same storage | 0 | $\approx 0$ |

The "split into heads" (and the "split into Q/K/V" in the fused-QKV variant) costs **zero FLOPs and zero data motion** — it is a strictly zero-cost reinterpretation of the same $(B, T, d)$ or $(B, T, 3d)$ buffer. All actual arithmetic lives in the fused matmul at the top of the table, which is exactly the observation §3.1 makes.

For the backward pass through these fused matmuls — how PyTorch autograd routes the gradient $\bar{Q}, \bar{K}, \bar{V}$ back through the transpose, through the view, and into the fused $W_Q$ / QKV weight matrices — see [PyTorch Autograd Engine Deep Dive §6 (linear-layer VJP) and §8 (MHA gradient flow)](./PyTorch_Autograd_Engine_Deep_Dive.md#6-worked-example--gradient-through-a-linear-layer).

---

## 12. References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. arXiv:1706.03762.
2. Elhage et al., *A Mathematical Framework for Transformer Circuits*, Anthropic, 2021. transformer-circuits.pub/2021/framework.
3. Olsson et al., *In-context Learning and Induction Heads*, Anthropic, 2022.
4. Ba, Kiros, Hinton, *Layer Normalization*, 2016. arXiv:1607.06450.
5. Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020 (pre-LN vs. post-LN analysis).
6. Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017 (weight tying).
7. nostalgebraist, *interpreting GPT: the logit lens*, LessWrong, 2020.
8. Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, ICLR 2023.
9. Huang, LeCun, Balestriero, *Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA*, arXiv:2602.22617, 2026.
10. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019 (autograd engine).
11. Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019 (AdamW).
12. Chen et al., *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174, 2016 (gradient checkpointing).
13. Micikevicius et al., *Mixed Precision Training*, ICLR 2018.
14. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
