# Encoder-Decoder with Scaled Dot-Product Attention — Architecture Guide

This document provides detailed UML class diagrams, sequence diagrams, and flowcharts for the toy encoder-decoder architecture implemented in the companion notebook (`encoder_decoder_with_attention_demo.ipynb`).

---

## Table of Contents

1. [High-Level Architecture Overview](#1--high-level-architecture-overview)
2. [Static Class Diagram](#2--static-class-diagram)
3. [Component Deep Dives](#3--component-deep-dives)
   - [Encoder](#31--encoder)
   - [Attention](#32--attention-module)
   - [Decoder](#33--decoder)
   - [Seq2SeqWithAttention](#34--seq2seqwithattention-orchestrator)
4. [Sequence Diagrams](#4--sequence-diagrams)
   - [Training Forward Pass](#41--training-forward-pass-single-batch)
   - [Greedy Decoding (Inference)](#42--greedy-decoding-inference)
5. [Flowcharts](#5--flowcharts)
   - [Scaled Dot-Product Attention](#51--scaled-dot-product-attention-computation)
   - [Single Decoder Step](#52--single-decoder-time-step)
   - [Training Loop](#53--training-loop)
   - [Dataset Generation](#54--dataset-generation)
6. [Tensor Shape Reference](#6--tensor-shape-reference)

---

## 1 — High-Level Architecture Overview

The system is a sequence-to-sequence model that reads a source sequence through an **Encoder** (GRU-based), computes **Attention** between the decoder state and all encoder hidden states using the scaled dot-product formula, and generates the output sequence one token at a time through a **Decoder** (also GRU-based).

```mermaid
graph LR
    subgraph Input
        X["Source tokens<br/>(B, src_len)"]
    end

    subgraph Encoder
        EMB_E["Embedding<br/>vocab → embed_dim"]
        GRU_E["GRU<br/>embed_dim → hidden_dim"]
    end

    subgraph Attention
        WQ["W_Q projection"]
        WK["W_K projection"]
        SCORE["QKᵀ / √d_k"]
        SOFT["softmax"]
        CTX["weights · V"]
    end

    subgraph Decoder
        EMB_D["Embedding<br/>vocab → embed_dim"]
        CAT["concat(embed, context)"]
        GRU_D["GRU<br/>(embed+hidden) → hidden"]
        FC["Linear<br/>hidden → vocab"]
    end

    X --> EMB_E --> GRU_E
    GRU_E -- "H (all hidden states)" --> WK
    GRU_E -- "H (values)" --> CTX
    GRU_E -- "h_last (init decoder)" --> GRU_D

    GRU_D -- "s_i (decoder state)" --> WQ
    WQ --> SCORE
    WK --> SCORE
    SCORE --> SOFT
    SOFT -- "weights" --> CTX
    CTX -- "context c" --> CAT

    EMB_D --> CAT
    CAT --> GRU_D
    GRU_D --> FC
    FC --> OUTPUT["Logits<br/>(B, vocab_size)"]
```

---

## 2 — Static Class Diagram

```mermaid
classDiagram
    class Encoder {
        -embedding: nn.Embedding
        -rnn: nn.GRU
        +__init__(vocab_size: int, embed_dim: int, hidden_dim: int)
        +forward(src: Tensor) Tuple~Tensor, Tensor~
    }
    note for Encoder "Embedding: (vocab_size, embed_dim)\nGRU: input=embed_dim, hidden=hidden_dim\n\nforward() returns:\n  outputs: (B, src_len, hidden_dim) — all H\n  hidden: (1, B, hidden_dim) — last state"

    class Attention {
        -d_k: int
        -W_Q: nn.Linear
        -W_K: nn.Linear
        +__init__(hidden_dim: int, d_k: int|None)
        +forward(decoder_state: Tensor, encoder_outputs: Tensor) Tuple~Tensor, Tensor~
    }
    note for Attention "W_Q: (hidden_dim → d_k) no bias\nW_K: (hidden_dim → d_k) no bias\nV = encoder_outputs (no projection)\n\nImplements:\n  softmax(Q·Kᵀ / √d_k) · V"

    class Decoder {
        -embedding: nn.Embedding
        -attention: Attention
        -rnn: nn.GRU
        -fc_out: nn.Linear
        +__init__(vocab_size: int, embed_dim: int, hidden_dim: int)
        +forward_step(token: Tensor, hidden: Tensor, encoder_outputs: Tensor) Tuple~Tensor, Tensor, Tensor~
    }
    note for Decoder "Embedding: (vocab_size, embed_dim)\nGRU: input=(embed_dim + hidden_dim)\n     hidden=hidden_dim\nfc_out: (hidden_dim → vocab_size)\n\nforward_step() decodes ONE time step\nreturns: (logits, hidden, attn_weights)"

    class Seq2SeqWithAttention {
        -encoder: Encoder
        -decoder: Decoder
        -tgt_sos_idx: int
        +__init__(encoder: Encoder, decoder: Decoder, tgt_sos_idx: int)
        +forward(src: Tensor, tgt: Tensor, teacher_forcing_ratio: float) Tuple~Tensor, Tensor~
    }
    note for Seq2SeqWithAttention "Orchestrates the full forward pass:\n1. Encode src → H, h_last\n2. Loop over tgt_len steps:\n   call decoder.forward_step()\n   apply teacher forcing\n3. Return all logits + attention weights"

    Seq2SeqWithAttention *-- Encoder : encoder
    Seq2SeqWithAttention *-- Decoder : decoder
    Decoder *-- Attention : attention
    Encoder ..|> nn_Module : extends
    Decoder ..|> nn_Module : extends
    Attention ..|> nn_Module : extends
    Seq2SeqWithAttention ..|> nn_Module : extends

    class nn_Module {
        <<abstract>>
        +parameters()
        +train()
        +eval()
    }
```

---

## 3 — Component Deep Dives

### 3.1 — Encoder

The Encoder converts a sequence of discrete token IDs into a sequence of continuous hidden-state vectors. It has two sub-layers:

| Sub-layer | Type | Input shape | Output shape | Role |
|-----------|------|-------------|--------------|------|
| `embedding` | `nn.Embedding(13, 32)` | `(B, src_len)` | `(B, src_len, 32)` | Maps each token ID to a learned 32-dim vector |
| `rnn` | `nn.GRU(32, 64)` | `(B, src_len, 32)` | outputs: `(B, src_len, 64)`, hidden: `(1, B, 64)` | Reads the embedded sequence left-to-right, producing a 64-dim hidden state at every position |

**Key outputs:**

- **`outputs` (H)** — the matrix of *all* hidden states across source positions. This becomes the **Key** and **Value** supply for the attention mechanism. Each column is a contextualized representation of one source token.
- **`hidden` (h_last)** — the final hidden state. Passed to the Decoder as its initial hidden state, giving it a compressed summary of the entire source.

```mermaid
flowchart LR
    subgraph Encoder
        A["Token IDs\n(B, src_len)"] --> B["nn.Embedding\n13 → 32"]
        B --> C["Embedded\n(B, src_len, 32)"]
        C --> D["nn.GRU\n32 → 64"]
        D --> E["H: all hidden states\n(B, src_len, 64)"]
        D --> F["h_last\n(1, B, 64)"]
    end

    E -. "to Attention as K and V" .-> G((Attention))
    F -. "init Decoder hidden state" .-> H((Decoder))
```

### 3.2 — Attention Module

The Attention module implements the scaled dot-product attention formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

It sits **between** the Encoder and Decoder. At every decoder time step it receives the current decoder state and all encoder outputs, and produces a context vector — a weighted blend of encoder hidden states, where the weights indicate *which source positions the decoder should focus on*.

| Sub-layer | Type | Input → Output | Role |
|-----------|------|----------------|------|
| `W_Q` | `nn.Linear(64, 64, bias=False)` | `(B, 1, 64)` → `(B, 1, 64)` | Projects decoder state into **query** space |
| `W_K` | `nn.Linear(64, 64, bias=False)` | `(B, src_len, 64)` → `(B, src_len, 64)` | Projects encoder hidden states into **key** space |
| *(no W_V)* | — | — | Values are the raw encoder outputs (no projection) |

**Why separate projections?** The decoder hidden state and encoder hidden states are produced by different GRUs — they share the same dimensionality by design but may encode information in incompatible subspaces. The learned `W_Q` and `W_K` matrices rotate them into a shared space where dot products are meaningful.

**Design choice — no value projection:** The values `V` are the unmodified encoder outputs `H`. This keeps the context vector in the same space as the encoder hidden states, which is appropriate because it will be concatenated with the target embedding before being fed to the decoder GRU.

### 3.3 — Decoder

The Decoder generates the output sequence one token at a time. Its `forward_step` method processes a **single** time step:

| Step | Operation | Input | Output |
|------|-----------|-------|--------|
| 1 | `embedding(token)` | `(B, 1)` token ID | `(B, 1, 32)` embedded token |
| 2 | `attention(state, H)` | decoder hidden `(B, 1, 64)` + encoder H `(B, src_len, 64)` | context `(B, 1, 64)` + weights `(B, 1, src_len)` |
| 3 | `cat(embedded, context)` | `(B, 1, 32)` + `(B, 1, 64)` | `(B, 1, 96)` concatenated |
| 4 | `rnn(concat, hidden)` | `(B, 1, 96)` input + `(1, B, 64)` hidden | output `(B, 1, 64)` + new hidden `(1, B, 64)` |
| 5 | `fc_out(output)` | `(B, 64)` | `(B, 13)` logits over vocabulary |

**Why concatenate embedding and context before the GRU?** This gives the GRU simultaneous access to (a) what the previous predicted token was (the embedding) and (b) what part of the source the model should focus on now (the context). The GRU then integrates both signals with its recurrent memory to produce the next hidden state.

```mermaid
flowchart TB
    TOKEN["Input token\n(B, 1)"] --> EMB["nn.Embedding\n13 → 32"]
    EMB --> EMBED["embedded\n(B, 1, 32)"]

    HIDDEN_IN["Previous hidden\n(1, B, 64)"] --> TRANSPOSE["transpose → (B, 1, 64)"]
    TRANSPOSE --> ATTN["Attention module"]
    ENC_H["Encoder H\n(B, src_len, 64)"] --> ATTN

    ATTN --> CTX["context\n(B, 1, 64)"]
    ATTN --> W["weights\n(B, 1, src_len)"]

    EMBED --> CONCAT["torch.cat dim=-1"]
    CTX --> CONCAT
    CONCAT --> RNN_IN["(B, 1, 96)"]
    RNN_IN --> GRU["nn.GRU\n96 → 64"]
    HIDDEN_IN --> GRU

    GRU --> OUT["output\n(B, 1, 64)"]
    GRU --> HIDDEN_OUT["new hidden\n(1, B, 64)"]
    OUT --> SQUEEZE["squeeze → (B, 64)"]
    SQUEEZE --> FC["nn.Linear\n64 → 13"]
    FC --> LOGITS["logits\n(B, 13)"]
```

### 3.4 — Seq2SeqWithAttention Orchestrator

This top-level module wires everything together. It owns the Encoder and Decoder and manages the autoregressive decoding loop with optional teacher forcing.

**Teacher forcing** is a training technique where, with probability `teacher_forcing_ratio`, the ground-truth target token is fed as the next decoder input instead of the model's own prediction. This stabilizes early training (the model doesn't compound its own errors) but is annealed toward 0 for inference.

**What it does on `forward()`:**

1. Encode the full source sequence → `H` and `h_last`.
2. Initialize: first decoder token = `<SOS>`, decoder hidden = `h_last`.
3. For each target position `t`:
   - Call `decoder.forward_step(token, hidden, H)` → logits, new hidden, attention weights.
   - Store logits and weights.
   - Decide next input: ground-truth token (teacher forcing) or argmax of logits (free running).
4. Return the stacked logits `(B, tgt_len, vocab_size)` and attention weight matrix `(B, tgt_len, src_len)`.

---

## 4 — Sequence Diagrams

### 4.1 — Training Forward Pass (single batch)

```mermaid
sequenceDiagram
    participant Main as Training Loop
    participant S2S as Seq2SeqWithAttention
    participant Enc as Encoder
    participant Dec as Decoder
    participant Attn as Attention
    participant Loss as CrossEntropyLoss

    Main->>S2S: forward(src_batch, tgt_batch, tf_ratio=0.5)
    activate S2S

    S2S->>Enc: forward(src_batch)
    activate Enc
    Note over Enc: embedding(src) → (B, src_len, 32)
    Note over Enc: GRU(embedded) → H, h_last
    Enc-->>S2S: H (B, src_len, 64), h_last (1, B, 64)
    deactivate Enc

    Note over S2S: token ← tgt[:, 0:1] (⟨SOS⟩)<br/>hidden ← h_last<br/>init outputs tensor

    loop t = 0 … tgt_len−1
        S2S->>Dec: forward_step(token, hidden, H)
        activate Dec

        Note over Dec: embedded ← embedding(token)<br/>decoder_state ← hidden.T → (B,1,64)

        Dec->>Attn: forward(decoder_state, H)
        activate Attn
        Note over Attn: Q = W_Q(decoder_state) → (B,1,d_k)
        Note over Attn: K = W_K(H) → (B,src_len,d_k)
        Note over Attn: scores = QKᵀ / √d_k
        Note over Attn: weights = softmax(scores)
        Note over Attn: context = weights · V
        Attn-->>Dec: context (B,1,64), weights (B,1,src_len)
        deactivate Attn

        Note over Dec: rnn_input ← cat(embedded, context)<br/>output, hidden ← GRU(rnn_input, hidden)<br/>logits ← fc_out(output)

        Dec-->>S2S: logits (B,13), hidden, weights
        deactivate Dec

        Note over S2S: outputs[:, t] ← logits<br/>coin flip: teacher forcing?
        alt teacher forcing (p=0.5)
            Note over S2S: token ← tgt[:, t+1:t+2]
        else free running
            Note over S2S: token ← argmax(logits)
        end
    end

    S2S-->>Main: outputs (B, tgt_len, 13), all_weights (B, tgt_len, src_len)
    deactivate S2S

    Main->>Loss: CE(outputs[:, :-1], tgt[:, 1:])
    Loss-->>Main: scalar loss
    Note over Main: loss.backward()<br/>clip_grad_norm_(1.0)<br/>optimizer.step()
```

### 4.2 — Greedy Decoding (Inference)

```mermaid
sequenceDiagram
    participant User as greedy_decode()
    participant Enc as Encoder
    participant Dec as Decoder
    participant Attn as Attention

    User->>Enc: forward(src_tensor)
    Enc-->>User: enc_out (H), hidden (h_last)

    Note over User: token ← [[SOS]]<br/>pred_ids ← [SOS]

    loop until EOS or max_len
        User->>Dec: forward_step(token, hidden, enc_out)
        activate Dec

        Dec->>Attn: forward(hidden.T, enc_out)
        Attn-->>Dec: context, weights

        Note over Dec: cat → GRU → fc_out

        Dec-->>User: logits, new hidden, weights
        deactivate Dec

        Note over User: next_id ← argmax(logits)
        Note over User: pred_ids.append(next_id)
        Note over User: collect weights

        alt next_id == EOS
            Note over User: break
        else continue
            Note over User: token ← [[next_id]]
        end
    end

    Note over User: return pred_ids, attention_matrix
```

---

## 5 — Flowcharts

### 5.1 — Scaled Dot-Product Attention Computation

This flowchart traces every tensor operation inside `Attention.forward()`, with shapes annotated at each stage.

```mermaid
flowchart TD
    DS["decoder_state s<br/>(B, 1, hidden_dim)"] --> PQ["Q = W_Q(s)<br/>(B, 1, d_k)"]
    EO["encoder_outputs H<br/>(B, src_len, hidden_dim)"] --> PK["K = W_K(H)<br/>(B, src_len, d_k)"]
    EO --> V["V = H<br/>(B, src_len, hidden_dim)"]

    PQ --> DOT["scores = Q · Kᵀ<br/>(B, 1, src_len)"]
    PK --> DOT

    DOT --> SCALE["scores / √d_k<br/>(B, 1, src_len)"]

    SCALE --> SM["softmax(scores, dim=-1)<br/>(B, 1, src_len)<br/>∑ weights = 1"]

    SM --> MUL["context = weights · V<br/>(B, 1, hidden_dim)"]
    V --> MUL

    MUL --> OUT_C["context vector c<br/>(B, 1, hidden_dim)"]
    SM --> OUT_W["attention weights w<br/>(B, 1, src_len)"]

    style PQ fill:#ffeaa7,stroke:#fdcb6e
    style PK fill:#ffeaa7,stroke:#fdcb6e
    style DOT fill:#fab1a0,stroke:#e17055
    style SCALE fill:#fab1a0,stroke:#e17055
    style SM fill:#74b9ff,stroke:#0984e3
    style MUL fill:#55efc4,stroke:#00b894
```

### 5.2 — Single Decoder Time Step

```mermaid
flowchart TD
    A["Input: token (B,1), hidden (1,B,64), H (B,src_len,64)"]
    A --> B["embedded = embedding(token)<br/>(B, 1, 32)"]
    A --> C["decoder_state = hidden.T<br/>(B, 1, 64)"]

    C --> D{"Attention"}
    A --> D

    D --> E["context (B, 1, 64)"]
    D --> F["weights (B, 1, src_len)"]

    B --> G["rnn_input = cat(embedded, context)<br/>(B, 1, 96)"]
    E --> G

    G --> H["output, hidden_new = GRU(rnn_input, hidden)"]

    H --> I["output (B, 1, 64)"]
    H --> J["hidden_new (1, B, 64)"]

    I --> K["logits = fc_out(output.squeeze)<br/>(B, 13)"]

    K --> L["Return: logits, hidden_new, weights"]
    J --> L
    F --> L
```

### 5.3 — Training Loop

```mermaid
flowchart TD
    START([Start Training]) --> INIT["Initialize:<br/>encoder, decoder, model<br/>optimizer = Adam(lr=3e-3)<br/>criterion = CrossEntropyLoss(ignore_index=PAD)"]

    INIT --> EPOCH{"epoch ≤ 40?"}
    EPOCH -- No --> DONE([Training Complete])
    EPOCH -- Yes --> SHUFFLE["Shuffle training indices<br/>perm = randperm(n_train)"]

    SHUFFLE --> BATCH{"More batches?"}
    BATCH -- No --> EVAL["model.eval()<br/>Compute val_loss<br/>(teacher_forcing=0.0)"]
    BATCH -- Yes --> FWD["Forward pass:<br/>logits, _ = model(src, tgt, tf=0.5)"]

    FWD --> LOSS["loss = CE(logits[:, :-1], tgt[:, 1:])<br/>Shift: predict next token from each position"]

    LOSS --> BACKWARD["optimizer.zero_grad()<br/>loss.backward()<br/>clip_grad_norm_(params, 1.0)<br/>optimizer.step()"]

    BACKWARD --> BATCH

    EVAL --> LOG{"epoch % 5 == 0?"}
    LOG -- Yes --> PRINT["Print train_loss, val_loss"]
    LOG -- No --> NEXT[" "]
    PRINT --> NEXT
    NEXT --> EPOCH
```

### 5.4 — Dataset Generation

The toy task is **digit-sequence reversal**: given a source sequence of random digits, the target is the same digits in reverse order.

```
Source: <s> 3 7 1 5 </s> _     Target: <s> 5 1 7 3 </s> _
```

```mermaid
flowchart TD
    START([make_reversal_dataset]) --> PARAMS["n_samples=3000<br/>min_len=4, max_len=8"]

    PARAMS --> LOOP{"i < n_samples?"}
    LOOP -- No --> PAD_STEP["Pad all sequences to max length<br/>with PAD=0 tokens"]
    LOOP -- Yes --> LEN["length = randint(4, 9)"]
    LEN --> DIGITS["digits = randint(0, 10, size=length)<br/>e.g. [3, 7, 1, 5]"]
    DIGITS --> SRC["src = [SOS] + (digits+3) + [EOS]<br/>Token IDs: [1, 6, 10, 4, 8, 2]"]
    DIGITS --> TGT["tgt = [SOS] + (reversed_digits+3) + [EOS]<br/>Token IDs: [1, 8, 4, 10, 6, 2]"]
    SRC --> APPEND["Append to sources, targets"]
    TGT --> APPEND
    APPEND --> LOOP

    PAD_STEP --> TENSOR["Convert to torch.Tensor<br/>src: (3000, max_src_len)<br/>tgt: (3000, max_tgt_len)"]
    TENSOR --> RETURN([Return src_data, tgt_data])
```

**Token vocabulary (13 tokens total):**

| Token ID | Meaning |
|----------|---------|
| 0 | `PAD` — padding |
| 1 | `SOS` — start of sequence |
| 2 | `EOS` — end of sequence |
| 3–12 | Digits 0–9 (offset by 3) |

---

## 6 — Tensor Shape Reference

End-to-end tensor flow for a single training step with the notebook's hyperparameters (`EMBED_DIM=32`, `HIDDEN_DIM=64`, `VOCAB_SIZE=13`).

```mermaid
flowchart LR
    subgraph Encoder Path
        SRC["src<br/>(B, src_len)"]
        -->|"Embedding"| EMB_E["(B, src_len, 32)"]
        -->|"GRU"| H["H<br/>(B, src_len, 64)"]
    end

    subgraph "Attention (per decoder step)"
        S["s_i<br/>(B, 1, 64)"]
        -->|"W_Q"| Q["Q<br/>(B, 1, 64)"]

        H2["H<br/>(B, src_len, 64)"]
        -->|"W_K"| K["K<br/>(B, src_len, 64)"]

        Q -->|"bmm + scale"| SC["scores<br/>(B, 1, src_len)"]
        K --> SC
        SC -->|"softmax"| W["weights<br/>(B, 1, src_len)"]
        W -->|"bmm with V=H"| CTX["context<br/>(B, 1, 64)"]
    end

    subgraph Decoder Path
        TKN["token<br/>(B, 1)"]
        -->|"Embedding"| EMB_D["(B, 1, 32)"]

        EMB_D -->|"cat"| CONC["(B, 1, 96)"]
        CTX --> CONC
        CONC -->|"GRU"| OUT["output<br/>(B, 1, 64)"]
        OUT -->|"squeeze + Linear"| LOGIT["logits<br/>(B, 13)"]
    end
```

### Quick shape cheat-sheet

| Name | Shape | Notes |
|------|-------|-------|
| `src` | `(B, src_len)` | Integer token IDs, padded |
| `tgt` | `(B, tgt_len)` | Integer token IDs, padded |
| Encoder `embedded` | `(B, src_len, 32)` | After `nn.Embedding` |
| `H` (encoder outputs) | `(B, src_len, 64)` | All GRU hidden states |
| `h_last` | `(1, B, 64)` | Final encoder hidden state |
| `Q` (query) | `(B, 1, 64)` | `W_Q @ decoder_state` |
| `K` (keys) | `(B, src_len, 64)` | `W_K @ H` |
| `V` (values) | `(B, src_len, 64)` | `H` (no projection) |
| `scores` | `(B, 1, src_len)` | `Q · Kᵀ / √64` |
| `weights` | `(B, 1, src_len)` | `softmax(scores)` — sums to 1 |
| `context` | `(B, 1, 64)` | `weights · V` |
| Decoder `rnn_input` | `(B, 1, 96)` | `cat(embedded_token, context)` |
| Decoder `output` | `(B, 1, 64)` | GRU output |
| `logits` | `(B, 13)` | Unnormalized scores over vocabulary |
| `all_weights` | `(B, tgt_len, src_len)` | Full attention matrix for visualization |
