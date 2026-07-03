---
title: "MoCo: Momentum Contrast for Visual Representation Learning"
description: "MoCo frames contrastive learning as a dictionary look-up using a queue and momentum encoder, enabling scalable unsupervised representation learning."
publishDate: 2026-06-30
tags:
  - self-supervised-learning
  - contrastive-learning
  - representation-learning
  - infonce
  - momentum-encoder
  - dynamic-dictionary
  - cvpr2020
series: contrastive-learning
seriesOrder: 5
---

> MoCo frames contrastive learning as a dictionary look-up problem -- a query searches for its matching key in the dictionary while pushing away all other keys. It uses a queue to scale the dictionary to 65,536 entries and a momentum encoder to ensure the features across all keys are highly consistent. This was the first method to let unsupervised pretraining surpass supervised ImageNet pretraining across the board.

---

## 0. Notation

| Symbol | Meaning |
|:---|:---|
| $x$ | Original image |
| $x^{\text{query}}$ | Query image (one augmented view of $x$) |
| $x^{\text{key}}$ | Key image (another augmented view of $x$) |
| $f_q$ | Query encoder, updated normally via gradient descent |
| $f_k$ | Key encoder, updated via momentum (no gradient backpropagation) |
| $q = f_q(x^{\text{query}})$ | Query feature, dimension $C$ |
| $k_+ = f_k(x^{\text{key}})$ | Positive key feature, dimension $C$ |
| $k_i$ | $i$-th key in the dictionary (mostly negatives), $i = 1, \dots, K$ |
| $K$ | Dictionary size (queue length), default 65536 |
| $C$ | Feature dimension, default 128 |
| $N$ | Batch size, default 256 |
| $m$ | Momentum coefficient, default 0.999 |
| $\tau$ | Temperature parameter, default 0.07 |
| $\theta_q$ | Parameters of $f_q$ |
| $\theta_k$ | Parameters of $f_k$ |

> **MoCo's terminology**: In MoCo's dictionary look-up framing, the usual "anchor / positive / negative" nomenclature is replaced by "query / key (positive) / key (negative)." There is only one query, and a whole dictionary of keys.

---

## 1. Motivation: Why Visual Unsupervised Learning Lagged Behind NLP

### 1.1 The landscape in 2019

- **NLP**: BERT and GPT had already shown that large-scale unsupervised pretraining followed by light fine-tuning works extremely well. A key enabler -- text is a **discrete** signal space (words, subwords), which naturally lends itself to building tokenized dictionaries.
- **CV**: Visual signals live in a **continuous, high-dimensional** space. The signal is dense and semantically uncompressed, making it much harder to build a clean dictionary the way NLP does. As a result, unsupervised pretraining in vision consistently lagged behind supervised pretraining.

### 1.2 What contrastive learning needs from a negative dictionary

Contrastive learning (especially the instance discrimination pretext task) boils down to: given a query $q$, make it similar to the positive key $k_+$ and dissimilar to all negative keys $k_i$. This is essentially a **$(K+1)$-way classification problem** -- one positive vs $K$ negatives.

A good negative dictionary needs two properties:

| Property | Requirement | Failure mode |
|:---|:---|:---|
| **Large** | $K$ must be big enough to cover varied visual features | Small $K$ lets the model take shortcuts (e.g. relying on color), leading to poor generalization |
| **Consistent** | All keys should come from the same or very similar encoders | Inconsistent encoders let the model learn "find keys produced by the same encoder" as a shortcut |

**Every method before MoCo was limited by at least one of these**:

| Method | Large dictionary? | Consistent? | Bottleneck |
|:---|:---:|:---:|:---|
| **End-to-end (SimCLR lineage)** | No | Yes | Batch size limits $K = N$, max 8192 (needs TPU) |
| **Memory Bank (InstDisc)** | Yes (1.28M) | No | Updated only once per epoch, features are stale and highly inconsistent |
| **MoCo** | Yes | Yes | Queue decouples $K$ from $N$, momentum encoder guarantees consistency |

---

## 2. Core Framework: Contrastive Learning as Dictionary Look-Up

### 2.1 The dictionary look-up perspective

This is MoCo's most elegant abstraction. Instead of thinking of contrastive learning as "pull positives together, push negatives apart," the paper reframes it as a **dictionary look-up task**:

```
Given a query q (the encoded feature of the current image)
Look up the key k+ that best matches q (the feature from another view of the same image)
All other keys in the dictionary are non-matches
```

The objective -- InfoNCE loss:

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}$$

The numerator is the similarity between query and positive, the denominator is the sum of similarities with **all** keys in the dictionary (1 positive + $K$ negatives).

### 2.2 Framework diagram (paper Figure 1)

```
         +------------------+         +------------------+
         |   Query encoder  |         |   Key encoder     |
         |       f_q        |         |       f_k         |
         |   gradient update|         |   momentum update |
         +--------+---------+         +--------+---------+
                  | q                          | k
                  |                            |
                  v                            v
         +---------------------------------------------+
         |       Contrastive loss (InfoNCE)             |
         |   q-k+ high    q-k_i low (i != +)           |
         +---------------------------------------------+
                                                |
                                                v
                                         +------------+
                                         |   Queue    |
                                         | (dictionary)|
                                         | K = 65536  |
                                         +------------+
                                         enqueue(k) -> ->
                                         <- <- dequeue(oldest)
```

---

## 3. Three Key Design Choices

### 3.1 Queue: Decoupling Dictionary Size from Batch Size -- Solving "Dictionary Must Be Large"

**Problem**: In end-to-end methods (SimCLR), negatives come only from the current batch, so dictionary size $K$ equals batch size $N$. Growing $K$ means growing the batch, which blows up memory and makes optimization harder.

**MoCo's solution**: Maintain a **FIFO queue** as the dictionary:

- Each training step, the current batch of $N$ keys is enqueued
- The oldest $N$ keys are dequeued from the tail
- Queue length $K = 65536$, far larger than batch size $N = 256$

**Key benefit**: Dictionary size and batch size are **fully decoupled**. $K$ can be set arbitrarily large (up to memory limits) without increasing the batch.

**Computational cost of the queue**:

> Most keys in the queue don't need to be re-encoded every iteration -- they are reused from previous batches. Only $N=256$ new keys are encoded per step, while $K-N = 65280$ old keys are reused directly. This makes maintaining a large dictionary computationally cheap.

**A nice side effect**: The queue continuously evicts the oldest keys. From a consistency standpoint, the oldest keys were produced by an encoder from many steps ago and have drifted the most -- evicting them actually improves overall dictionary consistency.

### 3.2 Momentum Encoder: Solving "Dictionary Must Be Consistent"

**This is MoCo's most elegant and impactful contribution.**

#### The problem

If the key encoder $f_k$ uses regular gradient updates like the query encoder $f_q$, it changes quickly. The queue holds 65,536 keys coming from the past 256 batches -- each batch was encoded by a slightly different $f_k$. The result: keys in the dictionary come from 65,536 different encoding spaces that are not comparable. The model degenerates into learning a shortcut: "find keys from a similar encoder."

#### The solution: momentum update

$$ \theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q $$

where $m = 0.999$, very close to 1.

**What this means**:

| $m$ value | Information absorbed from $\theta_q$ per step | $\theta_k$ update speed |
|:---|:---|:---|
| 0.999 | 0.1% (1 per mille) | Extremely slow -- very smooth |
| 0.99 | 1% | Moderate |
| 0.9 | 10% | Fast -- close to end-to-end |

**Intuition**: With $m=0.999$, a key stays in the queue for $65536 / 256 = 256$ steps. Over those 256 steps, $\theta_k$ changes minimally (it takes in only 1 per mille of $\theta_q$ per step). So all 65,536 keys in the queue come from nearly the same encoding space -- dictionary consistency is very high.

**Mathematical intuition (moving average)**: The momentum update is essentially an **exponentially weighted moving average** of $\theta_q$'s history. $m=0.999$ means $\theta_k$ is a smoothed version of $\theta_q$ over roughly the past 1,000 steps. Since $\theta_q$ itself changes slowly (bounded by the learning rate), $\theta_k$ changes even less.

> **Inspiration: DQN's target network**. The original paper's Related Work section explicitly cites the target network from **DQN (Deep Q-Network, Mnih et al., 2015)** as the inspiration.
>
> **Terminology clarification**: DQN specifically refers to the concrete algorithm proposed in the Mnih et al. Nature paper -- using a deep convolutional network to approximate the Q-function, combined with experience replay and a target network, to play Atari games. It is not a catch-all for "the deep Q-Learning family" (which includes many variants like Double DQN, Dueling DQN, Rainbow, etc.), but rather the pioneering instance within that space.
>
> DQN maintains two Q-networks: an online network (normal gradient updates) and a target network (whose parameters are **hard-copied** from online every $C$ steps) to stabilize TD target computation. MoCo upgrades "hard copy" to "soft momentum" -- instead of copying every $C$ steps, it shifts by only 1 per mille ($m=0.999$) at **every step**, replacing a piecewise constant function with an exponential moving average. Smoother, more elegant.

<details>
<summary>Background: From Q-Learning to DQN (a crash course in RL)</summary>

> **If you're not familiar with reinforcement learning, here's a quick explanation of what DQN's target network actually solves:**
>
> **What Q-Learning is**: A model-free RL algorithm that learns the **Q-function**:
>
> $$Q(s, a) = \text{expected cumulative reward after taking action } a \text{ in state } s$$
>
> Core update (Bellman equation):
>
> $$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - Q(s, a) \big]$$
>
> In plain terms: take an action, get a reward $r$, enter a new state $s'$. The estimated best future return is $\max_{a'} Q(s', a')$ -- that's the **TD target**. The gap between it and the current $Q(s,a)$ tells you how to correct.
>
> **Traditional Q-Learning's bottleneck**: It uses a table (Q-table) to store Q-values for all $(s, a)$ pairs. When the state space explodes (e.g., Atari game pixels), the table won't fit.
>
> ---
>
> **Deep Q-Learning**: Not a single algorithm, but a **family of methods** built around one core idea -- use a deep neural network $Q(s, a; \theta)$ to approximate the Q-function instead of a table. The loss function is the MSE between the TD target and the current Q-value:
>
> $$\mathcal{L}(\theta) = \mathbb{E}\left[ \big( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \big)^2 \right]$$
>
> But this introduces **two critical problems**:
>
> | Problem | Cause | Consequence |
> |:---|:---|:---|
> | **Highly correlated samples** | Consecutive frames are strongly correlated, not i.i.d. | Gradient updates are unstable, the network easily diverges |
> | **Moving target** | The TD target uses $Q(s',a';\theta)$ with the same parameters as the $Q(s,a;\theta)$ being updated | Chasing a moving target -- like a dog chasing its own tail |
>
> ---
>
> **DQN (Deep Q-Network)**: Mnih et al. 2015 Nature, the **most classic concrete algorithm** in the deep Q-Learning family. It solves the two problems above with two innovations:
>
> **Innovation 1: Experience Replay** -- Solving "highly correlated samples"
>
> Store each step $(s, a, r, s')$ in a replay buffer. During learning, sample a mini-batch randomly -- this breaks the correlation between consecutive frames, making the data approximately i.i.d.
>
> **Innovation 2: Target Network** -- Solving "moving target" (this is the inspiration for MoCo's momentum encoder)
>
> Maintain two Q-networks:
> - Online network $Q(s, a; \theta)$: normal gradient updates
> - Target network $Q(s, a; \theta^-)$: **hard-copied from online every $C$ steps**
>
> In the loss function, the TD target uses $\theta^-$ while the current estimate uses $\theta$. Since $\theta^-$ stays fixed for $C$ steps, the TD target stops moving every step -- training stabilizes.
>
> ```
> Parameter changes during training (assuming C=100):
>
> step:    0    100   200   300   400   500   ...
> theta:  ████████████████████████████████████     (gradient update every step)
> theta-: ████________████________████________    (hard copy every C steps)
> ```
>
> **Term hierarchy summary**:
>
> ```
> Q-Learning
>   +-- Uses a table for Q-values, Bellman equation iterative update
>
> Deep Q-Learning
>   +-- Method family: uses neural networks to approximate the Q-function
>   |    |
>   |    +-- DQN (Mnih 2015)            <- Pioneer: Experience Replay + Target Network
>   |    +-- Double DQN (2016)          <- Fixes Q-value overestimation
>   |    +-- Dueling DQN (2016)         <- Separates V(s) and A(s,a)
>   |    +-- Prioritized ER             <- Prioritized replay of important samples
>   |    +-- Rainbow (2018)             <- All of the above combined
>   |    +-- ... more variants
> ```
>
> DQN is a concrete instance of deep Q-Learning, just like ResNet is a concrete instance of CNN architectures.
>
> ---
>
> **Back to MoCo's comparison**:
>
> | | DQN's Target Network | MoCo's Momentum Encoder |
> |:---|:---|:---|
> | **Two networks** | online Q-network + target Q-network | query encoder $f_q$ + key encoder $f_k$ |
> | **Online update** | Gradient descent | Gradient descent |
> | **Target update** | **Hard copy**: $\theta^- \leftarrow \theta$ every $C$ steps | **Soft momentum**: $\theta_k \leftarrow 0.999\theta_k + 0.001\theta_q$ every step |
> | **Parameter curve** | Step function (piecewise constant) | Smooth curve (exponential moving average) |
> | **Purpose** | Stabilize the TD target | Maintain key consistency across the dictionary |
>
> DQN's hard copy means that by step $C-1$, $\theta^-$ is already $C-1$ steps behind and the gap to the target is large. MoCo's soft momentum moves only 1 per mille per step -- all keys always come from nearly the same encoding space. **A step function turned into a smooth curve.**

</details>

#### Sensitivity experiment for $m$ (paper Table 3)

| $m$ | ImageNet Top-1 | Note |
|:---|:---:|:---|
| 0.9 | Significant drop | Too fast, consistency breaks |
| 0.99 | ~58% | Works, but not optimal |
| **0.999** | **60.6%** | **Optimal** |
| 0.9999 | Slight drop | Too slow, $\theta_k$ barely learns new information |

Conclusion: $m$ has a sweet spot -- too fast breaks consistency, too slow and the encoder can't learn new information. $m=0.999$ is the optimal balance.

### 3.3 Shuffling BN: Preventing BatchNorm Information Leakage

> **Warning**: This is a critical implementation detail that many online summaries completely miss. It's also central to understanding MoCo's engineering.

#### How BN causes "cheating" in contrastive learning

The BN cheating mechanism is covered in detail in the [SimCLR post](/blog/simclr#23-global-bn-how-batchnorm-lets-the-model-cheat-and-how-to-fix-it). In short: per-GPU BatchNorm statistics embed a "GPU fingerprint" into features; since both views of an image land on the same GPU, the model learns to match BN offsets instead of learning semantics. Pretext accuracy looks great, representation quality is garbage.

#### MoCo's Shuffle BN

MoCo's solution is elegant -- **shuffle the samples only during the key encoder's forward pass**:

```
Multi-GPU training (assuming 8 GPUs):

Step 1: Compute queries normally
  GPU-0: x_q[0] -> f_q -> q[0]   (local BN statistics)
  GPU-1: x_q[1] -> f_q -> q[1]
  ...

Step 2: Shuffle the key samples
  Original: GPU-0 has x_k[0:32], GPU-1 has x_k[32:64], ...
  After shuffle: each GPU gets a mix of samples from different GPUs

Step 3: Compute keys using the shuffled samples
  GPU-0: x_k_shuffled[0] -> f_k -> k_shuffled[0]   (BN statistics from mixed samples)
  ...

Step 4: Unshuffle keys to restore the original order
  k = unshuffle(k_shuffled)  # restore the order matching q

Step 5: Compute the loss
  loss = InfoNCE(q, k, queue)
```

**Why this fixes the leak**:

- The key's BN statistics come from **cross-GPU mixed samples**, no longer from "the same GPU batch that produced the query."
- The BN statistics for query and positive key are **decoupled** -- the GPU fingerprint is no longer correlated.
- The query encoder's BN is left normal (no shuffle), since the query side doesn't need this defense.

#### Comparison with other BN strategies

| Method | BN strategy | Cost |
|:---|:---|:---|
| **MoCo** | **Shuffle BN** | Requires extra shuffle/unshuffle operations |
| **SimCLR** | Global BN (sync $\mu, \sigma$ across GPUs) | Requires all-reduce communication |
| **CPC v2** | Layer Norm instead of BN | LN is generally less effective than BN on CNNs |

> **A subtle point about Shuffle BN**: Shuffle BN does **not** completely eliminate the leak -- samples on the same GPU after shuffling still have some statistical correlation. But it reduces the leakage to a level the model can't exploit. From the paper's experiments, removing Shuffle BN causes a significant performance drop, showing it's critical to MoCo's success.

---

## 4. Loss Function: InfoNCE

MoCo uses the InfoNCE loss (also called NT-Xent) with $K$ negative keys drawn from the queue:

$$ \mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)} $$

This is a $(K+1)$-way softmax cross-entropy — the correct answer is always $k_+$ (position 0), and the other $K$ keys are wrong classes. For a full derivation of InfoNCE from NCE, see the [[NCE]] and [[CPC]] posts.

### Temperature $\tau$: MoCo vs SimCLR

The general role of $\tau$ is explained in the [SimCLR post](/blog/simclr#33-the-role-of-temperature-tau). The MoCo-specific insight is **why $\tau=0.07$ rather than SimCLR's $\tau=0.1$**:

$$ \text{softmax probability} = \frac{\exp(\text{sim} / \tau)}{\sum \exp(\text{sim} / \tau)} $$

MoCo's smaller $\tau$ makes the model more sensitive to hard negatives. This is deliberate: MoCo's queue holds 65,536 negatives where hard negatives are more prevalent — a smaller $\tau$ prevents the model from being overwhelmed. SimCLR has fewer negatives per batch (at most 8,191), so a larger $\tau$ distributes gradients more evenly. Two different negative strategies, two different optimal temperatures.

---

## 5. Algorithm Pseudocode and Implementation Details

### 5.1 Complete pseudocode

Paper Algorithm 1 with line-by-line annotations:

```python
# Initialization: key encoder parameters copied from query encoder
f_k.params = f_q.params

for x in loader:                    # x: N unlabeled images, N = batch_size = 256
    x_q = aug(x)                    # query augmented view (random crop + color distortion, etc.)
    x_k = aug(x)                    # key augmented view (another random augmentation of the same image)

    q = f_q.forward(x_q)            # query features, shape NxC (256x128)
    k = f_k.forward(x_k)            # key features, shape NxC (requires Shuffle BN first)
    k = k.detach()                  # keys don't participate in gradient computation!

    # Positive logit: match between each image and its own augmented feature
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))       # -> Nx1

    # Negative logit: match between each query and all keys in the queue
    l_neg = mm(q.view(N,C), queue.view(C,K))         # -> NxK (256x65536)

    # Concatenate: positive is always at position 0
    logits = cat([l_pos, l_neg], dim=1)              # -> Nx(1+K)

    # Ground truth: all zeros! Because the positive is always at position 0
    labels = zeros(N)                                 # -> N (all 0)

    # InfoNCE = cross-entropy with temperature
    loss = CrossEntropyLoss(logits / tau, labels)

    # Backpropagation
    loss.backward()
    update(f_q.params)                               # only f_q is updated via gradients

    # Momentum update of key encoder (no gradients!)
    f_k.params = m * f_k.params + (1 - m) * f_q.params

    # Update queue: enqueue current batch's keys, dequeue oldest
    enqueue(queue, k)                                 # add k (NxC) at the front
    dequeue(queue)                                    # pop N oldest from the tail
```

### 5.2 Why $K=65536$?

| Aspect | Detail |
|:---|:---|
| **Queue size** $K$ | 65536 ($2^{16}$), roughly 5% of ImageNet's 1.28M training images |
| **Batch size** $N$ | 256 |
| **$K/N$ ratio** | 256x -- each key lives in the queue for roughly 256 steps on average |
| **Feature dimension** $C$ | 128 |
| **Queue memory** | $65536 \times 128 \times 4 \text{ bytes} \approx 33.6 \text{ MB}$ (tiny) |

> **Key clarification: $K=65536$ is the dictionary size, not "over 65,000 images"**. Some summaries vaguely say "over 60,000 images." The precise statement: the dictionary holds **65,536 key feature vectors**, each encoded by the key encoder from an augmented view of some image. These 65,536 keys come from the most recent 256 batches (each batch has 256 images). Since it's a sliding window, the 65,536 keys correspond to roughly 65,536 **different augmented samples** (original images may repeat, but the random augmentations differ each time).

### 5.3 Implementation Notes from Hand-Written Code

Common implementation details when writing contrastive learning from scratch:

- **Positive logit computation**: Use `bmm` (batch matrix multiply) to compute $q \cdot k_+$ per sample, giving an $N \times 1$ vector
- **Negative logit computation**: Use `mm` (matrix multiply) to compute all dot products between all queries and all keys in the queue at once, giving an $N \times K$ matrix
- **The `labels = zeros(N)` trick**: Since `cat` places `l_pos` at column 0, the correct class ID for every positive sample is always 0 -- so the ground truth is just a zero vector. This avoids building labels per sample.

---

## 6. Experimental Results

### 6.1 Scaling behavior of three methods (paper Figure 3)

The paper plots $K$ (number of negatives / dictionary size) on the x-axis against ImageNet Top-1 on the y-axis:

| Method | Max $K$ | Trend |
|:---|:---|:---|
| End-to-end | At most 1024 (memory-bound) | Stops after 3 points; worse than MoCo at $K=1024$ |
| Memory Bank | Up to 65536 | Improves with $K$ but always below MoCo (poor consistency) |
| **MoCo** | 65536 (and beyond) | Steadily improves with $K$; best at every $K$ value |

**Core takeaway**: MoCo is not only the best performer, but also the most hardware-friendly (works with small batches), and scales the best (can grow $K$ without adding GPUs).

### 6.2 ImageNet linear classification

Using a frozen pretrained ResNet-50 representation with a single linear classifier:

| Method | Top-1 | Top-5 |
|:---|:---:|:---:|
| **MoCo** | **60.6%** | -- |
| InstDisc | 54.0% | -- |
| BigBiGAN | 56.6% | -- |
| CMC | 60.0% | 82.3% |
| PIRL | 63.6% | -- |
| CPC v2 | 63.8% | -- |
| *SimCLR (later, Feb 2020)* | *69.3%* | -- |

> MoCo v1's 60.6% looks lower than SimCLR's later 69.3%, but it was ahead of all concurrent methods from late 2019. MoCo v2 then quickly caught up to 71.1% by incorporating SimCLR's improvements.

### 6.3 Transfer learning: first time surpassing supervised pretraining across the board

This is the most striking result in the paper. Evaluated on **7 downstream tasks**:

| Task | Random init | Supervised pretrain | **MoCo pretrain** | MoCo (IG-1B) |
|:---|:---:|:---:|:---:|:---:|
| PASCAL VOC detection | 33.8 | 81.3 | **82.2** | **83.0** |
| COCO detection | -- | -- | Slightly above supervised | Higher |
| COCO instance segmentation | -- | Slightly higher | Slightly lower | On par / slightly higher |
| COCO semantic segmentation | -- | Slightly higher | Slightly lower | On par |
| COCO human keypoint detection | -- | -- | **Higher** | **Higher** |

> **Historical significance**: This is the **first time unsupervised pretraining comprehensively surpassed supervised pretraining in computer vision**. Before this, the consensus was that unsupervised pretraining could at best approach supervised, never surpass it. MoCo broke that belief, demonstrating that unsupervised pretraining is not only viable but actually better for some tasks (especially detection and keypoint detection).

### 6.4 The 30x learning rate observation

When fine-tuning a MoCo-pretrained model on downstream tasks, the optimal learning rate (lr=30) is much larger than the fine-tuning learning rate for supervised pretraining (lr=0.03).

**What this means**: MoCo learns a feature distribution that is **fundamentally different** from supervised pretraining. MoCo's features are more spread out in the feature space (because the contrastive loss pushes different samples apart), requiring a larger learning rate to adjust them effectively during fine-tuning.

**MoCo's solution**: Apply full BatchNorm normalization to the entire model (including the FPN structure for detection) -- this unifies the feature distribution and lets you use the same hyperparameters as supervised pretraining for fine-tuning, without grid-searching the learning rate every time.

---

## 7. MoCo's Evolution: v2 to v3

### 7.1 MoCo v2 (March 2020): Absorbing SimCLR's improvements

Chen et al.'s MoCo v2 changed only three things to push ImageNet Top-1 from 60.6% to **71.1%**:

| Improvement | Source | Effect |
|:---|:---|:---|
| **MLP projection head** (2-layer FC + ReLU) | SimCLR | +~6% |
| **Stronger data augmentation** (blur + stronger color distortion) | SimCLR | +~3% |
| **Cosine learning rate schedule** | SimCLR | +~1.5% |

MoCo v2 still only needs a **batch size of 256**, nowhere near SimCLR's 8192. This proves the architectural advantage of queue + momentum encoder -- it can absorb the data augmentation improvements from end-to-end methods while keeping its memory efficiency.

### 7.2 MoCo v3 (April 2021): Adapting to ViT, discovering training instability

MoCo v3 attempted to transfer the MoCo framework from ResNet to Vision Transformer (ViT):

**Key changes**:
- Encoder switched from ResNet to ViT
- Queue removed -- switched to SimCLR-style in-batch negatives (batch size 4096)
- Momentum encoder retained
- Added an extra **prediction head** (inspired by BYOL)

**Surprising discovery: training instability**:

> MoCo v3 shows **sudden spikes/dips in the training curve** when the batch size is large (4096+) -- the loss suddenly spikes and then slowly recovers. This does not happen with small batch sizes.

MoCo v3 partially mitigates this by **freezing the patch projection layer** (ViT's first linear projection), but the root cause is still not fully understood -- it remains an open problem. Later methods like DINO and DINO v2 took the self-distillation route, partially bypassing this instability.

### 7.3 Evolution summary

```
MoCo v1 (Nov 2019, CVPR 2020)
  +-- Core contributions: queue + momentum encoder + Shuffle BN
  +-- First to surpass supervised pretraining on multiple downstream tasks
  |
  +--> MoCo v2 (Mar 2020)
  |     +-- Absorbed SimCLR: MLP head + stronger augmentation + cosine lr
  |     +-- Proved: queue architecture is more efficient than large batches
  |
  +--> MoCo v3 (Apr 2021)
        +-- Adapted to ViT, discovered contrastive learning instability on ViT
        +-- Freezing patch projection layer as a mitigation
        +-- Issue not fully resolved -> DINO/DINO v2 went the self-distillation route
```

---

## 8. Systematic Comparison with Concurrent Methods

### 8.1 Fundamental differences between the three paradigms

| Dimension | End-to-end (SimCLR) | Memory Bank (InstDisc) | **MoCo** |
|:---|:---|:---|:---|
| **Negative source** | Current batch | Stored features from the full dataset | Momentum encoder + sliding window queue |
| **Dictionary size $K$** | $= N$ (batch-limited) | $= \|$Dataset$\|$ (full) | 65536 (configurable) |
| **Feature consistency** | Very high (real-time updates) | Low (updated once per epoch) | High (momentum update, very smooth) |
| **Memory scalability** | Poor (needs large batch) | Poor (stores all features) | Good (queue stores only $K$) |
| **Extra components** | None | Proximal regularization on encoder | Momentum encoder + queue |
| **Batch size** | >= 4096 | 256 | 256 |

### 8.2 Who "won"?

Looking back historically, there was no absolute winner -- just convergence:

- **SimCLR won on methodology**: Data augmentation, projection head, large batch + long training -- these design principles were absorbed by everyone.
- **MoCo won on architecture**: The momentum encoder was adopted by BYOL, SimSiam, MoCo v3, and others. It proved that "you don't need large batches to do contrastive learning well."
- **Memory Bank faded out**: Replaced by the queue, which only keeps a sliding window rather than the full dataset, scaling much better.

---

## 9. Strengths and Weaknesses

### Strengths

1. **Elegant architecture**: Unifies contrastive learning under the dictionary look-up framework, bringing all previous methods into a single coherent perspective. The paper is only 4 pages (CVPR double-column), but the design is crystal clear.

2. **Momentum encoder**: Simple design with lasting impact. The $m=0.999$ value was adopted by almost all subsequent work -- BYOL, SimSiam, DINO, and others. It was inspired by the target network from DQN (Deep Q-Network, Mnih et al. 2015), but the soft momentum is smoother and more elegant than hard copying.

3. **Hardware-friendly**: Requires only batch size 256, trainable on a single V100 16GB. On 8 V100s, 200 epochs takes about 53 hours. Compared to SimCLR's need for TPU pods or 32+ V100s, the barrier to entry is extremely low -- letting many more researchers work on contrastive learning.

4. **Scalable**: $K$ and $N$ are decoupled, so the dictionary can grow arbitrarily large. Performance keeps improving when pretraining on the Instagram 1B dataset.

5. **First to break the unsupervised vs. supervised ceiling**: Surpassed supervised pretraining on detection and keypoint detection tasks.

### Weaknesses

1. **Extra components to maintain**: The queue and momentum encoder add engineering complexity. Compared to SimCLR's "pure end-to-end, no extras," MoCo's code is more involved.

2. **Shuffle BN is a patch**: It solves the BN leakage problem, but not in the most elegant way -- it essentially admits that BN is problematic for contrastive learning while CNNs can't live without it. In the Transformer era, Layer Norm naturally avoids the issue.

3. **InfoNCE still needs negatives**: Later work (BYOL, SimSiam) showed that you don't even need negatives, as long as you can prevent representation collapse. MoCo's reliance on negatives was later shown to be non-essential.

4. **CNN-era methodology**: Shuffle BN and queue design both depend heavily on the CNN + BN stack. The ViT + LN combination makes these problems naturally disappear. MoCo v3's instability on ViT also suggests this methodology doesn't fully transfer to Transformers.

5. **The ceiling of contrastive learning**: InfoNCE optimizes a lower bound on mutual information, but maximizing mutual information may not be sufficient for good representations. MAE went in the opposite direction (reconstructing pixels) and achieved better results on some tasks.

---

## 10. Related Notes

- [[NCE]] -- The mathematical roots of InfoNCE: from multi-class to binary classification
- [[CPC]] -- The paper that introduced InfoNCE: contrastive prediction in representation space
- [[SimCLR]] -- Concurrent contrastive learning method: end-to-end, large batch, simplicity-first
- [[BYOL]] -- No negatives needed: proving the momentum encoder's value is not about providing negatives
- [[SimSiam]] -- Further simplification: stop-gradient alone is enough
- [[SSL]] -- Self-supervised learning overview

---

## 11. References

- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR 2020. arXiv:1911.05722.
- Chen, X., Fan, H., Girshick, R., & He, K. (2020). *Improved Baselines with Momentum Contrastive Learning* (MoCo v2). arXiv:2003.04297.
- Chen, X., Xie, S., & He, K. (2021). *An Empirical Study of Training Self-Supervised Vision Transformers* (MoCo v3). ICCV 2021. arXiv:2104.02057.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature. -- DQN target network, inspiration for the momentum encoder
- Wu, Z., Xiong, Y., Yu, S., & Lin, D. (2018). *Unsupervised Feature Learning via Non-Parametric Instance Discrimination* (InstDisc). CVPR 2018. -- Memory Bank approach
