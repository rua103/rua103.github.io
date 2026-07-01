---
title: "BYOL: Bootstrap Your Own Latent"
description: "BYOL proves contrastive learning works without negative samples. A student predicts a momentum teacher's output, and the architecture itself prevents collapse."
publishDate: 2026-07-01
tags:
  - self-supervised-learning
  - contrastive-learning
  - byol
  - ema
  - stop-gradient
  - momentum-encoder
  - no-negative-samples
---

# BYOL: Bootstrap Your Own Latent

> Contrastive learning does not need negative samples. Train one network (the student) to predict the output of another slowly-updated network (the teacher) on a different view of the same image. Match these two representations with MSE -- no repulsion, no collapse.

---

## 0. Notation

| Symbol | Meaning |
|:---|:---|
| $x$ | Original image |
| $v = t(x)$ | A view of $x$ under data augmentation $t \sim \mathcal{T}$ |
| $v' = t'(x)$ | Another view of $x$ under an independent augmentation $t' \sim \mathcal{T}$ |
| $f_\theta$ | **Online** encoder (ResNet-50), updated via normal gradient descent |
| $f_\xi$ | **Target** encoder, updated via EMA, **no gradient backpropagation** |
| $g_\theta$ | **Online** projector (MLP), updated via normal gradient descent |
| $g_\xi$ | **Target** projector (MLP), updated via EMA |
| $q_\theta$ | **Predictor** (MLP), **only present in the online network** -- this is the key asymmetry |
| $z_\theta = g_\theta(f_\theta(v))$ | Representation output by the online projector |
| $z'_\xi = g_\xi(f_\xi(v'))$ | Representation output by the target projector |
| $\tilde{z}_\theta = q_\theta(z_\theta) / \|q_\theta(z_\theta)\|_2$ | Prediction output by the online predictor (after L2 normalization) |
| $\bar{z}'_\xi = z'_\xi / \|z'_\xi\|_2$ | Target projector representation (after L2 normalization) |
| $\tau$ | EMA coefficient, default $\tau = 0.996$ (note: different meaning from $\tau$ in InfoNCE!) |
| $\xi$ | Target network parameters ($\xi \leftarrow \tau \xi + (1-\tau)\theta$) |

> **Warning: the double meaning of $\tau$**
> In MoCo / SimCLR, $\tau$ is the temperature parameter in InfoNCE. In BYOL, $\tau$ is the EMA momentum coefficient, defaulting to 0.996. The two $\tau$'s have nothing to do with each other.
>
> **L2 normalization**
> In the paper, both $\tilde{z}_\theta$ and $z'_\xi$ are L2-normalized before use, so they have unit norm. This is the prerequisite for the MSE = 2 - 2 cos equivalence. For brevity, $\tilde{z}_\theta$ and $z'_\xi$ throughout this post refer to the normalized vectors.

---

## 1. Motivation: throwing away negative samples

### 1.1 The consensus in early 2020

By early 2020, SimCLR and MoCo had already established that the core of contrastive learning is InfoNCE:

$$ \mathcal{L} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)} $$

The role of the $K$ negative samples in the denominator is crystal clear: **they hold the model back, preventing it from pulling everyone into the same place (collapsing to a constant output).** Without this repulsive force in the denominator, the model's easiest move is: no matter what the input is, the encoder outputs a constant vector (e.g., all zeros or some fixed value) that is independent of the input. Positive-pair similarity would also be zero, the loss would be perfect, but the representation would be completely meaningless.

**The consensus**: negative samples are necessary to prevent collapse.

BYOL asked a rebellious question: really?

### 1.2 Reflection: why didn't we think of this

> "Top researchers really are on another level. I would never have thought to remove negative samples... I was completely trapped inside the contrastive framework, while the real heavyweights step right out of it with a single stride."

This quote nails the intellectual breakthrough behind BYOL. At the time, everyone was thinking about how to make negative samples better -- more balanced sampling, memory bank clustering, incorporating AET-style equivalent objectives -- all just tinkering inside the contrastive box. BYOL's approach was to smash the box entirely.

> "BYOL's story and implementation are both remarkably simple -- simpler than SimCLR and MoCo. What's not simple is being able to think of it."

---

## 2. Core framework: predicting yourself from yourself

### 2.1 Dual-network architecture

```
                   x (original image)
                   │
          ┌────────┴────────┐
          ▼                 ▼
    augment t ~ T      augment t' ~ T
          │                 │
          v                 v'
          │                 │
    ┌─────┴─────┐     ┌─────┴─────┐
    │ online θ  │     │ target ξ  │
    │           │     │           │
    │ f_θ → g_θ │     │ f_ξ → g_ξ │
    │     ↓     │     │     ↓     │
    │   z_θ     │     │   z'_ξ    │
    │     ↓     │     │           │
    │ q_θ(z_θ)  │     │           │
    │     ↓     │     │           │
    │   z̃_θ     │────▶│  MSE loss │  ← stop_gradient(z'_ξ)
    └───────────┘     └───────────┘

    Gradient flow: online ← backprop   ✗ target ← no backprop
    Parameter update: θ ← SGD   |   ξ ← τ·ξ + (1-τ)·θ
```

**The key asymmetry**: only the online network has an extra **predictor** $q_\theta$. The target network has no predictor.

### 2.2 Loss function

For two views, the online network predicts the target network's output:

$$ \begin{aligned} \mathcal{L}_{\theta, \xi} &= \left\| \tilde{z}_\theta - z'_\xi \right\|_2^2 \\ &= \|\tilde{z}_\theta\|_2^2 + \|z'_\xi\|_2^2 - 2 \langle \tilde{z}_\theta, z'_\xi \rangle \quad \text{(expand the square)} \\ &= 1 + 1 - 2 \langle \tilde{z}_\theta, z'_\xi \rangle \quad \text{(L2 normalization, unit norm)} \\ &= 2 - 2 \cdot \frac{\langle \tilde{z}_\theta, z'_\xi \rangle}{\|\tilde{z}_\theta\|_2 \cdot \|z'_\xi\|_2} \quad \text{(unit norm, equivalent to cosine similarity)} \end{aligned} $$

> Since both $\tilde{z}_\theta$ and $z'_\xi$ are L2-normalized, MSE is equivalent to $2 - 2 \cdot \cos(\tilde{z}_\theta, z'_\xi)$ -- essentially maximizing cosine similarity.

**Symmetric loss** (swap the views and do it again):

$$ \mathcal{L}_{\text{BYOL}} = \mathcal{L}_{\theta, \xi} + \tilde{\mathcal{L}}_{\theta, \xi} $$

where $\tilde{\mathcal{L}}_{\theta, \xi}$ is the symmetric version with $v$ fed to the target and $v'$ fed to the online network.

### 2.3 Parameter updates

```
online θ: normal backpropagation → SGD/Adam update
target ξ: no gradient backprop (stop_gradient(z'_ξ)), updated via EMA:
  ξ ← τ · ξ + (1 - τ) · θ
  where τ_base = 0.996, and τ follows a cosine schedule from 0.996 up to 1.0 during training
```

**τ changes during training**:

The paper actually uses $\tau = 1 - (1 - \tau_{\text{base}}) \cdot \frac{\cos(\pi k / K) + 1}{2}$, starting at 0.996 and approaching 1.0 by the end. The target always follows the online network slowly: each step only mixes in $(1-\tau)$ of the online parameters. At the start of training, $(1-\tau) = 0.004$ (the target takes only 0.4% of the online information per step, retaining 99.6% of itself), and this fraction gradually decreases to 0 under the cosine schedule -- throughout training, the target moves extremely slowly, eventually stopping entirely.

### 2.4 Full algorithm pseudocode

```python
# Initialization
θ = random_init()        # online parameters
ξ = θ                    # target parameters copied from online

for x in loader:         # batch of images
    v  = t(x)            # view 1
    v' = t'(x)           # view 2

    # Online forward (v → online, v' → target)
    z_θ   = g_θ(f_θ(v))          # online projector output
    z̃_θ   = q_θ(z_θ)             # online predictor output
    z'_ξ  = g_ξ(f_ξ(v'))         # target projector output
    loss1 = MSE(z̃_θ, stop_grad(z'_ξ))

    # Symmetric: v' → online, v → target
    z_θ2  = g_θ(f_θ(v'))
    z̃_θ2  = q_θ(z_θ2)
    z_ξ   = g_ξ(f_ξ(v))
    loss2 = MSE(z̃_θ2, stop_grad(z_ξ))

    loss = loss1 + loss2
    loss.backward()              # only θ receives gradients
    update(θ)                    # SGD/Adam update for online

    # Target momentum update, no gradients
    ξ = τ * ξ + (1 - τ) * θ

return f_θ  # same as SimCLR: discard the projection head, keep only the encoder
```

---

## 3. Why it doesn't collapse without negative samples

This is the single most-discussed question about the paper, and the one that most needs a clear explanation.

### 3.1 The empirical facts about collapse

> **Note on epoch configuration**
> The ablation experiments in the paper all run at **300 epochs** (not the full 1000-epoch training), batch size 4096, $\tau_{\text{base}}=0.99$, with a baseline of **72.5%**. The 74.3% from the full training run appears in Table 1. All ablation numbers discussed below are 300-epoch results.

Let's look at the ablations (Table 5 in the paper):

**Table 5a: Different target modes** (300 epochs, $\tau_{\text{base}}=0.99$)

| Target mode | Top-1 |
|:---|---:|
| EMA target ($\tau_{\text{base}}=0.99$, BYOL default) | **72.5%** |
| Constant random network ($\tau=1$, never updated) | 18.8% |
| EMA $\tau_{\text{base}}=0.999$ | 68.4% |
| $\tau=0$ (copy online directly each step) | Unstable training, extremely poor |
| Stop-gradient of online (no EMA) | 0.3% |

**Table 5b: Intermediate variants between BYOL and SimCLR** ($\lambda$ is the negative sample weight)

| Predictor | Target (EMA) | $\lambda$ | Top-1 | Corresponds to |
|:---:|:---:|:---:|:---:|:---|
| Yes | Yes | 0 | **72.5%** | BYOL |
| No | No | 1 | 69.1% | SimCLR |
| No | Yes | 0 | 0.3% | Collapse |
| Yes | No | 0 | 0.2% | Collapse |

The conclusion is clear: **EMA and predictor are both indispensable.** Together they form the anti-collapse mechanism. Note that $\tau=1$ (target never updated) gives 18.8%, which is not collapse -- training is stable but cannot iteratively improve, so the final representation quality is low. True collapse (0.1%--0.3%) only happens when either the predictor or EMA is missing. So how do they each work? There are three levels of explanation.

### 3.2 Level 1: EMA target = a moving target

This is the most intuitive explanation.

If target = online ($\tau=0$), the problem becomes:

> Use a fixed view $v$ to predict the same network's output on another view $v'$.

The network quickly discovers: "I don't need to learn any semantics. Whatever the input is, I can just output a constant. The two views will have the same output either way, so MSE will be zero."

But an EMA target is a **slowly moving target**. When the online network tries to output a constant, it finds that it cannot -- the target is gradually changing, and a constant output cannot keep up. The online network must continuously learn useful features to match the target's output.

### 3.3 Level 2: Predictor asymmetry = information filtering

Having EMA alone, without a predictor, also leads to collapse (second row of Table 5).

What does the predictor $q_\theta$ do? Here is one way to think about it:

- **The projector output $z_\theta$** encodes "what this image looks like under the current view"
- **The predictor $q_\theta$** learns to "predict the encoding of another view, given the encoding of the current view"

This forces the predictor -- and, through backpropagation, the encoder -- to learn **information that is consistent across both views**. Color? Different views may differ (one reddish, one bluish), so the predictor learns to ignore it. Shape and semantics? The same across both views, so the predictor learns to preserve them. This is exactly the kind of good representation we want.

**Intuition**: the predictor is an "information sieve." It filters out information that varies across views (color, texture, orientation) and retains information that is invariant across views (semantics, structure).

> This is the **complementary opposite** of SimCLR's projection head. SimCLR's projector is a "sacrificial layer" that absorbs the invariance pressure from the contrastive loss, protecting the information completeness of the encoder output $h$ (see [[SimCLR#Finding 2]]). BYOL's predictor goes in the opposite direction: it actively learns "what information needs to be retained to make good predictions" and feeds that filtering ability back to the encoder.

### 3.4 Level 3: CNNs have built-in contrastiveness -- the "ladder-climbing" hypothesis

> **Note on this section**
> The following Section 3.4 is not an analysis from the paper itself, but rather a widely-circulated interpretation of BYOL's mechanism from the Zhihu community. Section 3 of the paper provides the empirical basis for bootstrapping (a fixed random network yields 18.8% > the random baseline of 1.4%), but phrases like "CNNs have built-in contrastiveness," "BYOL mainly adds transformation invariance," and "EMA balances stability and efficiency" are community interpretations, not claims from the paper. The paper's own theoretical intuition (Section 3.2) follows the route of conditional variance minimization + optimal predictor.

A Zhihu analysis proposed an extraordinarily elegant explanation. The original paper does not discuss this explicitly, but the intuition is profoundly insightful.

> "A randomly initialized CNN already possesses a certain capacity for extracting image features. Furthermore, if a randomly initialized CNN already maps different input samples to different locations in feature space, and maps them sufficiently far apart, then it has already accomplished the first objective of the contrastive learning framework."

Contrastive learning has two objectives:
1. **Representations of different images should be different** (contrastive property) -- traditionally guaranteed by negative samples
2. **Representations of different views of the same image should be identical** (invariance property) -- guaranteed by the positive pair term in InfoNCE

The hypothesis:

> A randomly initialized CNN naturally possesses Objective 1. So BYOL does not need negative samples. It only needs to use predictor + EMA to carefully achieve Objective 2, while maintaining Objective 1 without destroying it.

**The way Objective 1 is maintained is "don't let the teacher catch up to the student too quickly"**:

> Table 5 in the paper shows that catching up too fast destroys the contrastive property (collapse to a constant), while catching up too slowly harms training efficiency. EMA is the tool for balancing stability and efficiency -- when the teacher follows slowly, the features the student learns naturally maintain discriminability across different samples.

**If this hypothesis is correct**, then the foundation that makes BYOL work is: the architecture, on image data, naturally possesses contrastive representational capacity that does not depend on training. What BYOL mainly does is add transformation invariance on top of this natural representation. If you switched to an architecture without this natural capacity (e.g., a pure MLP) or a different data modality (e.g., text), BYOL might not work, and negative samples would need to come back to the rescue. However, subsequent experiments showed that ViTs also work well with this framework (see the revision in Section 3.6), suggesting this "natural capacity" is not CNN-specific.

**The "ladder-climbing" thought experiment**:

The core insight of Section 3 of the paper is: if you have two randomly initialized networks A and B, and A is held fixed while B is trained using A's output as the supervision signal, B ends up better than A. Then if you let C learn from B, D learn from C... left foot on right foot, spiraling upward.

```
A (random init)
  │  held fixed, output used as supervision
  ▼
B (random init, trained on A's output)
  │  B > A! magical
  │  held fixed
  ▼
C (random init, trained on B's output)
  │  C > B!
  │
  ▼
D … left foot on right foot, limitless power
```

**EMA simply turns this alternating learning into small, rapid steps**: instead of fully training B before starting C, at each step the online network (student) learns from the target network (teacher / past self), and the target slowly chases the online network. The essence is equivalent to the ladder-climbing thought experiment, just implemented more efficiently.

### 3.5 Extra line of defense: the BatchNorm controversy

BYOL's projector and predictor both use BatchNorm. During computation, BatchNorm subtracts the batch mean and divides by the batch variance -- meaning **each sample's output depends on the statistics of other samples in the same batch**.

Although BYOL's MSE loss is independent across samples (unlike InfoNCE, which has explicit inter-sample interaction), BN implicitly introduces cross-sample dependence. One way to understand it:

> If all samples collapse to the same direction, the variance after BN would be zero, and dividing by zero would blow everything up. So BN imposes an implicit pressure on the model -- maintain enough variance; don't output the same thing for everything.

**But this does not explain everything**: later work (Richemond et al., 2020) showed that even when replacing BN with GroupNorm + Weight Standardization (removing all cross-sample dependence), BYOL still does not collapse -- performance drops slightly but no collapse. So BN plays a supporting role, but it is not the sole anti-collapse mechanism. The combined effect of EMA + predictor is the core.

### 3.6 Revising the ladder-climbing hypothesis: it works on ViTs too, so it's not CNN-specific

The hypothesis in Section 3.4 carries an implicit corollary -- if BYOL depends on the initial contrastiveness specific to CNN architectures, it should fail on ViTs. But subsequent experiments proved otherwise.

**Empirical evidence for teacher-student + EMA on ViTs**:

| Work | Architecture | Framework | Result |
|:---|:---|:---|:---|
| **MoCo v3** (2021) | ViT | InfoNCE + momentum encoder | Works, but loss spikes (instability) at large batch sizes |
| **DINO** (2021) | ViT | EMA teacher → student distillation (essentially a BYOL variant + softmax output) | **Extremely successful**, ViT features spontaneously exhibit semantic segmentation |
| **DINO v2** (2023) | ViT | EMA teacher + larger scale | Among the strongest self-supervised visual features today |

So the teacher-student + EMA framework not only works on ViTs, it has become state-of-the-art. The ladder-climbing hypothesis needs revision: it is not a CNN-specific inductive bias, but rather **the EMA + predictor + stop-gradient triad is itself a cross-architecture universal anti-collapse mechanism.**

**So where does the "initial separation" come from in ViTs?**

If the logic that "random initialization brings built-in contrastiveness" is correct, ViT's initial separation ability comes from a different source:

| Architecture | Possible source of natural separation |
|:---|:---|
| CNN | Local connectivity of convolutions + translation equivariance → images with different textures/edges naturally land at different positions in feature space |
| ViT | Patch embedding (linear projection) is a random Gaussian matrix → the Johnson-Lindenstrauss lemma guarantees that random projections approximately preserve distance relationships; plus the distinctness of positional encodings |

A randomly initialized ViT's patch embedding is essentially a locality-sensitive hash function, projecting different images to different positions. The mechanism differs from CNNs, but the effect is similar: **random initialization already separates different samples.**

**A counterintuitive point: contrastive learning is actually harder on ViTs**

MoCo v3 found that InfoNCE is unstable on ViTs (loss spikes), requiring frozen patch projection to mitigate. The BYOL/DINO teacher-student framework does not have this problem -- possibly because it does not need explicit negative-sample repulsion, resulting in smoother training dynamics.

So the conclusion is actually the reverse: it is not that BYOL depends on CNNs and therefore cannot transfer to ViTs, but rather that **BYOL's teacher-student framework is more suitable for ViTs than contrastive learning.** The core logic of the ladder-climbing hypothesis -- architecture brings built-in initial separation + EMA carefully maintains it -- holds for ViTs; only the source of "initial separation" shifts from convolutional bias to random projection.

---

## 4. Deep comparison with MoCo's momentum encoder

BYOL's target network and MoCo's key encoder both use momentum updates, but for entirely different purposes:

|              | MoCo momentum encoder                                      | BYOL target network                            |
| :----------- | :---------------------------------------------- | :---------------------------------------- |
| **Momentum update formula**   | $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ | $\xi \leftarrow \tau\xi + (1-\tau)\theta$ |
| **Can they be identical?**  | $m=0.999$ (V1)                                   | $\tau=0.996$, cosine schedule up to 1.0                   |
| **Negative samples?**  | Yes (65,536 keys in the queue)                              | **No**                                    |
| **Extra component**     | Queue (65,536 key features)                              | Predictor MLP (online only)                 |
| **Loss function**     | InfoNCE (Softmax cross-entropy)                            | MSE (equivalent to maximizing cosine similarity)                           |
| **Role of momentum encoder** | Maintain feature consistency across the key dictionary                                | **Prevent representation collapse** (provides a moving target + implicit regularization)                 |
| **After training**      | $f_k$ discarded, $f_q$ kept                                | Target network discarded, online encoder $f_\theta$ kept   |

The essential difference:

- MoCo's momentum encoder solves an **explicit problem**: consistency of the negative sample queue
- BYOL's target network solves an **implicit problem**: how to prevent collapse without negative samples

The latter is far less intuitive -- MoCo's momentum encoder serves the dictionary; BYOL's target network serves itself. Some analyses argue that the mean teacher in BYOL is merely an "accelerator," and the true prime mover is the fact that ladder-climbing works at all. This is also why this comparison matters: **BYOL uses a technique almost identical to MoCo's (EMA), but to solve a completely different problem.**

---

## 5. Experimental results

### 5.1 ImageNet linear evaluation

Freeze the pretrained encoder, train only a linear classification head:

| Method | Architecture | Top-1 |
|:---|:---|:---:|
| Supervised | ResNet-50 | 76.5% |
| SimCLR | ResNet-50 | 69.3% |
| MoCo v2 | ResNet-50 | 71.1% |
| **BYOL** | ResNet-50 | **74.3%** |
| **BYOL** | ResNet-50 (2x) | **77.4%** |
| **BYOL** | ResNet-50 (4x) | 78.6% |
| **BYOL** | ResNet-200 (2x) | **79.6%** |

BYOL's 74.3% is only 2.2 points below the supervised baseline. Scaling up the network pushes it past supervised performance.

### 5.2 Semi-supervised fine-tuning

| Method | ImageNet 1% labels | ImageNet 10% labels |
|:---|:---:|:---:|
| Supervised (from scratch) | 25.4% | 56.4% |
| SimCLR | 48.3% | 65.6% |
| **BYOL** | **53.2%** | **68.8%** |

BYOL's advantage is even more pronounced in the low-data regime -- 53.2% is nearly double the random-init baseline (25.4%).

### 5.3 Transfer learning (Table 4 in the paper)

Across 12 downstream classification tasks + VOC detection + COCO detection/segmentation, BYOL comprehensively surpasses SimCLR and matches or slightly exceeds supervised pretraining.

### 5.4 Robustness to data augmentation

This is an important property of BYOL. SimCLR depends heavily on color distortion -- removing it causes a severe performance drop. BYOL is far less affected when various augmentations are removed (Table 17 in the paper, 300 epochs):

- Remove color jittering: BYOL drops 0.7 points, SimCLR drops 4.2 points
- Remove color distortion (jittering + grayscale): BYOL drops 9.1 points, SimCLR drops 22.2 points
- Random crop only (remove all other augmentations): BYOL still gets 59.4% (down 13.1 points), far better than SimCLR's 40.3% (down 27.6 points), though still noticeably below the 72.5% with full augmentations
- The reason: BYOL's predictor is itself learning to "infer another view from the current view" -- it is more tolerant of augmentation changes. SimCLR relies on augmentation to "eliminate" the color shortcut (see SimCLR Finding 1); BYOL relies on the predictor to "learn to ignore" color. **Passive exclusion vs. active learning to ignore -- two fundamentally different mechanisms.**

---

## 6. The deeper meaning of the ladder-climbing thought experiment

> "As the paper and its title suggest, anyone with a bit of experience, upon hearing this idea, would immediately suspect it will easily converge to a trivial solution... The important role of negative samples is to hold the model back, preventing it from running wild and outputting all zeros. Take them away, and what ensures the model won't go off the rails?"

BYOL gives its answer: **use your past self as supervision, and let your present self chase it.** The target is a moving one, so you cannot cheat by outputting a constant.

The philosophy behind this is a profound insight: **what really matters in self-supervised learning may not be "contrasting" (pulling positives together and pushing negatives apart), but rather the more fundamental objective of "predicting consistent representations from different views."** Negative samples are merely **one** way to achieve this objective -- SimCLR and MoCo use explicit repulsive forces to guarantee discriminability; BYOL uses implicit architectural design (EMA + predictor + BN) to guarantee discriminability.

Later, SimSiam further showed that **even EMA is unnecessary** -- stop-gradient + predictor alone suffices to prevent collapse. This suggests that BYOL revealed a principle even more fundamental than BYOL itself: as long as you do "predict different views of yourself from yourself" under suitable conditions, representation learning happens naturally, without needing negative samples as "scaffolding."

### Analogy: Noisy Student

| | Noisy Student | BYOL |
|:---|:---|:---|
| **Starting point** | ImageNet supervised pretrained teacher | Random initialization |
| **Knowledge transfer** | Teacher → Student (distillation) | Past self → Present self (self-prediction) |
| **External signal** | Yes (ImageNet labels) | No (only data augmentation) |
| **Essence** | Bootstrapping with labels | Bootstrapping without labels |

A comment from Zhihu:

> "Noisy Student at least had an ImageNet-pretrained teacher as its starting point, while BYOL starts from scratch with random initialization. With no external label assistance whatsoever, relying on accumulation across generations, the later wave (student) continuously inherits knowledge from the earlier wave (mean teacher) and ultimately pulls off a comeback -- it's practically inspirational."

---

## 7. Subsequent impact and evolution

```
BYOL (2020.06, NeurIPS 2020 Oral)
  ├─ Core contribution: proved negative samples are not necessary for contrastive learning
  │   ├─ EMA target network (momentum update → moving target prevents collapse)
  │   ├─ Predictor asymmetry (information filtering + equivalent to alternating learning)
  │   └─ Ladder-climbing intuition: randomly initialized CNNs have built-in contrastiveness
  │
  ├─→ SimSiam (2020.11)
  │     └─ Further simplified: even EMA is unnecessary!
  │     └─ stop-gradient + predictor is the minimal sufficient condition for preventing collapse
  │     └─ "stop-gradient is the key to preventing representation collapse; EMA is just the icing on the cake"
  │
  ├─→ DINO (2021.04)
  │     └─ Self-distillation + ViT: EMA teacher output treated as soft labels
  │     └─ BYOL's teacher-student framework shifts from regression to distillation
  │
  └─→ MAE (2021.11)
        └─ Takes a different path: reconstruct pixels rather than predict representations
        └─ But the idea of "asymmetric architecture prevents collapse" traces back to the same lineage
```

---

## 8. Strengths and weaknesses

### Strengths

1. **Eliminates negative samples**: no longer needs large batches (unlike SimCLR), no negative sample queue (unlike MoCo), no negative sampling strategy. The training pipeline is substantially simplified.

2. **More robust to data augmentation**: removing color distortion (jittering + grayscale) costs SimCLR 22.2 points but BYOL only 9.1 points. BYOL's predictor learns to "ignore color" on its own, rather than relying on augmentation to forcibly erase color.

3. **Strong performance**: 74.3% with ResNet-50, substantially outperforming contemporaneous SimCLR (69.3%) and MoCo v2 (71.1%), just a step away from supervised (76.5%).

4. **Conceptual breakthrough**: demonstrated that "the key to contrastive learning's success may not be negative samples, but the deeper principle of predicting consistent representations from multiple views." This opened the door for subsequent negative-sample-free methods like SimSiam and DINO.

5. **The ladder-climbing intuition is broadly applicable**: the teacher-student bootstrapping framework was later carried forward by DINO and DINO v2, experiencing a second renaissance in the ViT era.

### Weaknesses

1. **Why collapse is prevented is not fully explained**: the paper provides ablations for EMA + predictor, but lacks a rigorous mathematical proof. The debate over whether BN implicitly provides a negative-sample-equivalent effect lasted two years before reaching a preliminary conclusion (Richemond et al. 2020).

2. **The design of EMA and predictor is somewhat heuristic**: why $\tau = 0.996$? Why 2 layers for the predictor? Why do both projector and predictor use a two-layer MLP with hidden dim 4096 and output dim 256? The paper only provides ablations (Table 14), stating that depth=2 works best, but does not explain why this particular structure.

3. **A product of the CNN era**: outside the CNN + BN tech stack, BYOL may require re-validation. Its performance under ViT + LN was not fully explored until the DINO era.

4. **τ requires a cosine schedule**: BYOL's EMA coefficient follows a cosine schedule from 0.996 to 1.0 during training. This is an extra hyperparameter, and while the interpretation of "gradually decelerating pursuit" is intuitive, why this gradual deceleration is needed lacks sufficient theoretical analysis.

5. **The predictor becomes a new bottleneck**: negative samples are gone, but the predictor's design (width, depth, BN placement) becomes a new tuning burden. SimSiam later showed that the predictor does not even need to be complex to work.

---

## 9. Related notes

- [[SimCLR]] -- The strongest contrastive learning baseline before BYOL, requires large batches + negative samples
- [[MoCo]] -- The direct predecessor of the momentum encoder, but MoCo uses momentum to solve dictionary consistency; BYOL uses momentum to prevent collapse
- [[SimSiam]] -- BYOL's direct simplification: removes EMA, keeps only stop-gradient + predictor
- [[SSL]] -- Overview of the self-supervised learning field

---

## 10. References

- Grill, J.-B. et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. NeurIPS 2020 (Oral). arXiv:2006.07733.
- Richemond, P. H. et al. (2020). *BYOL works even without batch statistics*. arXiv:2010.10241. -- Shows that BN is not the sole reason BYOL avoids collapse
- Chen, X. & He, K. (2021). *Exploring Simple Siamese Representation Learning* (SimSiam). CVPR 2021. -- Further simplifies BYOL; EMA is not necessary
- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV 2021. -- BYOL's teacher-student framework carried forward on ViTs
- Zhihu discussion. (2020). How to evaluate DeepMind's new work BYOL? https://www.zhihu.com/question/402452508/answer/1293518245
