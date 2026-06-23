---
title: "Contrastive Predictive Coding"
description: "How CPC reframes unsupervised learning as predicting the future in a compact representation space, using InfoNCE to separate real futures from random ones."
publishDate: 2026-06-23
tags:
  - self-supervised-learning
  - contrastive-learning
  - representation-learning
  - infonce
  - predictive-coding
  - mutual-information
draft: false
---

> Abstract: CPC avoids predicting the future in raw data space — that approach is expensive and requires reconstructing every pixel or sample. Instead, it predicts in a compact representation space, using contrastive learning (InfoNCE) to tell whether a future observation is the real one or a randomly drawn impostor. The resulting representations encode the slow features that capture a signal's global structure.

## Notation

Probability notation ($\mathbb{E}$, $\sim$, $\mid$, $P(\cdot)$) follows the conventions established in the [NCE note](/blog/noise-contrastive-estimation). CPC-specific symbols:

| Symbol | Meaning |
|:---|:---|
| $x_t$ | Raw observation at step $t$ (waveform sample, image patch, word) |
| $z_t = g_{\text{enc}}(x_t)$ | Compact representation produced by the encoder |
| $c_t = g_{\text{ar}}(z_{\leq t})$ | Context vector: the autoregressive model's summary of all past $z$ |
| $f_k(x_{t+k}, c_t)$ | Scoring function for step $k$, estimating the density ratio $\frac{p(x_{t+k}\mid c_t)}{p(x_{t+k})}$ |
| $W_k$ | Linear transformation matrix for prediction step $k$ (one per step) |
| $N$ | InfoNCE candidate set size (1 positive + $N-1$ negatives) |

## 1. Why Direct Prediction Fails

Using MSE to predict the future gives an optimal solution that is just the conditional expectation: $\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x \sim p(x \mid c)}[\|\hat{x} - x\|^2]$, and the minimizer is $\hat{x}^* = \mathbb{E}[x \mid c]$. The result is a pixel-wise weighted average of every plausible future. Given the context "a dog in a park," the dog might be running, sitting, jumping, or sniffing the ground. Each of these is a sharp, coherent image, but their average is a blurry superposition of all possible actions. You are answering a one-bit question ("what will happen?") with a 196,608-dimensional output.

Cross-entropy faces a discretization catastrophe with continuous signals. Autoregressive pixel-by-pixel modeling gives $p_{\theta}(x \mid c) = \prod_{i=1}^{H \times W \times 3} p_{\theta}(x_i \mid x_{<i}, c)$. A single 256×256 RGB image has $256^{196608}$ possible pixel configurations. WaveNet-style models require one forward pass per 16kHz sample. Furthermore, most of the KL divergence goes toward modeling high-frequency details — textures and noise — that have no semantic relevance. The computation is spent on useless precision.

Conditional generative models like PixelCNN and WaveNet can technically model $p(x|c)$, but they must place a precise probability distribution over every pixel or sample. A vast number of parameters are wasted on reconstructing textures and high-frequency audio artifacts that carry essentially no information about high-level semantics.

The essence of CPC's insight is a compression argument. An image carries thousands of bits of pixel information, but its semantic label (e.g., "cat") requires roughly ten bits. Modeling $p(\text{image}|\text{context})$ directly means using a cannon to kill a mosquito.

### Slow Features: Why Predict Farther Ahead

CPC draws on a concept from computational neuroscience: slow features — properties that remain stable across many time steps. In speech, these include phonemes and speaker timbre. In images, object identity. In books, narrative structure.

Predicting the very next frame is too easy. Adjacent frames are nearly identical; a model can coast by copying. Forcing the model to predict twelve steps ahead (roughly 120ms in speech) causes local correlations to decay, so the model must latch onto the structure that persists across longer spans.

The paper validates this with two complementary measurements. Figure 3 tracks training signal — the fraction of cases where the positive sample outscores a negative — showing it drops from roughly 95% at 1 step to around 60% at 20 steps. Table 2 measures representation quality via linear phoneme classification accuracy on LibriSpeech. At 2 steps (20ms), accuracy is 28.5%. This rises steadily: 57.6% at 4 steps, 63.6% at 8 steps, peaking at 64.6% for 12 steps (120ms), then declining to 63.8% at 16 steps as the prediction horizon becomes too distant. Figure 3 measures how hard the training task is; Table 2 measures how useful the resulting representations are. Both point in the same direction: short horizons are easy but uninformative; very long horizons become too noisy to learn from.

## 2. Architecture

The pipeline runs: raw data passes through the encoder $g_{\text{enc}}$ to produce local representations $z_t$; these are aggregated by the autoregressive model $g_{\text{ar}}$ into a context vector $c_t$; the context is then used to contrastively predict future representations $z_{t+k}$.

### Encoder $g_{\text{enc}}$: Compression

The encoder's job is to discard local noise and high-frequency detail, keeping only what is structurally informative: $z_t = g_{\text{enc}}(x_t)$.

For speech, CPC uses a 5-layer strided CNN that takes 16kHz raw PCM waveforms directly (no MFCC preprocessing) and outputs a 512-dimensional vector every 10ms, with strides of [5, 4, 2, 2, 2] giving 160× total downsampling. For images, a ResNet-v2-101 without BatchNorm encodes each 64×64 patch into a 1024-dimensional vector, using the third residual block (not the final layer) followed by spatial mean-pooling — the intermediate layer preserves a 7×7 spatial grid with semantic information. For text, a 1D convolution followed by ReLU and mean-pooling produces a 2400-dimensional sentence vector, analogous to skip-thought vectors, with out-of-vocabulary words handled by a linear mapping learned from word2vec. For reinforcement learning, a CNN similar to the Nature DQN architecture encodes Atari frames.

The omission of BatchNorm for images is deliberate, though the paper gives only a one-line note: "We did not use Batch-Norm." SimCLR later provided the systematic explanation: in contrastive learning, samples within a batch serve as negatives for each other, and BatchNorm's cross-sample normalization creates information leakage — a form of cheating. MoCo addressed this later with Shuffle BN.

### Autoregressive Model $g_{\text{ar}}$: Aggregating History

The autoregressive component takes all past $z$ vectors and produces a single context vector: $c_t = g_{\text{ar}}(z_{\leq t})$. CPC uses a GRU for speech and text, and a PixelCNN with row-wise convolutions for images.

The distinction between $z_t$ and $c_t$ matters. A $z_t$ vector is a local snapshot with a limited receptive field — useful for tasks where temporal context is irrelevant, like image classification. A $c_t$ vector aggregates the full history and has a wide receptive field — necessary for tasks requiring temporal awareness, like speech recognition. When a global representation is needed, either can be spatially pooled.

## 3. The Density Ratio

CPC does not learn the generative model $p(x_{t+k} \mid c_t)$ — that would require conditional generation in the raw high-dimensional observation space. Instead it learns a density ratio:

$$f_k(x_{t+k}, c_t) \propto \frac{p(x_{t+k} \mid c_t)}{p(x_{t+k})}$$

The numerator $p(x \mid c)$ is the probability of seeing this future given the context; the denominator $p(x)$ is the probability of drawing this sample at random, ignoring any context. A ratio above 1 means the context provides predictive leverage — this is likely the real future. A ratio near 1 means the context adds nothing — this is probably noise or a fake sample.

To see the magnitude of the simplification, compare the three approaches side by side. MSE produces $\hat{x} \in \mathbb{R}^{H \times W \times 3}$, a 196,608-dimensional pixel reconstruction. Cross-entropy produces $p(y_i \mid x_{<i})$, a 256-way categorical distribution per token. CPC produces $f_k \in \mathbb{R}^+$, a single positive real number. The contrast is stark: $\hat{x}^* = \mathbb{E}[x \mid c]$ (MSE: 196,608-dimensional blurred average) versus $f_k^* \propto \frac{p(x \mid c)}{p(x)}$ (CPC: one-dimensional density ratio).

The paper notes explicitly that $f_k$ does not need to integrate to 1 — any positive real number works, which neatly sidesteps the expensive normalization sum that plagues probabilistic models. This is the same core insight as in [NCE](/blog/noise-contrastive-estimation): a density ratio need not be a probability; it only needs to separate real from fake.

### From Mutual Information to Density Ratio

The paper's Equation 1 is the definition of mutual information: $I(x; c) = \sum_{x, c} p(x, c) \log \frac{p(x \mid c)}{p(x)}$. The density ratio $\frac{p(x|c)}{p(x)}$ appears directly in this formula.

The derivation chain works as follows. The goal is $\max_{\theta} I(x_{t+k}; c_t)$: maximize the mutual information between the future observation and the context. The bottleneck is that $p(x|c)$ and $p(x)$ are incalculable in high-dimensional observation space. The strategy then shifts to $\max_{\theta} I(z_{t+k}; c_t)$: maximize mutual information in the lower-dimensional representation space instead. The justification comes from the Data Processing Inequality: $I(z; c) \leq I(x; c)$ since $z = g_{\text{enc}}(x)$ is a deterministic function that cannot create new information.

One might ask: why not compute $I(z; c)$ directly? Because $I(z; c) = \mathbb{E}[\log \frac{p(z \mid c)}{p(z)}]$ — the mutual information formula itself contains the density ratio. Even in the compressed representation space, $p(z \mid c)$ and $p(z)$ remain unknown. CPC borrows the NCE trick: instead of computing these probabilities explicitly, train a scoring function $f_k$ to estimate the ratio. InfoNCE automatically drives $f_k \propto \frac{p(z \mid c)}{p(z)}$, requiring only positive-negative comparisons and no normalization.

### The Scoring Function: Log-Bilinear

$$f_k(x_{t+k}, c_t) = \exp\!\big(z_{t+k}^T W_k \, c_t\big)$$

The components: $z_{t+k} = g_{\text{enc}}(x_{t+k})$ is the future frame's representation. The paper writes $f_k(x_{t+k}, c_t)$ rather than $f_k(z_{t+k}, c_t)$ — the external signature emphasizes dependence on the raw input, but internally the computation goes through $z$; since $g_{\text{enc}}$ is a deterministic mapping, no information is lost or gained. $c_t$ is the context vector from the GRU. $W_k$ is a separate linear transformation for each prediction step $k$ — $W_1$ handles short-term predictions while $W_{12}$ handles long-term ones. The exponential ensures $f_k > 0$ and amplifies differences.

The scoring function evaluates pairs — "does this context go with this sample?" — not individual samples. The paper states that any positive real score can be used; a nonlinear network or RNN could substitute for the bilinear form, but the linear version suffices because the encoder and autoregressive model already carry all the nonlinearity.

During InfoNCE training, $f_k$ is automatically driven toward $\frac{p(x|c_t)}{p(x)}$. A constant multiple does not matter: if $f_k = \alpha \cdot \frac{p(x|c)}{p(x)}$, the $\alpha$ cancels in the softmax. The designer chooses the shape of the scoring function; InfoNCE handles the calibration.

## 4. InfoNCE Loss

CPC is the paper that introduced InfoNCE, later inherited by SimCLR, MoCo, and the broader contrastive learning literature.

### Construction

Given context $c_t$, one positive sample, and $N-1$ negatives drawn independently from $p(x_{t+k})$, form the candidate set $X = \{x_1, ..., x_N\}$:

$$\mathcal{L}_N = -\mathbb{E}_X\Bigg[ \log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} \Bigg]$$

This is softmax cross-entropy. Let $\mathbf{s} = [f_k(x_1,c_t), \dots, f_k(x_N,c_t)]$ with positive index $\text{pos}$; then $p_i = \frac{e^{s_i}}{\sum_{j=1}^{N} e^{s_j}} = \text{softmax}(\mathbf{s})_i$ and $\mathcal{L}_N = -\log p_{\text{pos}}$.

The loss operates at three levels simultaneously. At the procedural level, it is an $N$-way cross-entropy: drive the positive sample's probability toward 1. At the statistical level, it estimates density ratios: the optimal $f_k$ is proportional to $p(x \mid c)/p(x)$. At the information-theoretic level, it measures the gap $\log N - I$: minimizing the loss maximizes mutual information. The multiple-choice format is the mechanism, the density ratio is the statistical engine, and mutual information is the ultimate objective. $\mathcal{L}_N$ is a mutual information gauge — it converts the abstract quantity $I(x;c)$ into a scalar that SGD can optimize.

### Proof: $f_k$ Converges to the Density Ratio

The paper's Section 2.3 states: "Optimizing this loss will result in $f_k(x_{t+k}, c_t)$ estimating the density ratio." Here is the proof in six steps with no gaps. This extends NCE's core theorem to the $N$-way classification setting (NCE corresponds to $N=2$).

Step 1: the data generating process. In the set $X = \{x_1, \dots, x_N\}$, exactly one sample is drawn from the true conditional $p(x \mid c_t)$; the remaining $N-1$ are drawn independently from the marginal $p(x)$. Let $d \in \{1,\dots,N\}$ index the positive sample. The event "$d=i$" means $x_i$ is the real future and all others are independent negatives.

Step 2: the joint probability under the hypothesis $d=i$ is $p(X \mid d=i, c_t) = p(x_i \mid c_t) \cdot \prod_{l \neq i} p(x_l)$.

Step 3: with uniform prior over the $N$ hypotheses, Bayes' rule gives the posterior: $p(d=i \mid X, c_t) = \frac{p(x_i \mid c_t) \prod_{l \neq i} p(x_l)}{\sum_{j=1}^{N} p(x_j \mid c_t) \prod_{l \neq j} p(x_l)}$.

Step 4: factor out the common term $\prod_{l=1}^{N} p(x_l)$. For the numerator, $p(x_i \mid c_t) \prod_{l \neq i} p(x_l) = \frac{p(x_i \mid c_t)}{p(x_i)} \cdot \prod_{l=1}^{N} p(x_l)$. After canceling the common product from numerator and denominator: $p(d=i \mid X, c_t) = \frac{\frac{p(x_i \mid c_t)}{p(x_i)}}{\sum_{j=1}^{N} \frac{p(x_j \mid c_t)}{p(x_j)}}$. Every term in both numerator and denominator is now a density ratio — the marginal $p(x_l)$ factors have vanished entirely.

Step 5: the model's softmax is $p_{\text{model}}(d=i \mid X, c_t) = \frac{f_k(x_i, c_t)}{\sum_j f_k(x_j, c_t)}$. Minimizing cross-entropy forces $p_{\text{model}} = p_{\text{optimal}}$, which implies $f_k(x_i, c_t) \propto \frac{p(x_i \mid c_t)}{p(x_i)}$. The $\propto$ notation means $f_k^*(x,c) = \alpha \cdot \frac{p(x \mid c)}{p(x)}$ for some $\alpha > 0$, which the softmax cancels.

Step 6 yields the conclusion:

$$
\boxed{f_k^*(x_{t+k}, c_t) \propto \frac{p(x_{t+k} \mid c_t)}{p(x_{t+k})}}
$$

This result is independent of $N-1$ — whether you use 10 negatives or 10,000, the optimal form of $f_k$ does not change.

### Mutual Information Lower Bound

The inequality $I(x_{t+k}; c_t) \geq \log N - \mathcal{L}_N^{\text{opt}}$ means a lower loss gives a tighter lower bound, approaching the true mutual information. Larger $N$ makes the bound tighter. CPC is mathematically equivalent to maximizing the mutual information between future observations and context.

### Negative Sampling as Implicit Information Filtering

The paper's speech experiments test five negative sampling strategies (Table 2). Mixed-speaker negatives (cross-sequence, the default) achieve 64.6% phoneme accuracy. Same-speaker negatives (cross-sequence, different sentences from the same person) reach 65.5% — the model is forced to ignore speaker identity and attend to phonetic content because timbre can no longer distinguish positive from negative. Mixed-speaker negatives excluding the current sequence drop to 57.3% due to a smaller negative pool. Same-speaker negatives excluding the current sequence produce 64.6%. Within-sequence negatives (other positions in the same utterance) achieve 65.2%.

Same-speaker negatives outperform mixed-speaker negatives on phoneme classification because when positive and negative samples share the same voice, acoustic features like pitch and timbre become useless for the classification task. The model is forced to rely on phonetic content alone. Negative sample selection is itself a form of implicit information filtering — you choose what the model should ignore by making those features unhelpful for the contrastive task.

## 5. CPC on Images

Images lack a natural temporal ordering, so CPC imposes one. The pipeline starts with a 256×256 image, randomly crops it to 300×300, then crops again to 256×256, applies a 50% horizontal flip, and converts to grayscale. The image is divided into a 7×7 grid of 64×64 patches with 32px overlap. Each patch is further cropped to a random 60×60 sub-region and padded back to 64×64. A ResNet-v2-101 encodes each patch into a 1024-dimensional vector, yielding a 7×7×1024 feature map. A PixelCNN processes these row by row and predicts representations for the next five rows.

Predicting five rows ahead is approximately equivalent to predicting the bottom half of an image from the top half. Only slow features — object identity, scene layout, spatial structure — can bridge that gap. Data augmentation (crop, flip, grayscale, sub-crop at the patch level) contributes substantially to the results.

## 6. Experimental Results

On LibriSpeech, the same CPC representation encodes both phonetic content and speaker identity — two entirely different information dimensions — within a single vector. For 41-class phoneme classification: random features achieve 27.6%, MFCC baselines reach 39.7%, CPC gets 64.6%, and supervised training reaches 74.6%. With a single-hidden-layer MLP replacing the linear probe, CPC climbs to 72.5%, approaching the supervised baseline of 74.6% — not all learned information is linearly accessible. For 251-way speaker classification: random achieves 1.87%, MFCC reaches 17.6%, CPC gets 97.4%, and supervised reaches 98.5%. The context window spans 20,480 samples, approximately 1.28 seconds at 16kHz.

On ImageNet, CPC achieves 48.7% top-1 and 73.6% top-5 accuracy. This is a domain-agnostic method — it uses no image-specific priors (unlike colorization, jigsaw puzzles, or rotation prediction) — yet it substantially surpasses all prior self-supervised approaches, whose best combination reached 69.3% top-5. Neuron visualizations from the paper show individual units that spontaneously become sensitive to animal heads, wheels, and water ripples, providing qualitative evidence of the representation quality.

For NLP, the encoder uses 1D convolution producing 2400-dimensional sentence vectors; a GRU with a 2400-dimensional hidden state predicts up to three future sentences. The paper notes that more complex encoders did not significantly improve results — a simple bag-of-words-style representation was already adequate. On the TREC question-type classification task, CPC reaches 96.8%, substantially outperforming Skip-thought (91.4%). Across other benchmarks (MR sentiment, CR product reviews, Subj subjectivity, MPQA opinion polarity), CPC performs comparably to the NLP-specific Skip-thought model despite being a general-purpose method.

For reinforcement learning, batched A2C with a CPC auxiliary loss (predicting future frame representations) significantly improves performance on four of five tested Atari games. The fifth game, Lasertag, showed no change — the authors speculate it is purely reactive and requires no memory.

CPC v2, published a year later, focuses specifically on vision: ResNet-161 with finer patches, stronger augmentation, and 128 GPUs pushes ImageNet top-1 from 48.7% to 71.5%, close to supervised ResNet performance. This confirms that contrastive predictive coding is fully viable for visual representation learning without relying on domain-specific pretext tasks like colorization or jigsaw puzzles.

## 7. NCE to CPC to InfoNCE to Contrastive Learning

CPC inherits several elements from [NCE](/blog/noise-contrastive-estimation). The density ratio formulation $\frac{p_\theta}{P_n}$ in NCE becomes $\frac{p(x_{t+k}\mid c_t)}{p(x_{t+k})}$ — shifting from discriminating words to discriminating futures. The unnormalized scoring approach carries over directly: $f_k$ can be any positive real, no normalization required. The contrastive loss evolves from NCE's binary log loss to InfoNCE's $N$-way cross-entropy. Negative sampling transitions from NCE's noise distribution to sampling from the batch or sequence. And the fundamental philosophy persists: do not reconstruct, just learn the density ratio.

CPC's innovations beyond NCE are significant. The predictive coding framework — encoder plus autoregressive model plus contrastive prediction — upgrades NCE from discrimination to prediction, enabling general-purpose representation learning. Predicting in representation space ($z$-space) separates compression (the encoder removes noise) from reasoning (the autoregressive model handles temporal structure). The slow feature insight — that predicting farther ahead forces the model to capture global structure — is novel and experimentally validated. And the framework proves cross-modal: the same architecture works for speech, images, text, and reinforcement learning without modality-specific modifications.

The lineage traces a clear path. NCE (2012) introduced density ratios and contrastive loss to solve language model normalization. CPC (2018) added predictive coding and representation-space contrast, yielding a universal unsupervised representation learning method with the InfoNCE loss and mutual information lower bound. SimCLR (2020) applied InfoNCE to instance discrimination. MoCo (2020) added momentum queues to scale negatives. BYOL (2020) showed negatives could be removed entirely. CLIP (2021) applied InfoNCE to image-text matching.

The through-line: NCE said "you do not need normalization; telling real from fake is enough." CPC generalized this to "you do not need to reconstruct the future; telling which future is real is enough." InfoNCE, invented by CPC, became the common language of the contrastive learning paradigm.

## Related Reading

- [Noise Contrastive Estimation](/blog/noise-contrastive-estimation) — the theoretical root: density ratios and the foundation of InfoNCE
- [SimCLR](https://arxiv.org/abs/2002.05709) — InfoNCE applied to visual contrastive learning
- [MoCo](https://arxiv.org/abs/1911.05722) — momentum queues for scaling up negative samples
- [BYOL](https://arxiv.org/abs/2006.07733) — removing negative samples entirely

## References

- van den Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
- Henaff et al. (2019). *Data-Efficient Image Recognition with Contrastive Predictive Coding* (CPC v2). arXiv:1905.09272.
- Reference blog (in Chinese): [Contrastive Predictive Coding (CPC) on Zhihu](https://zhuanlan.zhihu.com/p/317711322)
