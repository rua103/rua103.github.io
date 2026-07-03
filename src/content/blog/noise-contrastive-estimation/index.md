---
title: "Noise Contrastive Estimation"
description: "How turning 100,000-way classification into binary classification makes language model training tractable and sparked modern contrastive learning."
publishDate: 2026-06-22
tags:
  - optimization
  - contrastive-learning
  - language-modeling
  - estimation-theory
  - negative-sampling
series: contrastive-learning
seriesOrder: 1
draft: false
---

## 1. Why Normalization Hurts

Given a context $h$ and model parameters $\theta$, predicting the next word $w$ uses the softmax:

$$P_\theta^h(w) = \frac{\exp(s_\theta(w, h))}{Z_\theta^h} = \frac{u_\theta(w, h)}{Z_\theta^h}$$

where $s_\theta(w, h)$ is the logit (score) for word $w$, $u_\theta(w, h) = \exp(s_\theta(w, h))$ is the unnormalized score, and $Z_\theta^h = \sum_{w' \in V} \exp(s_\theta(w', h))$ is the partition function that sums over every word in the vocabulary $V$.

When $|V|$ ranges from $10^5$ to $10^6$, each training step must enumerate the entire vocabulary just to compute $Z_\theta^h$. The MLE gradient reveals the bottleneck:

$$\frac{\partial}{\partial \theta} \mathbb{E}_{w \sim P_d^h}[\log P_\theta^h(w)] = \underbrace{\mathbb{E}_{w \sim P_d^h}\Big[\frac{\partial s_\theta(w,h)}{\partial\theta}\Big]}_{\text{positive contribution from real words}} - \underbrace{\mathbb{E}_{w \sim P_\theta^h}\Big[\frac{\partial s_\theta(w,h)}{\partial\theta}\Big]}_{\text{must sum over all words!}}$$

The core problem: MLE demands the gradient of $Z_\theta^h$ at every step, and that gradient is an expectation over the model's own distribution $P_\theta^h$, which covers the entire vocabulary. Millions of training steps times a $10^5$-word sum is infeasible.

To see exactly how this gradient breaks down, start from the MLE objective: maximize the expected log-probability $\mathbb{E}_{w \sim P_d^h(w)}[\log P_\theta^h(w)]$. Write the log-probability as $\log P_\theta^h(w) = \log u_\theta(w,h) - \log Z_\theta^h$. The second term does not depend on which $w$ was drawn from the data, so it sits outside the expectation over $w$. Taking the derivative: $\frac{\partial}{\partial\theta} \mathbb{E}_{w \sim P_d^h}[\log P_\theta^h(w)] = \mathbb{E}_{w \sim P_d^h}[\frac{\partial}{\partial\theta} \log u_\theta(w,h)] - \frac{\partial}{\partial\theta} \log Z_\theta^h$.

The first term only touches words that actually appear in the training batch — cheap. The second term is the problem. Expanding it step by step: $\frac{\partial}{\partial\theta} \log Z_\theta^h = \frac{1}{Z_\theta^h} \cdot \frac{\partial Z_\theta^h}{\partial\theta}$ (chain rule on $\frac{d}{dx}\log f = f'/f$), then $= \frac{1}{Z_\theta^h} \cdot \frac{\partial}{\partial\theta} \sum_{w'} \exp(s_\theta(w', h))$ (expand $Z_\theta^h$'s definition), then $= \frac{1}{Z_\theta^h} \cdot \sum_{w'} \frac{\partial}{\partial\theta} \exp(s_\theta(w', h))$ (swap sum and derivative), then $= \frac{1}{Z_\theta^h} \cdot \sum_{w'} \exp(s_\theta(w', h)) \cdot \frac{\partial s_\theta(w', h)}{\partial\theta}$ (chain rule: $\frac{d}{dx}e^f = e^f \cdot f'$), then $= \sum_{w'} \frac{\exp(s_\theta(w', h))}{Z_\theta^h} \cdot \frac{\partial s_\theta(w', h)}{\partial\theta}$ (push $1/Z$ back into the sum), which is $= \mathbb{E}_{w' \sim P_\theta^h}[\frac{\partial s_\theta(w', h)}{\partial\theta}]$.

That last line is the killer: $\mathbb{E}_{w' \sim P_\theta^h}$ means every word in the vocabulary contributes to the gradient, weighted by the model's own softmax probability. Expanding it explicitly: $\mathbb{E}_{w' \sim P_\theta^h}[\frac{\partial s_\theta}{\partial\theta}] = P_\theta^h(\text{cat}) \cdot \frac{\partial s_\theta(\text{cat}, h)}{\partial\theta} + P_\theta^h(\text{dog}) \cdot \frac{\partial s_\theta(\text{dog}, h)}{\partial\theta} + P_\theta^h(\text{the}) \cdot \frac{\partial s_\theta(\text{the}, h)}{\partial\theta} + \cdots$. Each word needs one forward pass (computing $P_\theta^h(w')$) times one backward pass (computing $\frac{\partial s_\theta}{\partial\theta}$). For a 100,000-word vocabulary, each gradient descent step costs 100,000 forward-backward passes.

In plain terms: MLE pushes the score of the correct word up, but to prevent the model from cheating — giving every word an infinite score — it also pushes all word scores down, weighted by how likely the model currently thinks each word is. That "push all scores down" step runs over the whole vocabulary. NCE sidesteps this by reformulating the problem entirely.

## Notation and Probability Quick-Start

If the equations feel unfamiliar, the issue is likely probability notation, not NCE itself. Here is a minimal primer.

| Symbol | How to read it | Meaning | Example |
|:---|:---|:---|:---|
| $P(w)$ | "probability of $w$" | Probability a random variable takes the value $w$ | $P(\text{cat}) = 0.3$ means cat appears 30% of the time |
| $P(w \mid h)$ | "probability of $w$ given $h$" | Conditional probability: $w$'s probability once we know $h$ | Given the previous word is "the", how likely is the next word "cat"? |
| $w \sim P$ | "$w$ follows distribution $P$" | $w$ is sampled from the probability distribution $P$ | $w \sim P_d$ means $w$ comes from the data distribution |
| $\mathbb{E}_{w \sim P}[f(w)]$ | "expectation of $f(w)$ under $P$" | Weighted average, using $P$ as the weights | For details see the explanation below |
| $\sum_w$ | "sum over all $w$" | Add up over every possible value | $\sum_{w \in V}$ means summing over the whole vocabulary |
| $P_d^h(w)$ | data distribution | The true frequency of word $w$ given context $h$ in training data | How often "cat" really follows "the" in human text |
| $P_\theta^h(w)$ | model distribution | The model's predicted probability of $w$ given $h$ (with parameters $\theta$) | The softmax output |
| $P_n(w)$ | noise distribution | An arbitrary distribution we choose ourselves | A unigram frequency distribution (common words get higher probability) |
| $\frac{\partial}{\partial\theta}$ | "partial derivative with respect to $\theta$" | Gradient: how the function changes when $\theta$ moves | $\frac{\partial L}{\partial\theta}$ is the gradient of the loss |

The expectation $\mathbb{E}$ is simply a weighted average: $\mathbb{E}_{w \sim P}[f(w)] = \sum_{w} P(w) \cdot f(w)$. Take each possible value of $f(w)$, multiply it by how often it occurs under $P(w)$, and sum. If your vocabulary has three words and your model assigns $P(\text{cat})=0.5$, $P(\text{dog})=0.3$, $P(\text{the})=0.2$, with scores $s(\text{cat})=10$, $s(\text{dog})=5$, $s(\text{the})=2$, then $\mathbb{E}_{w \sim P}[s(w)] = 0.5 \times 10 + 0.3 \times 5 + 0.2 \times 2 = 6.9$. That is all there is to it.

The vertical bar in $P(w \mid h)$ means "under the condition that." $P(\text{rain}) = 0.3$ is the baseline; $P(\text{rain} \mid \text{dark clouds}) = 0.8$ is the updated belief once you have seen clouds. In language modeling, $P(\text{cat} \mid \text{the})$ is the probability of "cat" given that the preceding word was "the."

Bayes' rule is the central tool in NCE's derivation. Its essence is flipping conditions around: $P(A \mid B) = \frac{P(A, B)}{P(B)}$, which says: the probability of A once you know B is the fraction of times A and B occur together, divided by how often B occurs at all. If "cat and from real data" happens 5 times out of 100, and "cat" appears 20 times total (real or noise), then a cat-token is from real data with probability $0.05 / 0.20 = 0.25$. NCE applies exactly this logic: given a word $w$, work backwards to decide whether it came from real data or from noise.

## 2. Core Idea: Classification, Not Counting

NCE introduces a **noise distribution** $P_n(w)$ (a simple, fast-to-sample distribution like unigram frequency) and defines a binary classification problem:

| Label | Meaning | Sampling ratio |
| :--- | :--- | :--- |
| $D = 1$ | The word came from real data $P_d^h(w)$ | $\frac{1}{k+1}$ |
| $D = 0$ | The word came from noise $P_n(w)$ | $\frac{k}{k+1}$ |

where $k$ is the theoretical ratio of noise samples to real samples.

The derivation proceeds through Bayes' rule. The joint probability of drawing a word $w$ with label $D=1$ is $P^h(D=1, w) = \frac{1}{k+1} P_d^h(w)$: first pick the "real data" source (probability $1/(k+1)$), then sample $w$ from the data distribution. Similarly, $P^h(D=0, w) = \frac{k}{k+1} P_n(w)$. The marginal probability of seeing word $w$ at all (from either source) is $p^h(w) = P^h(D=1, w) + P^h(D=0, w) = \frac{1}{k+1}[P_d^h(w) + kP_n(w)]$.

Given a word $w$, what is the posterior probability that it came from real data?

$$P^h(D=1 \mid w, \theta) = \frac{P^h(D=1, w)}{p^h(w)} = \frac{\frac{1}{k+1} P_d^h(w)}{\frac{1}{k+1}[P_d^h(w) + kP_n(w)]} = \frac{P_d^h(w)}{P_d^h(w) + kP_n(w)}$$

We do not know $P_d^h(w)$ — that is the distribution we want to learn. So we substitute the model's prediction $P_\theta^h(w)$ in its place:

$$P^h(D=1 \mid w, \theta) = \frac{P_\theta^h(w)}{P_\theta^h(w) + kP_n(w)}$$

$$P^h(D=0 \mid w, \theta) = \frac{kP_n(w)}{P_\theta^h(w) + kP_n(w)}$$

The intuition: when $P_\theta^h(w)$ dominates $kP_n(w)$, the model believes this word is real; when the noise term dominates, the model leans toward fake. NCE trains the model to make $P_\theta^h(w)$ converge to the true $P_d^h(w)$ through this proxy classification task.

To turn this into a loss function, start from the standard binary log loss $J(\theta) = \mathbb{E}_{(x,y) \sim P_d}[y\log\sigma(h_\theta(x)) + (1-y)\log(1-\sigma(h_\theta(x)))]$. NCE substitutes $P^h(D|w,\theta)$ as the classifier's output and $D$ as the label: when $D=1$, the loss is $-\log P^h(D=1|w,\theta)$; when $D=0$, the loss is $-\log P^h(D=0|w,\theta)$. The total loss weights every sample according to its source.

Expanding the expectation over the full mixture distribution $p^h(w)$ and substituting the two conditional probabilities:

$$J^h(\theta) = \mathbb{E}_{w \sim P_d^h}\Big[\log \frac{P_\theta^h(w)}{P_\theta^h(w) + kP_n(w)}\Big] + k \cdot \mathbb{E}_{w \sim P_n}\Big[\log \frac{kP_n(w)}{P_\theta^h(w) + kP_n(w)}\Big]$$

The constant factor $1/(k+1)$ is dropped (it does not affect optimization). The first term encourages the model to assign high scores to real words; the second term, weighted by $k$ because there are $k$ times as many noise samples, encourages the model to assign low scores to noise words. Together they form a weighted binary classification log loss.

## 3. The Critical Result: NCE Approaches MLE When $k \to \infty$

In NCE we do not compute $Z_\theta^h$ explicitly. Instead we introduce a learnable parameter $c^h$ as a per-context scaling factor: $P_\theta^h(w) = P_{\theta^0}^h(w) \cdot \exp(c^h)$, where $P_{\theta^0}^h(w) = \exp(s_{\theta^0}(w, h))$ is the unnormalized score from the neural network $\theta^0$, and $\exp(c^h)$ approximates the normalization constant. The full parameter set is $\theta = \{\theta^0, c^h\}$.

And yet, the most remarkable property of NCE is that in practice you can simply fix $\exp(c^h) = 1$ and not learn it at all. The model will learn to self-normalize: the score magnitudes will automatically adjust so that $\sum_w \exp(s_\theta(w,h)) \approx 1$. This sounds like it should not work. If we are not learning the normalization factor, what stops the network from pushing all scores to infinity?

The answer is in the structure of the binary classification loss. Suppose the network goes haywire and assigns enormous $P_\theta^h(w)$ to every word, real or fake. In every term's denominator $P_\theta^h(w) + kP_n(w)$, the $P_\theta^h$ term dominates and the noise term $kP_n$ is washed out. For noise words, the loss becomes $\log\frac{kP_n}{P_\theta^h + kP_n} \approx \log\frac{\text{tiny}}{\text{huge}} \to -\infty$: the loss explodes. For real words, the loss becomes $\log\frac{P_\theta^h}{P_\theta^h + kP_n} \approx \log\frac{\text{huge}}{\text{huge}} \approx \log 1 = 0$: loss is zero, which cannot offset the explosion from the noise terms. Conversely, if the network pushes all scores toward zero, the real-word loss explodes while the noise-word loss sits at zero. Either extreme is punished. The only equilibrium is $\sum_w \exp(s_\theta) \approx 1$. The binary classification loss is a natural braking mechanism: push too high and the noise terms penalize you; push too low and the real terms penalize you. The math itself enforces normalization, no explicit partition function required.

### Gradient Derivation

Computing $\frac{\partial}{\partial\theta} J^h(\theta)$ gives the core insight. Using Leibniz rule to move the derivative inside the expectations:

$$\frac{\partial J^h}{\partial\theta} = \mathbb{E}_{w \sim P_d^h}\Big[\frac{\partial}{\partial\theta} \log \frac{P_\theta^h}{P_\theta^h + kP_n}\Big] + k \cdot \mathbb{E}_{w \sim P_n}\Big[\frac{\partial}{\partial\theta} \log \frac{kP_n}{P_\theta^h + kP_n}\Big]$$

For the first term, instead of attacking the fraction directly, rewrite it to avoid the quotient rule:

$$\frac{\partial}{\partial\theta} \log \frac{P_\theta^h}{P_\theta^h + kP_n} = -\frac{\partial}{\partial\theta} \log\Big(1 + \frac{kP_n}{P_\theta^h}\Big) = -\frac{1}{1 + \frac{kP_n}{P_\theta^h}} \cdot \Big(-\frac{kP_n}{(P_\theta^h)^2} \cdot \frac{\partial P_\theta^h}{\partial\theta}\Big)$$

After algebraic simplification: $= \frac{kP_n}{P_\theta^h + kP_n} \cdot \frac{1}{P_\theta^h} \cdot \frac{\partial P_\theta^h}{\partial\theta} = \frac{kP_n}{P_\theta^h + kP_n} \cdot \frac{\partial}{\partial\theta} \log P_\theta^h$.

The second term follows similarly: $\frac{\partial}{\partial\theta} \log \frac{kP_n}{P_\theta^h + kP_n} = -\frac{\partial}{\partial\theta} \log(1 + \frac{P_\theta^h}{kP_n}) = -\frac{P_\theta^h}{P_\theta^h + kP_n} \cdot \frac{\partial}{\partial\theta} \log P_\theta^h$.

Combining both expectations:

$$\frac{\partial J^h}{\partial\theta} = \mathbb{E}_{w \sim P_d^h}\Big[\frac{kP_n}{P_\theta^h + kP_n} \cdot \frac{\partial \log P_\theta^h}{\partial\theta}\Big] - k \cdot \mathbb{E}_{w \sim P_n}\Big[\frac{P_\theta^h}{P_\theta^h + kP_n} \cdot \frac{\partial \log P_\theta^h}{\partial\theta}\Big]$$

Writing these expectations as discrete sums reveals the structure more clearly:

$$\frac{\partial J^h}{\partial\theta} = \sum_w \frac{kP_n}{P_\theta^h + kP_n} \cdot \big(P_d^h(w) - P_\theta^h(w)\big) \cdot \frac{\partial \log P_\theta^h}{\partial\theta}$$

Now take the derivative with respect to $\theta^0$ (the network parameters, excluding the normalization parameter $c^h$). Since $\log P_\theta^h(w) = \log P_{\theta^0}^h(w) + c^h$ and $c^h$ does not depend on $\theta^0$, the $\frac{\partial \log P_\theta^h}{\partial\theta^0}$ term simplifies to $\frac{\partial \log P_{\theta^0}^h(w)}{\partial\theta^0}$:

$$\frac{\partial J^h}{\partial\theta^0} = \sum_w \frac{kP_n}{P_\theta^h + kP_n} \cdot \big(P_d^h(w) - P_\theta^h(w)\big) \cdot \frac{\partial \log P_{\theta^0}^h(w)}{\partial\theta^0}$$

Now the critical step: take $k \to \infty$. The weighting factor $\frac{kP_n}{P_\theta^h + kP_n} \to 1$ because $kP_n$ dominates $P_\theta^h$ in both numerator and denominator. The gradient becomes:

$$\frac{\partial J^h}{\partial\theta^0} \xrightarrow{k \to \infty} \sum_w \big(P_d^h(w) - P_\theta^h(w)\big) \cdot \frac{\partial \log P_{\theta^0}^h(w)}{\partial\theta^0}$$

This is exactly the MLE gradient. Recall from Section 1: the MLE gradient was $\sum_w (P_d^h - P_\theta^h) \cdot \frac{\partial s_\theta}{\partial\theta}$, and here $\frac{\partial \log P_{\theta^0}^h}{\partial\theta^0} = \frac{\partial s_{\theta^0}}{\partial\theta^0}$. The two expressions are identical.

So $\text{NCE gradient} \xrightarrow{k \to \infty} \text{MLE gradient}$. The crucial difference: in MLE, the sum over $w$ runs over the entire vocabulary (because $P_\theta^h(w)$ is the full softmax). In NCE, the sum only runs over the positive sample plus $k$ noise samples. The complexity drops from $O(|V|)$ to $O(k)$.

## 4. NCE in Practice: Sigmoid + Log Loss

The posterior probability $P^h(D=1 \mid w, \theta) = \frac{P_\theta^h(w)}{P_\theta^h(w) + kP_n(w)}$ can be rewritten in a much more practical form. Divide numerator and denominator by $P_\theta^h(w)$ to get $\frac{1}{1 + \frac{kP_n(w)}{P_\theta^h(w)}}$, then rewrite the fraction inside using $\exp(\log \cdot)$: $\frac{1}{1 + \exp(\log \frac{kP_n(w)}{P_\theta^h(w)})} = \frac{1}{1 + \exp(\log(kP_n(w)) - \log P_\theta^h(w))}$. Pulling out a negative sign gives $\frac{1}{1 + \exp(-[\log P_\theta^h(w) - \log(kP_n(w))])}$, which is exactly the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ with input $x = \log P_\theta^h(w) - \log(kP_n(w))$.

In practice we set $\exp(c^h) = 1$ (self-normalization), so $P_\theta^h(w) = \exp(s_{\theta^0}(w,h))$ and $\log P_\theta^h(w) = s_{\theta^0}(w,h)$. This yields:

$$P^h(D=1 \mid w, \theta) = \sigma\big(s_{\theta^0}(w, h) - \log(kP_n(w))\big)$$

Define $\Delta s = s_{\theta^0}(w, h) - \log(kP_n(w))$. This quantity asks a natural question: how much higher is the model's score for this word compared to its expected score if it were random noise? The larger the gap, the more likely the word is real.

The empirical loss function (following the original paper's recommendation of $m=1$ real sample and $n=k$ noise samples) becomes:

$$\widehat{J^h}(\theta) = \log \sigma\big(s_\theta(w_0, h) - \log(kP_n(w_0))\big) + \sum_{j=1}^{k} \log\big(1 - \sigma(s_\theta(w_j, h) - \log(kP_n(w_j)))\big)$$

The computation flow: take one real word $w_0$ (label = 1) and $k$ noise words (label = 0); for each word compute $\Delta s = s_\theta(w, h) - \log(kP_n(w))$; pass through sigmoid; compute binary cross-entropy; backpropagate.

## 5. A Common Confusion: $k$ vs $n$

Nearly every blog post and even TensorFlow's `tf.nn.nce_loss` gets this wrong. The two parameters are conceptually distinct.

$k$ is the theoretical ratio of noise samples to real samples. A larger $k$ brings NCE closer to MLE in the limit. $n$ is the actual number of noise samples drawn from $P_n$ for Monte Carlo estimation of the noise expectation. A larger $n$ gives a more accurate estimate of the expectation. Confusing them does not break anything in practice when $n$ is large enough, but conceptually they are independent design choices.

## 6. Limitations

NCE only speeds up training, not inference. At test time you still need to enumerate the vocabulary if you want to generate the next word — NCE does not bypass the partition function at inference. The noise distribution $P_n(w)$ matters: if it is too far from the true data distribution, the classifier never sees difficult negative examples and the learned model degrades. And NCE does not guarantee a properly normalized probability distribution (though self-normalization works well in practice, the output is not a strict probability).

## 7. From NCE to InfoNCE

NCE is the mathematical foundation of InfoNCE loss. InfoNCE generalizes the idea from language modeling to representation learning, making essentially one change: instead of $k+1$ independent binary classifications, it poses a single $(k+1)$-way classification.

In NCE, each negative sample is judged independently: "are you real or fake?" In InfoNCE, the question becomes "among these $k+1$ candidates, which one is the real positive?" The negatives compete with each other, and only one can win.

NCE's loss for a positive sample plus $k$ negatives: $J^h = \log\sigma(\Delta s_0) + \sum_{j=1}^k \log(1-\sigma(\Delta s_j))$. InfoNCE transforms this into a softmax over all $k+1$ samples: $\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s_\theta(x, x^+))}{\exp(s_\theta(x, x^+)) + \sum_{j=1}^{k} \exp(s_\theta(x, x_j^-))}$.

| | NCE | InfoNCE |
|:---|:---|:---|
| Problem | Language modeling (predict the next word) | Representation learning (classify positive vs. negative pairs) |
| Positive | $(h, w)$ from training data | A positive pair $(x, x^+)$ |
| Negatives | Sampled from $P_n(w)$ | Other samples in the batch or a memory bank |
| Classification | $k+1$ independent binary (Sigmoid) | One $(k+1)$-way classification (Softmax) |
| Normalization | Not needed (each Sigmoid is independent) | Required — denominator sums $\sum_{j=0}^k \exp(s_j)$ |
| Negative relationship | Independent | Competitive |

### Why "Info" NCE: From Loss to Mutual Information

This is InfoNCE's deepest theoretical contribution. Many sources give the conclusion without the derivation; here is the full chain.

Start with the InfoNCE loss: $\mathcal{L}_N = -\mathbb{E}_X [\log \frac{f_\theta(x, x^+)}{\sum_{j=1}^{N} f_\theta(x, x_j)}]$, where $x^+$ is the positive sample for $x$ and $x_2, ..., x_N$ are $N-1$ negatives drawn independently from the marginal $p(x^+)$. The scoring function $f_\theta$ is what the model learns — for instance, $\exp(\text{sim}/\tau)$ in SimCLR.

At optimality, the scoring function converges to a density ratio: $f^*(x, x^+) \propto \frac{p(x^+ \mid x)}{p(x^+)}$. The model's score estimates how much more likely $x^+$ is to appear when paired with $x$, compared to appearing randomly in the dataset. A high ratio means a strong association.

Substituting this optimal form back into the loss and noting that negative samples are drawn independently from $p(x^+)$ (so $\frac{p(x_j \mid x)}{p(x_j)} = \frac{p(x_j)}{p(x_j)} = 1$), we get:

$$\mathcal{L}_N^{\text{opt}} = -\mathbb{E}_X \Bigg[ \log \frac{ \frac{p(x^+|x)}{p(x^+)} }{ \frac{p(x^+|x)}{p(x^+)} + (N-1) } \Bigg] \approx -\mathbb{E}_X \Bigg[ \log \frac{p(x^+|x)}{p(x^+)} \Bigg] + \log N$$

where the approximation holds when $N$ is large and the density ratio is small relative to $N$. Now recall the definition of mutual information: $I(x; x^+) = \mathbb{E}_{(x, x^+) \sim p(x, x^+)}[\log \frac{p(x, x^+)}{p(x)p(x^+)}] = \mathbb{E}_X[\log \frac{p(x^+ \mid x)}{p(x^+)}]$. Mutual information measures how much uncertainty about $x^+$ is removed by knowing $x$.

This yields $\mathcal{L}_N^{\text{opt}} \approx -I(x; x^+) + \log N$, and since any trained loss is at least the optimal loss, $\mathcal{L}_N \geq \log(N-1) - I(x; x^+)$. Rearranging gives the useful bound:

$$I(x; x^+) \geq \log(N-1) - \mathcal{L}_N$$

The quantity $\log(N-1) - \mathcal{L}_N$ is a lower bound on the mutual information $I$. Minimizing the InfoNCE loss pushes this lower bound upward: the model is forced to preserve more shared information between positive pairs. A large $N$ tightens the bound (the $\log(N-1)$ term grows), which is why contrastive learning craves large numbers of negatives.

The narrative flow is worth tracing in full. InfoNCE sets up a game: among $N$ candidates, find the real positive. This game has a theoretical ceiling — the best any classifier can do is determined by the mutual information shared between positive pairs. When the model plays well (low loss), its representations retain a lot of mutual information. When it cannot tell positives from negatives (high loss), the representations have discarded most of that shared structure. Minimizing InfoNCE is not an arbitrary trick; it is mathematically equivalent to maximizing a lower bound on mutual information.

In SimCLR, a batch of $N$ images, each with two augmented views, produces $2N$ representations. For each positive pair $(z_i, z_j)$, the loss takes the form $\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$, where $\text{sim}(z_i, z_j)$ is typically cosine similarity and $\tau$ is a temperature parameter controlling the sharpness of the distribution. This is InfoNCE (also called NT-Xent loss): the denominator sums over all $2N-1$ other samples as negatives. InfoNCE has been adopted by SimCLR, MoCo, CPC, BYOL, and essentially all major contrastive learning methods.

### Why InfoNCE Outperforms NCE in Representation Learning

NCE was built for language modeling; InfoNCE was built for representation learning. When compared within the representation learning setting, InfoNCE has several structural advantages.

The key difference is negative competition. In NCE, each negative is judged independently — even a hard negative (one very similar to the positive) gets the same treatment as an easy one: just another "fake" label. In InfoNCE, all negatives share the same denominator. To win the softmax, the positive must not only score highly but must outrank every negative. A strong hard negative steals probability mass from the positive, producing a stronger gradient.

The gradient itself reveals this property. For each negative $x_j^-$, $\frac{\partial \mathcal{L}}{\partial s(x, x_j^-)} = \frac{\exp(s(x, x_j^-) / \tau)}{\sum_{k} \exp(s(x, x_k^-) / \tau)}$. The penalty on a negative is proportional to the probability the model mistakenly assigns to it being the positive. The more a negative resembles the positive, the higher its softmax probability, and the harder the gradient hits it. The loss is doing automatic hard negative mining — no manual filtering needed. NCE's sigmoid has no equivalent mechanism: all negatives receive equal gradient weight regardless of difficulty.

InfoNCE also removes the need for an external noise distribution. NCE requires choosing $P_n(w)$ by hand; a poor choice degrades performance. InfoNCE uses the batch itself as the negative pool, with zero overhead. MoCo extends this further with a momentum encoder maintaining a large negative queue for when the batch alone is too small.

The temperature parameter $\tau$ in the softmax provides a tunable difficulty knob: $\tau \to 0$ concentrates all gradient signal on the hardest negative; $\tau \to \infty$ treats all negatives equally. SimCLR's $\tau = 0.1$ was found through tuning. NCE's sigmoid has no comparable control.

And as shown in the derivation above, InfoNCE provides an information-theoretic guarantee: $\mathcal{L}_{\text{InfoNCE}} \geq \log(k) - I(x; x^+)$. Minimizing the loss is mathematically equivalent to maximizing the mutual information between positive pairs while minimizing it for negative pairs. NCE cannot make this claim.

In short, replacing independent sigmoids with a shared softmax denominator triggers a cascade of consequences — negative competition, automatic hard negative mining, mutual information interpretation, and temperature control. NCE's binary classifier is solving an easier problem, and in representation learning, that produces a less tightly organized representation space.

## Related Reading

- [Contrastive Predictive Coding](/blog/contrastive-predictive-coding) — the paper that proposed InfoNCE and extended NCE to representation learning
- [SimCLR](https://arxiv.org/abs/2002.05709) — InfoNCE applied to visual contrastive learning
- [MoCo](https://arxiv.org/abs/1911.05722) — momentum queue for scaling negative samples
- [BYOL](https://arxiv.org/abs/2006.07733) — contrastive learning without negative samples

## References

- [Lei Mao's Blog: Noise Contrastive Estimation](https://leimao.github.io/article/Noise-Contrastive-Estimation/)
- Gutmann & Hyvarinen (2012). *Noise-Contrastive Estimation of Unnormalized Statistical Models*
- Mnih & Teh (2012). *A Fast and Simple Algorithm for Training Neural Probabilistic Language Models*
- Ma & Collins (2018). *Noise Contrastive Estimation and Negative Sampling for Conditional Models: Consistency and Statistical Efficiency*
