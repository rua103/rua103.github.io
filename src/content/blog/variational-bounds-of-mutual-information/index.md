---
title: "On Variational Bounds of Mutual Information"
description: "A unified framework connecting MINE, InfoNCE, NWJ, and TUBA — all variational MI estimators with their derivation relationships and bias-variance tradeoffs."
publishDate: 2026-06-25
tags:
  - information-theory
  - mutual-information
  - variational-inference
  - contrastive-learning
  - representation-learning
  - self-supervised-learning
series: contrastive-learning
seriesOrder: 3
draft: false
---

> One-sentence summary: All variational MI estimators — MINE, InfoNCE, NWJ, TUBA — fit into a single framework. The paper clarifies their derivation relationships and bias-variance tradeoffs.

## 1. Starting Point: Two Equivalent Forms of Mutual Information

$$
I(X; Y) = \mathbb{E}_{p(x,y)}\left[\log \frac{p(x \mid y)}{p(x)}\right]
= \mathbb{E}_{p(x,y)}\left[\log \frac{p(y \mid x)}{p(y)}\right]
\tag{1}
$$

The two expressions are perfectly symmetric, conditioning on $X$ or $Y$ respectively. The derivations below will pick whichever side is more convenient.

## 2. Normalized Bounds: Using Proper Probability Distributions as Stand-Ins

Why derive these bounds at all? Equation (1) defines mutual information, but the true $p(y \mid x)$ and $p(y)$ cannot be computed in high-dimensional spaces. The standard variational inference move is: introduce a computable substitute distribution (a variational distribution), turning an incalculable quantity into an optimizable upper or lower bound.

"Normalized" means the variational distribution itself is a valid, normalized probability distribution ($\int q = 1$). This is the most direct approach — you cannot compute $p(y)$, so you replace it with some $q(y)$.

The next two subsections are symmetric: Section 2.1 introduces $q(y)$ to replace $p(y)$, yielding an upper bound. Section 2.2 introduces $q(x \mid y)$ to replace $p(x \mid y)$, yielding a lower bound.

### 2.1 Upper Bound (VAE-Style)

The goal is a computable upper bound. The strategy: replace the intractable $p(y)$ with $q(y)$.

**Step 1.** Multiply and divide by $q(y)$ inside the log:

$$
I(X; Y) = \mathbb{E}_{p(x,y)}\left[\log \frac{p(y \mid x)}{p(y)}\right]
= \mathbb{E}_{p(x,y)}\left[\log \left( \frac{p(y \mid x)}{q(y)} \cdot \frac{q(y)}{p(y)} \right)\right]
$$

**Step 2.** Split into two terms:

$$
= \underbrace{\mathbb{E}_{p(x,y)}\left[\log \frac{p(y \mid x)}{q(y)}\right]}_{\text{term 1}}
+ \underbrace{\mathbb{E}_{p(x,y)}\left[\log \frac{q(y)}{p(y)}\right]}_{\text{term 2}}
$$

**Step 3.** Simplify each term.

For ①, expand the joint expectation:

$$
\begin{aligned}
\text{①} &= \iint p(x,y) \log \frac{p(y \mid x)}{q(y)} \,dx\,dy \\
&= \iint p(x) p(y \mid x) \log \frac{p(y \mid x)}{q(y)} \,dx\,dy \\
&= \int p(x) \underbrace{\left[\int p(y \mid x) \log \frac{p(y \mid x)}{q(y)} \,dy\right]}_{D_{KL}(p(y \mid x) \parallel q(y))} dx \\
&= \mathbb{E}_{p(x)}\left[D_{KL}(p(y \mid x) \parallel q(y))\right]
\end{aligned}
$$

For ②, since $p(x,y) = p(x \mid y) p(y)$, the joint expectation over $(x,y)$ can be integrated over $x$ first:

$$
\begin{aligned}
\text{②} &= \iint p(x,y) \log \frac{q(y)}{p(y)} \,dx\,dy \\
&= \int \left[\int p(x \mid y) \,dx\right] p(y) \log \frac{q(y)}{p(y)} \,dy \\
&= \int p(y) \log \frac{q(y)}{p(y)} \,dy \\
&= -D_{KL}(p(y) \parallel q(y))
\end{aligned}
$$

**Step 4.** Combine:

$$
I(X; Y) = \mathbb{E}_{p(x)}\left[D_{KL}(p(y \mid x) \parallel q(y))\right] - D_{KL}(p(y) \parallel q(y))
$$

**Step 5.** Since $D_{KL} \geq 0$, dropping the second term yields an upper bound:

$$
\boxed{I(X; Y) \leq \mathbb{E}_{p(x)}\left[D_{KL}(p(y \mid x) \parallel q(y))\right]}
$$

Equality holds when $q(y) = p(y)$. The literature often calls this upper bound the rate $R$ (Alemi et al., 2017), used as a regularizer that limits representation capacity in VAE, $\beta$-VAE, and VIB. It is not the same as the ELBO — the ELBO has an additional reconstruction term.

### 2.2 Lower Bound: Barber-Agakov (IBA, 2003)

Symmetrically, replace $p(x \mid y)$ with $q(x \mid y)$.

**Step 1.** Start from the $p(x \mid y)$ side, multiply and divide by $q(x \mid y)$:

$$
I = \mathbb{E}_{p(x,y)}\left[\log \frac{p(x \mid y)}{p(x)}\right]
= \mathbb{E}_{p(x,y)}\left[\log \left( \frac{q(x \mid y)}{p(x)} \cdot \frac{p(x \mid y)}{q(x \mid y)} \right)\right]
$$

**Step 2.** Split into two terms:

$$
= \mathbb{E}_{p(x,y)}\left[\log \frac{q(x \mid y)}{p(x)}\right]
+ \mathbb{E}_{p(x,y)}\left[\log \frac{p(x \mid y)}{q(x \mid y)}\right]
$$

**Step 3.** Expand the log in the first term:

$$
\mathbb{E}_{p(x,y)}\left[\log \frac{q(x \mid y)}{p(x)}\right]
= \mathbb{E}_{p(x,y)}[\log q(x \mid y)] - \mathbb{E}_{p(x,y)}[\log p(x)]
$$

Denote $h(X) = -\mathbb{E}_{p(x,y)}[\log p(x)]$ as the differential entropy of $X$, giving $\mathbb{E}_{p(x,y)}[\log q(x \mid y)] + h(X)$.

**Step 4.** The second term is a KL divergence:

$$
\mathbb{E}_{p(x,y)}\left[\log \frac{p(x \mid y)}{q(x \mid y)}\right]
= \mathbb{E}_{p(y)}\left[D_{KL}(p(x \mid y) \parallel q(x \mid y))\right] \geq 0
$$

**Step 5.** Since KL $\geq 0$, dropping it gives a lower bound:

$$
\boxed{I(X; Y) \geq \mathbb{E}_{p(x,y)}[\log q(x \mid y)] + h(X) \triangleq I_{BA}}
$$

Two problems arise: (1) $h(X)$ is incalculable because $p(x)$ is unknown; (2) modeling $q(x \mid y)$ is hard when $x$ is high-dimensional data.

## 3. Unnormalized Lower Bounds: Using Energy-Based Functions as Stand-Ins

Recall the two problems with IBA:

1. $h(X)$ is incalculable — it contains $p(x)$, which is unknown.
2. $q(x \mid y)$ is hard to model — $x$ might be an image, and having a neural network output a normalized probability distribution over images is extremely difficult.

The unnormalized bounds approach: instead of directly modeling $q(x \mid y)$, define it indirectly through an energy function (a critic) $f(x,y)$. This sidesteps the difficulty of outputting a valid probability distribution.

But there is a more important question: how to also get rid of $h(X)$? The answer lies in the design of equation (2).

### 3.1 Key Construction: Energy-Based Variational Distribution

The authors design the variational distribution as follows:

$$
\boxed{q(x \mid y) = \frac{p(x)}{Z(y)} e^{f(x,y)}} \tag{2}
$$

Each component has a specific design purpose:

**$f(x, y)$ — the critic function.** An arbitrary neural network that takes $(x, y)$ as input and outputs a scalar. It only needs to score each pair $(x,y)$ — high scores indicate the pair is more likely from the joint distribution, low scores indicate it is more likely from the product of marginals. Modeling a scalar function is far easier than modeling a high-dimensional probability distribution.

**$p(x)$ — explicitly multiplied in.** This is deliberate. When we substitute this into IBA, $\log q(x \mid y)$ expands to include $\log p(x)$, which exactly cancels with $h(X) = -\mathbb{E}[\log p(x)]$. In other words, $p(x)$ is inserted specifically to "eat" the $h(X)$ term.

Do we need the density value of $p(x)$ in practice? No. After substituting into IBA, $\log p(x)$ cancels out in the derivation. The final objective $I_{UBA} = \mathbb{E}[f] - \mathbb{E}[\log \mathbb{E}[e^f]]$ contains no $p(x)$ whatsoever. Training only requires sampling $x$ from the dataset (to estimate outer expectations and $\mathbb{E}_{p(x)}[e^f]$), never the probability density of $p(x)$.

**$Z(y) = \mathbb{E}_{p(x)}[e^{f(x,y)}]$ — the partition function.** The numerator $p(x) e^{f(x,y)}$ does not guarantee integration to 1 over $x$, so dividing by $Z(y)$ enforces normalization: $\int q(x \mid y)\, dx = \frac{1}{Z(y)}\int p(x) e^{f(x,y)} dx = \frac{Z(y)}{Z(y)} = 1$.

To summarize the construction in one sentence: use an easy-to-model scalar critic $f$ in place of a hard-to-model high-dimensional distribution $q$, while inserting $p(x)$ to cancel $h(X)$ later — at the cost of introducing a new intractable term $\log Z(y)$.

Now we substitute into IBA and watch each design intention play out step by step.

### 3.2 IUBA: Substituting the Energy-Based Form into IBA

**Step 1.** Substitute (2) into $I_{BA}$:

$$
I_{UBA} = \mathbb{E}_{p(x,y)}[\log q(x \mid y)] + h(X)
$$

First compute $\log q(x \mid y)$:

$$
\log q(x \mid y) = \log \frac{p(x)}{Z(y)} e^{f(x,y)}
= \log p(x) - \log Z(y) + f(x,y)
$$

**Step 2.** Substitute:

$$
I_{UBA} = \mathbb{E}_{p(x,y)}\left[\log p(x) - \log Z(y) + f(x,y)\right] + h(X)
$$

Expand:

$$
= \underbrace{\mathbb{E}_{p(x,y)}[\log p(x)]}_{-h(X)} - \mathbb{E}_{p(x,y)}[\log Z(y)] + \mathbb{E}_{p(x,y)}[f(x,y)] + h(X)
$$

**Step 3.** $\mathbb{E}_{p(x,y)}[\log p(x)] = -h(X)$ and the outer $+h(X)$ cancel exactly:

$$
I_{UBA} = \mathbb{E}_{p(x,y)}[f(x,y)] - \mathbb{E}_{p(y)}[\log Z(y)]
$$

Expand $Z(y)$:

$$
\boxed{I_{UBA} = \mathbb{E}_{p(x,y)}[f(x,y)] - \mathbb{E}_{p(y)}\left[\log \mathbb{E}_{p(x)}[e^{f(x,y)}]\right]} \tag{3}
$$

$h(X)$ is gone. But $\log \mathbb{E}[e^f]$ remains intractable — the expectation sits inside the log, so Monte Carlo estimation cannot be unbiased.

What does unbiased Monte Carlo estimation mean? Monte Carlo methods approximate expectations with sample means: $\mathbb{E}[g(x)] \approx \frac{1}{n}\sum_i g(x_i)$. As $n \to \infty$, the approximation converges to the true value, and for any finite $n$, the expected value of the estimator equals $\mathbb{E}[g(x)]$ — this is what "unbiased" means.

For the second term of $I_{UBA}$, $\mathbb{E}_{p(y)}[\log \mathbb{E}_{p(x)}[e^f]]$: the inner expectation $\mathbb{E}_{p(x)}[e^f]$ can be estimated unbiasedly with $\frac{1}{n}\sum_i e^{f(x_i,y)}$, but the outer $\log$ is nonlinear. Jensen's inequality tells us $\mathbb{E}[\log(\text{estimate})] \neq \log(\mathbb{E}[\text{estimate}])$, so plugging the sample mean into the log — $\log(\frac{1}{n}\sum_i e^{f_i})$ — produces a biased estimator.

The rest of Section 3 is essentially about: using various inequalities to pull the log out from around the expectation.

**What is the optimal critic?** Set $I_{UBA}$ to achieve equality when $q(x \mid y) = p(x \mid y)$ (recovering IBA's tightness condition):

$$
\begin{aligned}
q(x \mid y) &= \frac{p(x)}{Z(y)} e^{f(x,y)} = p(x \mid y) = \frac{p(x,y)}{p(y)} \\
e^{f(x,y)} &= \frac{p(x,y)}{p(x)} \cdot \frac{Z(y)}{p(y)} = p(y \mid x) \cdot \frac{Z(y)}{p(y)} \\
f^*(x,y) &= \log p(y \mid x) + \log \frac{Z(y)}{p(y)}
\end{aligned}
$$

$$
\boxed{f^*(x,y) = \log p(y \mid x) + c(y)}
$$

where $c(y) = \log \frac{Z(y)}{p(y)}$ is a function of $y$ alone.

### 3.3 IDV (Donsker-Varadhan) / MINE

The problem: $\log \mathbb{E}[e^f]$ is incalculable.

The strategy: use Jensen's inequality to move it outside the log. Since $\log$ is concave, $\mathbb{E}[\log \cdot] \leq \log \mathbb{E}[\cdot]$.

**Step 1.** Apply Jensen to the second term of $I_{UBA}$:

$$
\mathbb{E}_{p(y)}\left[\log \mathbb{E}_{p(x)}[e^{f(x,y)}]\right] \leq \log \mathbb{E}_{p(y)}\left[\mathbb{E}_{p(x)}[e^{f(x,y)}]\right]
$$

**Step 2.** Flip the inequality direction (since we subtract this term):

$$
I_{UBA} = \mathbb{E}_{p(x,y)}[f] - \mathbb{E}_{p(y)}[\log Z(y)]
\geq \mathbb{E}_{p(x,y)}[f] - \log \mathbb{E}_{p(y)}[Z(y)]
$$

**Step 3.** Merge into a joint expectation: $\mathbb{E}_{p(y)}[\mathbb{E}_{p(x)}[e^f]] = \mathbb{E}_{p(x)p(y)}[e^f]$:

$$
\boxed{I_{DV} = \mathbb{E}_{p(x,y)}[f(x,y)] - \log \mathbb{E}_{p(x)p(y)}[e^{f(x,y)}]} \tag{4}
$$

This is the Donsker-Varadhan bound used by MINE. Because of Jensen, $I_{DV} \leq I_{UBA} \leq I$. On paper, it is a valid lower bound.

But MINE in practice is not a strict bound. The paper's argument has two layers. First, MINE in practice "reverses" the Jensen direction: the original $I_{DV} = \mathbb{E}[f] - \log \mathbb{E}[e^f]$ is a lower bound on $I_{UBA}$ (because $\mathbb{E}[\log Z] \leq \log \mathbb{E}[Z]$). But the Monte Carlo estimate of $\log \mathbb{E}[e^f]$ effectively computes an approximation of $\mathbb{E}[f]$ rather than $\log \mathbb{E}[e^f]$, making the optimization target an upper bound on $I_{UBA}$. Second, layered on top of this is the finite-sample error from mini-batches. The final value is neither an upper nor a lower bound on the true MI.

From a gradient perspective: with mini-batch gradient estimates, the nonlinearity of $\log$ means the expected gradient of each batch is not equal to the true gradient. The optimizer is really optimizing $\mathbb{E}_{\text{batch}}[\log(\frac{1}{B}\sum e^f)]$ rather than the theoretically required $\log \mathbb{E}_{\text{batch}}[\frac{1}{B}\sum e^f]$. The Jensen directions of these two objectives are opposite.

Can MINE still be used? As an MI estimator (when you want a number), it is unreliable — you cannot take MINE's output of "3.2 nats" and claim the mutual information is at least 3.2 nats, because that 3.2 could be above or below the true MI. As a training objective (maximizing MI to learn representations), it can work. The gradient direction, though not a rigorous bound, still pushes $f$ to distinguish joint samples from marginal samples, and representation learning results in practice are decent.

The paper's contribution lies in diagnosing the root cause and then providing bounds that genuinely survive Monte Carlo estimation — ITUBA and InfoNCE. All subsequent work revolves around this goal.

### 3.4 ITUBA: A Truly Tractable Bound

The problem: $I_{DV}$ from Jensen still contains $\log \mathbb{E}[e^f]$, and its Monte Carlo estimate is not a bound.

The strategy: use a different inequality. For all $x, a > 0$, we have $\log x \leq \frac{x}{a} + \log a - 1$.

**Step 1.** Let $x = \mathbb{E}_{p(x)}[e^{f(x,y)}] = Z(y)$ and $a = a(y)$ (a positive function to be learned):

$$
\log Z(y) \leq \frac{Z(y)}{a(y)} + \log a(y) - 1
$$

Equality holds when $a(y) = Z(y)$.

**Step 2.** Substitute into $I_{UBA}$:

$$
\begin{aligned}
I_{UBA} &= \mathbb{E}_{p(x,y)}[f] - \mathbb{E}_{p(y)}[\log Z(y)] \\
&\geq \mathbb{E}_{p(x,y)}[f] - \mathbb{E}_{p(y)}\left[\frac{Z(y)}{a(y)} + \log a(y) - 1\right]
\end{aligned}
$$

**Step 3.** Expand $Z(y) = \mathbb{E}_{p(x)}[e^{f(x,y)}]$:

$$
\boxed{I_{TUBA} = \mathbb{E}_{p(x,y)}[f] - \mathbb{E}_{p(y)}\left[\frac{\mathbb{E}_{p(x)}[e^{f(x,y)}]}{a(y)} + \log a(y) - 1\right]} \tag{5}
$$

There is no more $\log \mathbb{E}[\cdot]$. All expectations are on the outside and can be estimated unbiasedly with Monte Carlo. The cost is needing to learn an additional network $a(y)$.

### 3.5 INWJ (Nguyen-Wainwright-Jordan, 2010)

The strategy: do not want to learn $a(y)$? Set it to a constant.

**Step 1.** Set $a(y) = e$ (chosen to simplify the algebra):

$$
I_{TUBA} = \mathbb{E}_{p(x,y)}[f] - \mathbb{E}_{p(y)}\left[\frac{\mathbb{E}_{p(x)}[e^{f}]}{e} + \log e - 1\right]
$$

$\log e - 1 = 0$, which drops out:

$$
= \mathbb{E}_{p(x,y)}[f] - \frac{1}{e} \mathbb{E}_{p(y)}\left[\mathbb{E}_{p(x)}[e^{f}]\right]
$$

**Step 2.** Merge the double expectation $\mathbb{E}_{p(y)}\mathbb{E}_{p(x)} = \mathbb{E}_{p(x)p(y)}$:

$$
\boxed{I_{NWJ} = \mathbb{E}_{p(x,y)}[f(x,y)] - e^{-1}\, \mathbb{E}_{p(x)p(y)}[e^{f(x,y)}]} \tag{6}
$$

Also known as the f-GAN bound or MINE-f. No $a(y)$ is needed, but the critic must self-normalize.

**Optimal critic.** Via variational calculus:

$$
\frac{\delta I_{NWJ}}{\delta f} = p(x,y) - e^{-1} p(x) p(y) e^{f(x,y)} = 0
$$

$$
e^{f^*(x,y)} = e \cdot \frac{p(x,y)}{p(x)p(y)} = e \cdot \frac{p(x \mid y)}{p(x)}
$$

$$
\boxed{f^*(x,y) = 1 + \log \frac{p(x \mid y)}{p(x)}} \tag{7}
$$

### 3.6 $I_{JS}$: Training a Critic with JS Divergence (Practical Variant)

Training a critic with INWJ's own objective can be unstable — the $e^f$ term has high variance. The paper proposes an engineering compromise called $I_{JS}$:

First, train the critic using Jensen-Shannon divergence (as in GANs / Hjelm et al., 2018). This trains a binary classifier to distinguish joint samples from marginal samples, yielding a log density ratio estimate $V(x,y) \approx \log \frac{p(x,y)}{p(x)p(y)}$. JS training is far more stable than INWJ because it avoids the exponential $e^f$ term.

Second, evaluate MI by plugging $V$ into the INWJ formula:

$$
I_{JS} = 1 + \mathbb{E}_{p(x,y)}[V(x,y)] - \mathbb{E}_{p(x)p(y)}[e^{V(x,y)}]
$$

The key point: training and evaluation use different objectives — JS for training (stable), INWJ for evaluation (a valid MI lower bound). The paper's dSprites experiments actually use $I_{JS}$.

## 4. Comparing the Unnormalized Bounds

| Bound | Derivation Technique | No $\log\mathbb{E}$ Nesting | Still a Bound After MC | Bias | Variance |
|---|---|---|---|---|---|
| IUBA | Substitute energy-based form | No | — | — | — |
| IDV (MINE) | Jensen lower bound | Yes | No | Low | High |
| ITUBA | $\log x \leq x/a + \log a -1$ | Yes | Yes | Low | High |
| INWJ (f-GAN) | $a(y) = e$ special case | Yes | Yes | Unbiased* | High |

Under the optimal critic, INWJ is an unbiased (not low-bias) estimator — see the paper's Figure 3. ITUBA and INWJ/IDV are all single-sample unnormalized bounds, which the paper groups together as "high variance" methods. Learning $a(y)$ in ITUBA helps but does not eliminate the problem. ITUBA's variance is typically somewhat lower than INWJ, but both fall in the high-variance category.

The core tension: these single-sample bounds — which use only one $x$ to estimate $Z(y)$ per evaluation — all suffer from high variance, because $Z(y) = \mathbb{E}_{p(x)}[e^{f}]$ is an estimate that depends heavily on the tails of the $x$ distribution.

The solution direction: use multiple samples within a batch to share the partition function estimate.

## 5. Multi-Sample Bounds: Reducing Variance with Batches

### 5.1 The Idea

Instead of using 1 sample $x$ to estimate $Z(y)$, use $K$ samples: $Z(y) \approx \frac{1}{K}\sum_{i=1}^K e^{f(x_i, y)}$. The cost is introducing bias (the estimator itself is no longer an unbiased estimate of $Z(y)$), but the variance drops substantially.

### 5.2 Deriving InfoNCE from INWJ

**Step 0.** Setup. We have one pair $(x_1, y)$ from $p(x_1)p(y \mid x_1)$, and $K-1$ independent samples $x_{2:K} \sim r^{K-1}(x)$. In practice $r$ is usually the data distribution $p(x)$ itself. Denote $x_{1:K} = (x_1, x_2, \ldots, x_K)$.

Since $x_{2:K}$ is independent of $(x_1, y)$, we can "absorb" the extra negative samples into the mutual information without changing its true value. This happens in two conceptual steps.

Step A: wrap an expectation around it, which does not change the value.

$$
I(X_1; Y) = \mathbb{E}_{r^{K-1}(x_{2:K})}[I(X_1; Y)]
$$

$I(X_1; Y)$ has nothing to do with $x_{2:K}$. Taking the expectation over $x_{2:K}$, the term $I(X_1; Y)$ is a constant pulled outside — the expectation of a constant is the constant. It is like your exam score depending only on how much you studied, not on what you ate for dinner; the average of your exam score over all possible dinners is still your exam score.

Step B: "swallow" the expectation into the mutual information.

$$
\mathbb{E}_{r^{K-1}(x_{2:K})}[I(X_1; Y)] = I(X_1, X_{2:K}; Y)
$$

By the chain rule of mutual information:

$$
I(X_1, X_{2:K}; Y) = I(X_1; Y) + \underbrace{I(X_{2:K}; Y \mid X_1)}_{=0}
$$

Since $X_{2:K}$ is completely independent of $(X_1, Y)$, knowing $X_1$ does not create any information overlap between $X_{2:K}$ and $Y$. The conditional mutual information is zero. Therefore:

$$
\boxed{I(X_1; Y) = I(X_1, X_{2:K}; Y)} \tag{8}
$$

In one sentence: the extra negative samples are independent of $(X_1, Y)$, and through the chain rule they can be absorbed into $I(X_1, X_{2:K}; Y)$ without changing the true MI. The benefit is enormous — we can now apply INWJ to $I(X_1, X_{2:K}; Y)$ and let these "extra samples" help estimate the partition function.

**Step 1.** Construct a critic that considers all $K$ samples:

$$
f(x_{1:K}, y) = 1 + \log \frac{e^{f(x_1, y)}}{a(y; x_{1:K})} \tag{9}
$$

where $a(y; x_{1:K})$ is a substitute for the partition function estimated from $K$ samples.

**Step 2.** Substitute (9) into the INWJ formula $I_{NWJ} = \mathbb{E}_{p}[f] - e^{-1}\mathbb{E}_{p \otimes p}[e^{f}]$:

$$
I \geq \mathbb{E}_{p(x_{1:K})p(y \mid x_1)}\left[1 + \log \frac{e^{f(x_1, y)}}{a(y; x_{1:K})}\right]
- e^{-1}\, \mathbb{E}_{p(x_{1:K})p(y)}\left[e^{1 + \log \frac{e^{f(x_1, y)}}{a(y; x_{1:K})}}\right]
$$

**Step 3.** Simplify the second term. Note $e^{1 + \log \frac{e^f}{a}} = e \cdot \frac{e^f}{a}$:

$$
\begin{aligned}
\text{Second term} &= e^{-1} \cdot \mathbb{E}_{p(x_{1:K})p(y)}\left[e \cdot \frac{e^{f(x_1, y)}}{a(y; x_{1:K})}\right] \\
&= \mathbb{E}_{p(x_{1:K})p(y)}\left[\frac{e^{f(x_1, y)}}{a(y; x_{1:K})}\right]
\end{aligned}
$$

**Step 4.** Choose $a(y; x_{1:K})$ to be the sample mean:

$$
\boxed{a(y; x_{1:K}) = m(y; x_{1:K}) = \frac{1}{K}\sum_{i=1}^{K} e^{f(x_i, y)}} \tag{10}
$$

Now the second term has a crucial property — by symmetry, every $x_i$ in the batch has equal status:

$$
\begin{aligned}
\mathbb{E}_{p(x_{1:K})p(y)}\left[\frac{e^{f(x_1, y)}}{m(y; x_{1:K})}\right]
&= \mathbb{E}_{p(x_{1:K})p(y)}\left[\frac{\frac{1}{K}\sum_{i=1}^K e^{f(x_i, y)}}{m(y; x_{1:K})}\right] \\
&= \mathbb{E}_{p(x_{1:K})p(y)}\left[\frac{m(y; x_{1:K})}{m(y; x_{1:K})}\right] \\
&= 1
\end{aligned}
$$

**Step 5.** The second term is exactly 1, canceling with the $+1$ from the first term:

$$
I \geq \mathbb{E}_{p(x_{1:K})p(y \mid x_1)}\left[\log \frac{e^{f(x_1, y)}}{\frac{1}{K}\sum_{i=1}^{K} e^{f(x_i, y)}}\right]
$$

**Step 6.** Extend to the full batch. For a batch $\{(x_i, y_i)\}_{i=1}^{K}$, each pair takes a turn as the positive:

$$
\boxed{I_{NCE} = \mathbb{E}\left[\frac{1}{K}\sum_{i=1}^{K} \log \frac{e^{f(x_i, y_i)}}{\frac{1}{K}\sum_{j=1}^{K} e^{f(x_i, y_j)}}\right]} \tag{12}
$$

This is InfoNCE — the core loss function of [CPC](/blog/contrastive-predictive-coding) and SimCLR.

### 5.3 Key Properties of InfoNCE

**Upper bound ceiling.** $I_{NCE} \leq \log K$. When the critic perfectly distinguishes positive from negative pairs, for each positive $(x_i, y_i)$ we have $e^{f(x_i, y_i)} \gg e^{f(x_i, y_j)}$ for $j \neq i$, the denominator approaches $\frac{1}{K} e^{f(x_i, y_i)}$, and the log term approaches $\log K$. This means: if the true $I(X; Y) > \log K$, InfoNCE is loose — batch size determines the ceiling on estimation accuracy.

**Optimal critic.** $f^*(x, y) = \log p(y \mid x) + c(y)$, same as IUBA. The choice of $c(y)$ does not change the value of InfoNCE — all critics of the form $f(x,y) = \log p(y \mid x) + c(y)$ form an equivalence class within which InfoNCE takes identical values (Ma & Collins, 2018). The reason: for the $i$-th term of the softmax, $c(y_i)$ appears simultaneously in the numerator and the $j=i$ term of the denominator and cancels out, while for $j \neq i$ terms, $\log p(y_j \mid x_i) + c(y_j)$ has the same form, so the entire softmax does not depend on the specific choice of $c$.

## 6. Interpolated Bound $I_\alpha$: The Bias-Variance Tradeoff

### 6.1 Motivation

| Bound | Bias | Variance | Reason |
|---|---|---|---|
| INWJ ($\alpha=0$) | Low | High | Does not use batch to share $Z(y)$ estimates |
| InfoNCE ($\alpha=1$) | High | Low | Replaces $Z(y)$ with $m(y) = \frac{1}{K}\sum e^f$ |

Can we interpolate continuously between the two?

### 6.2 Derivation

**Step 1.** The key to InfoNCE's derivation was choosing $a(y; x_{1:K}) = m(y; x_{1:K})$ (the sample mean). What if we choose a convex combination of $m$ and some baseline $q(y)$?

$$
a_\alpha(y; x_{1:K}) = \alpha \cdot m(y; x_{1:K}) + (1 - \alpha) \cdot q(y), \quad \alpha \in [0, 1]
$$

where $q(y)$ is any distribution. In practice it can be a uniform distribution, or a neural network can learn $q(y)$ as in $I_{JS}$.

**Step 2.** Substitute $a_\alpha$ into equation (9), then into INWJ:

$$
I_\alpha = 1 + \mathbb{E}_{p(x_{1:K})p(y \mid x_1)}\left[\log \frac{e^{f(x_1,y)}}{\alpha \cdot m + (1-\alpha) \cdot q(y)}\right]
- \mathbb{E}_{p(x_{1:K})p(y)}\left[\frac{e^{f(x_1,y)}}{\alpha \cdot m + (1-\alpha) \cdot q(y)}\right] \tag{13}
$$

**Step 3.** Boundary cases:

$\alpha=0$ gives $a_0 = q(y)$, reverting to INWJ. $\alpha=1$ gives $a_1 = m(y)$, reverting to InfoNCE.

### 6.3 Upper Bound

$$
\boxed{I_\alpha \leq \log \frac{K}{\alpha}}
$$

| $\alpha$ | Equivalent Bound | Upper Bound |
|---|---|---|
| $0$ | INWJ | No upper bound limit |
| $0.5$ | Intermediate | $\log 2K$ |
| $1$ | InfoNCE | $\log K$ |

The practical takeaway: smaller $\alpha$ gets closer to the true MI (low bias) but with high variance and unstable training; larger $\alpha$ gives more stable training (low variance) but may underestimate MI. You can tune $\alpha$ according to the demands of the task.

## 7. Structured Bounds: When $p(y \mid x)$ Is Tractable

In representation learning, $Y$ is the learned representation, and the encoder $p(y \mid x)$ is itself tractable (e.g., a Gaussian distribution). In this setting we can do something cleaner.

### 7.1 InfoNCE with Tractable Conditional

Directly substitute the optimal critic form $f(x,y) = \log p(y \mid x)$ into InfoNCE (recall $c(y)$ cancels in the softmax):

$$
\boxed{I \geq \mathbb{E}\left[\frac{1}{K}\sum_{i=1}^{K} \log \frac{p(y_i \mid x_i)}{\frac{1}{K}\sum_{j=1}^{K} p(y_i \mid x_j)}\right]} \tag{14}
$$

### 7.2 Leave-One-Out Upper Bound

**Step 1.** Return to the VAE-style upper bound from Section 2.1:

$$
I \leq \mathbb{E}_{p(x)}\left[D_{KL}(p(y \mid x) \parallel q(y))\right]
$$

**Step 2.** Approximate $q(y)$ using the other samples in the batch. Given a batch $\{(x_i, y_i)\}_{i=1}^K$, estimate $q(y)$ for the $i$-th sample using the remaining $K-1$:

$$
q_i(y) = \frac{1}{K-1} \sum_{j \neq i} p(y \mid x_j)
$$

**Step 3.**

$$
\begin{aligned}
\mathbb{E}_{p(x_i)}[D_{KL}(p(y \mid x_i) \parallel q_i(y))]
&= \mathbb{E}_{p(x_i)p(y \mid x_i)}\left[\log \frac{p(y \mid x_i)}{q_i(y)}\right] \\
&= \mathbb{E}\left[\log \frac{p(y_i \mid x_i)}{\frac{1}{K-1}\sum_{j \neq i} p(y_i \mid x_j)}\right]
\end{aligned}
$$

Averaging over the batch:

$$
\boxed{I \leq \mathbb{E}\left[\frac{1}{K}\sum_{i=1}^K \log \frac{p(y_i \mid x_i)}{\frac{1}{K-1}\sum_{j \neq i} p(y_i \mid x_j)}\right]} \tag{17}
$$

### 7.3 The Sandwich Bound

Compare the lower bound (14) and upper bound (17): the only difference is whether the denominator includes the sample itself.

$$
\underbrace{\frac{p(y_i \mid x_i)}{\frac{1}{K}\sum_{j} p(y_i \mid x_j)}}_{\text{Lower bound: denominator includes self, smaller}} \;\leq\; \text{True MI} \;\leq\; \underbrace{\frac{p(y_i \mid x_i)}{\frac{1}{K-1}\sum_{j \neq i} p(y_i \mid x_j)}}_{\text{Upper bound: denominator excludes self, larger}}
$$

Without any additional variational distributions or critic networks, the MI is sandwiched between an upper and lower bound.

### 7.4 Reparameterized Critic: Connecting INWJ and the VAE Upper Bound

When $p(y \mid x)$ is known, INWJ's critic does not need to be learned from scratch. Recall INWJ's optimal critic $f^*(x,y) = 1 + \log \frac{p(x \mid y)}{p(x)}$. Using Bayes' rule $p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}$:

$$
f^*(x,y) = 1 + \log \frac{p(y \mid x)}{p(y)}
$$

But we do not know $p(y)$. Introduce a variational distribution $q(y)$ to replace it:

$$
\boxed{f(x,y) = 1 + \log \frac{p(y \mid x)}{q(y)}}
$$

The critic is now reparameterized to require only learning $q(y)$ (typically a small network), rather than learning a full $(x,y) \to \mathbb{R}$ critic from scratch. Substituting this form into INWJ:

$$
I \geq \mathbb{E}_{p(x,y)}\left[1 + \log \frac{p(y \mid x)}{q(y)}\right] - e^{-1}\, \mathbb{E}_{p(x)p(y)}\left[e \cdot \frac{p(y \mid x)}{q(y)}\right]
$$

After simplification, this yields a bound that only needs $q(y)$. The form is structurally symmetric with the VAE-style upper bound $I \leq \mathbb{E}_{p(x)}[D_{KL}(p(y \mid x) \parallel q(y))]$: one evaluates KL between $p(y \mid x)$ and $q(y)$, the other evaluates cross-entropy.

### 7.5 Upper Bounding Total Correlation (Application to Disentangled Representations)

The paper further shows how to use the sandwich bound to constrain the total correlation of a representation:

$$
TC(Y) = \sum_{i=1}^d I(X; Y_i) - I(X; Y)
$$

where $Y = (Y_1, \ldots, Y_d)$ is a $d$-dimensional representation. $TC(Y)$ measures the statistical dependence among the dimensions of the representation — the smaller $TC(Y)$ is, the more disentangled the representation.

Using the Section 7.1 lower bound to sandwich $I(X; Y)$ and the Section 7.2 upper bound to sandwich each $I(X; Y_i)$, one can constrain $TC(Y)$ using only the tractable encoder $p(y \mid x)$ and a single batch, with no additional networks. This provides an information-theoretic foundation for disentanglement methods like $\beta$-VAE and FactorVAE.

## 8. Summary: How All Bounds Relate

The hierarchy of bounds, starting from mutual information $I(X;Y)$ at the root:

The VAE-style upper bound branches into the Leave-One-Out upper bound (denominator excludes self). On the lower bound side, the path splits into normalized bounds — IBA plus $h(X)$, which is intractable — and unnormalized bounds, where $q = (p/Z) e^f$.

The unnormalized path leads to IUBA, which successfully removes $h(X)$ but leaves $\log \mathbb{E}[e^f]$ intractable. From IUBA, three branches emerge. IDV/MINE applies Jensen to produce a lower bound, but its Monte Carlo estimate is not actually a bound in practice. ITUBA uses the inequality $\log x \leq x/a + \log a - 1$, yielding a tractable bound at the cost of learning $a(y)$. INWJ/f-GAN sets $a(y)=e$ as a special case — low bias, high variance, and no extra network needed.

From INWJ, two multi-sample variants branch out. InfoNCE corresponds to $\alpha=1$: high bias, low variance, with an upper ceiling of $\log K$. The interpolated bound $I_\alpha$ allows tuning $\alpha \in (0,1)$, with an upper ceiling of $\log(K/\alpha)$, trading off continuously between bias and variance.

When the encoder $p(y \mid x)$ is tractable, substituting $f = \log p(y \mid x)$ into either InfoNCE or $I_\alpha$ yields structured bounds. The sandwich bound captures MI between a lower bound (denominator includes self) and an upper bound (denominator excludes self) — the only difference being whether the $i$-th sample participates in its own denominator.

### Three Core Takeaways

1. MINE is not a bound in empirical practice: the Monte Carlo estimate of $\log \mathbb{E}[e^f]$ reverses the Jensen direction relative to the theoretical bound.
2. The bias-variance tradeoff is tunable: NWJ ($\alpha = 0$) sits at one end, InfoNCE ($\alpha = 1$) at the other, and $I_\alpha$ adjusts $\alpha$ to balance between them.
3. When $p(y \mid x)$ is tractable, you can sandwich the MI: the upper and lower bounds differ only in whether the denominator includes the sample itself, requiring no additional networks.

## Related Reading

This paper connects to several other works in the information-theoretic foundations of contrastive learning. [Noise Contrastive Estimation](/blog/noise-contrastive-estimation) lays the statistical groundwork — turning density estimation into binary classification — that InfoNCE builds on. [Contrastive Predictive Coding](/blog/contrastive-predictive-coding) is InfoNCE's first large-scale application, learning representations by predicting future observations in a compact latent space. MINE (Belghazi et al., 2018) implements the $I_{DV}$ bound with neural networks and is subsumed as a special case within this unified framework.
