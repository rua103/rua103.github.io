---
title: "From TD(lambda) to GAE: The Forward-Backward View"
description: "Explains TD(lambda)'s forward-backward equivalence, how it leads to GAE for advantage estimation in PPO, and how eligibility traces work with O(1) memory."
publishDate: 2026-07-03
tags:
  - reinforcement-learning
  - gae
  - td-lambda
  - eligibility-traces
  - advantage
  - ppo
  - rlhf
---

# From TD(lambda) to GAE: Eligibility Traces and the Forward-Backward Equivalence

> *From TD(lambda) to Generalized Advantage Estimation -- Eligibility Traces and the Forward-Backward Equivalence*
>
> Starting from TD(lambda)'s forward-backward equivalence, this post explains why PPO uses GAE to estimate advantage, how eligibility traces achieve n-step credit assignment with $O(1)$ extra memory, and how it all connects to the reward signal backpropagation in RLHF training.
>
> Prerequisites: you should know what advantage is and the basics of TD learning.

---

## 0. Notation

| Symbol | Meaning |
|:---|:---|
| $s_t, a_t, r_t$ | State, action, and immediate reward at time $t$ |
| $\tau = (s_1, a_1, r_1, \dots, s_T, a_T, r_T)$ | A complete trajectory |
| $\pi_\theta(a_t \mid s_t)$ | Policy: given state, outputs a distribution over actions |
| $V_\phi(s_t)$ | Value function: expected cumulative reward starting from $s_t$ |
| $A_t$ | Advantage: how much better action $a_t$ is than expected |
| $\gamma \in [0,1]$ | Discount factor |
| $\lambda \in [0,1]$ | Decay parameter for TD(lambda) / GAE |
| $\delta_t$ | TD error: $r_t + \gamma V(s_{t+1}) - V(s_t)$ |

---

## 1. Motivation: The Fundamental Bias-Variance Tradeoff

In RL, evaluating how good an action is requires estimating the **return** -- the total reward from the current step until the end of the episode:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T$$

But the true return is unknowable until the episode finishes. Every method trades off between **bias** and **variance**:

| Method | Bias | Variance | Problem |
|:---|:---:|:---:|:---|
| **TD(0)** | High | Low | If $V$ is misestimated, errors propagate through bootstrapping |
| **Monte Carlo** | None | High | All the randomness across the entire trajectory piles up; needs massive samples |
| **n-step TD** | Medium | Medium | A fixed $n$ is rigid -- who decided $n=4$ is better than $n=5$? |

---

## 2. Three Basic Methods

### 2.1 TD(0): One Step at a Time

$$G_t^{(1)} = r_t + \gamma V(s_{t+1})$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

One-step bootstrap. Lowest variance, but bias in $V$ directly contaminates the update.

### 2.2 n-step TD: A Fixed Window

$$G_t^{(n)} = \underbrace{r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1}}_{n \text{ steps of real reward}} + \underbrace{\gamma^n V(s_{t+n})}_{\text{bootstrap}}$$

Larger $n$ means closer to MC (low bias, high variance). Smaller $n$ means closer to TD(0). The problem: **the optimal $n$ is task-dependent, and tuning it is painful.**

### 2.3 Monte Carlo: All the Way

$$G_t^{\text{MC}} = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

Zero bias, no bootstrap. But variance is $\propto (T-t) \cdot \sigma_r^2$, making long sequences nearly unusable.

---

## 3. TD(lambda): Mixing All n-step Returns with Geometric Weighting

The issue with a fixed $n$ is that the reward at step $n+1$ is **completely cut off** from the update -- it can only leak through indirectly via the bootstrap, which is painfully slow.

TD(lambda)'s solution: mix together returns for $n = 1, 2, 3, \dots, \infty$:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

The weight distribution:

$$(1-\lambda),\ (1-\lambda)\lambda,\ (1-\lambda)\lambda^2,\ \dots$$

---

### 3.1 Why Geometric Weighting?

**1. Continuous Bias-Variance Tradeoff**

$\lambda = 0$ gives only $G_t^{(1)}$, i.e. TD(0).
$\lambda = 1$ gives only $G_t^{(\infty)}$, i.e. MC (in finite-horizon settings).

Any intermediate $\lambda$ interpolates continuously between the two. No need to agonize over a fixed $n$.

**2. Eligibility Traces**

If the weights weren't geometric, implementing an n-step mixture would require **storing all data from the next n steps** in memory -- you'd have to wait until step $t+n$ to receive the reward before updating step $t$. This is the **forward view** -- you must wait for the episode to end; it cannot run online.

Geometric weights have a special property: **the current weight equals the previous weight times lambda**. From this, we derive the equivalent **backward view** -- each state only needs to maintain a single scalar trace (the accumulating trace):

$$E_t(s) = \gamma\lambda \cdot E_{t-1}(s) + \mathbb{I}_{\{S_t = s\}}$$

Once we have $\delta_t$, we update **all** states simultaneously:

$$V(s) \leftarrow V(s) + \alpha \cdot \delta_t \cdot E_t(s)$$

No n-step buffer needed, no waiting for the episode to end. $O(1)$ extra memory per state, fully online.

**3. Robustness to Noise**

If you only use a fixed $n$, one unusually noisy step at position $n$ can contaminate the entire update. A geometric mixture is effectively a weighted smoothing filter over all $n$ -- the noise from any single long-step return gets diluted. Lambda serves as a single knob, far less sensitive to hyperparameters than a fixed $n$.

---

### 3.2 Lemma: Expanding $G_{t:t+n} - V(S_t)$ via TD Errors

This identity underpins all subsequent derivations, so let's prove it separately:

$$G_{t:t+n} - V(S_t) = \sum_{i=0}^{n-1} \gamma^i \delta_{t+i}$$

**Proof (telescoping sum)**:

$$G_{t:t+n} - V(S_t) = \left(\sum_{i=1}^{n} \gamma^{i-1} R_{t+i} + \gamma^n V(S_{t+n})\right) - V(S_t)$$

Substitute $\delta_{t+i} = R_{t+i+1} + \gamma V(S_{t+i+1}) - V(S_{t+i})$ into the right-hand side $\sum_{i=0}^{n-1} \gamma^i \delta_{t+i}$:

$$\sum_{i=0}^{n-1} \gamma^i \delta_{t+i} = \sum_{i=0}^{n-1} \gamma^i \left[R_{t+i+1} + \gamma V(S_{t+i+1}) - V(S_{t+i})\right]$$

$$= \underbrace{\sum_{i=0}^{n-1} \gamma^i R_{t+i+1}}_{= \sum_{i=1}^{n} \gamma^{i-1} R_{t+i}} + \underbrace{\sum_{i=0}^{n-1} \gamma^{i+1} V(S_{t+i+1}) - \sum_{i=0}^{n-1} \gamma^i V(S_{t+i})}_{\text{telescoping}}$$

Writing out the telescoping of the last two terms makes it clear:

$$\sum_{i=0}^{n-1} \gamma^{i+1} V(S_{t+i+1}) = \gamma V(S_{t+1}) + \gamma^2 V(S_{t+2}) + \cdots + \gamma^n V(S_{t+n})$$

$$\sum_{i=0}^{n-1} \gamma^i V(S_{t+i}) = V(S_t) + \gamma V(S_{t+1}) + \cdots + \gamma^{n-1} V(S_{t+n-1})$$

Subtracting, the middle terms $\gamma V(S_{t+1}) \cdots \gamma^{n-1} V(S_{t+n-1})$ all cancel, leaving:

$$\gamma^n V(S_{t+n}) - V(S_t)$$

Therefore $\sum_{i=0}^{n-1} \gamma^i \delta_{t+i} = \sum_{i=1}^{n} \gamma^{i-1} R_{t+i} + \gamma^n V(S_{t+n}) - V(S_t) = G_{t:t+n} - V(S_t)$. $\square$

### 3.3 Forward View as $\sum (\gamma\lambda)^k r_{t+k}$

The forward view definition:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

Substituting the expansion of $G_t^{(n)}$:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \left[\sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})\right]$$

For a finite episode, once $n$ exceeds the trajectory endpoint, $V(s_{t+n})$ lands on a terminal state ($V=0$), and the bootstrap term vanishes naturally. For the infinite case, $|\gamma| < 1$ guarantees $\gamma^n V(s_{t+n}) \to 0$ (since $V$ is bounded). In both cases, only the reward portion remains in the lambda-return:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \sum_{k=0}^{n-1} \gamma^k r_{t+k}$$

Swap the order of the double sum. The outer $n$ runs from $1$ to $\infty$, the inner $k$ from $0$ to $n-1$. Equivalently, $k$ runs from $0$ to $\infty$ and $n$ runs from $k+1$ to $\infty$:

$$G_t^\lambda = (1-\lambda)\sum_{k=0}^{\infty} \gamma^k r_{t+k} \sum_{n=k+1}^{\infty} \lambda^{n-1}$$

Change variables in the inner sum: let $m = n-k-1$, so $n = m+k+1$:

$$\sum_{n=k+1}^{\infty} \lambda^{n-1} = \sum_{m=0}^{\infty} \lambda^{m+k} = \lambda^k \sum_{m=0}^{\infty} \lambda^m = \frac{\lambda^k}{1-\lambda}$$

Substituting back:

$$G_t^\lambda = (1-\lambda)\sum_{k=0}^{\infty} \gamma^k r_{t+k} \cdot \frac{\lambda^k}{1-\lambda} = \sum_{k=0}^{\infty} (\lambda\gamma)^k r_{t+k}$$

This is the forward view in reward form -- "the sum of future rewards decaying by $(\lambda\gamma)^k$."

---

### 3.4 Forward View to Backward View: Straight from the Definition

There is no need to first derive the reward form and then substitute $r \to \delta$ back in -- that route introduces an extra $S_2 - S_3$ cancellation step with unnecessary technical detail. The shortest path uses the lemma from section 3.2 directly.

Start from the forward view definition:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

Subtract $V(s_t)$ from both sides and apply section 3.2's result $G_t^{(n)} - V(s_t) = \sum_{i=0}^{n-1} \gamma^i \delta_{t+i}$:

$$G_t^\lambda - V(s_t) = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \sum_{i=0}^{n-1} \gamma^i \delta_{t+i}$$

Swap the summation order -- the exact same trick as section 3.3, with $i$ from $0$ to $\infty$ and $n$ from $i+1$ to $\infty$:

$$G_t^\lambda - V(s_t) = (1-\lambda)\sum_{i=0}^{\infty} \gamma^i \delta_{t+i} \sum_{n=i+1}^{\infty} \lambda^{n-1} = (1-\lambda)\sum_{i=0}^{\infty} \gamma^i \delta_{t+i} \cdot \frac{\lambda^i}{1-\lambda}$$

$$\boxed{G_t^\lambda = V(s_t) + \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}}$$

This is the **equivalent backward view** -- expressing lambda-return as $V(s_t)$ plus a decaying sum of TD errors. Note that $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ only depends on two adjacent states, so it can be computed online at each step, broadcast backward along the eligibility trace, with no need to wait for the episode to end.

The derivation never uses the $S_2 - S_3$ cancellation -- the lemma in section 3.2 already took care of the $V(s_t)$ cancellation upfront. All we did here was reorganize the weighted sum of $\delta_t$.

---

### 3.5 A Rigorous Proof of Forward-Backward Equivalence (Offline Case)

Under offline updates, $V$ stays fixed throughout the entire episode, so all $\delta_t$ are constant. We now prove that the total update from the forward view equals the total update from the backward view.

**Forward view**: at each time $t$, update $V$ toward the target $G_t^\lambda$:

$$\Delta V_{\text{Forward}}(s) = \alpha \sum_{t=0}^{T-1} \left(G_t^\lambda - V(S_t)\right) \mathbb{I}_{\{S_t = s\}}$$

Substitute $G_t^\lambda - V(S_t) = \sum_{k=0}^{T-1-t} (\gamma\lambda)^k \delta_{t+k}$:

$$\Delta V_{\text{Forward}}(s) = \alpha \sum_{t=0}^{T-1} \left[\sum_{k=0}^{T-1-t} (\gamma\lambda)^k \delta_{t+k}\right] \mathbb{I}_{\{S_t = s\}}$$

**Backward view**: at each time $t$, use $\delta_t$ and the eligibility trace $E_t(s)$ to update all states:

$$\Delta V_{\text{Backward}}(s) = \alpha \sum_{t=0}^{T-1} \delta_t E_t(s)$$

Expand the eligibility trace: $E_t(s) = \sum_{j=0}^{t} (\gamma\lambda)^{t-j} \mathbb{I}_{\{S_j = s\}}$. Substituting:

$$\Delta V_{\text{Backward}}(s) = \alpha \sum_{t=0}^{T-1} \delta_t \left[\sum_{j=0}^{t} (\gamma\lambda)^{t-j} \mathbb{I}_{\{S_j = s\}}\right]$$

**Swap the sums**: let $\tau = t$ and $t' = j$. In the forward view, set $\tau = t+k$ (so $k = \tau - t$, with $t \le \tau$):

$$\Delta V_{\text{Forward}}(s) = \alpha \sum_{\tau=0}^{T-1} \delta_\tau \left[\sum_{t=0}^{\tau} (\gamma\lambda)^{\tau-t} \mathbb{I}_{\{S_t = s\}}\right]$$

In the backward view (with $t = \tau$, $j = t'$):

$$\Delta V_{\text{Backward}}(s) = \alpha \sum_{\tau=0}^{T-1} \delta_\tau \left[\sum_{t'=0}^{\tau} (\gamma\lambda)^{\tau-t'} \mathbb{I}_{\{S_{t'} = s\}}\right]$$

The expressions inside the brackets are identical, therefore:

$$\boxed{\Delta V_{\text{Forward}}(s) = \Delta V_{\text{Backward}}(s)}$$

The forward and backward views are strictly equivalent in the offline sense. Through eligibility traces $E_t(s)$, the backward view achieves -- with $O(|S|)$ extra memory -- the same credit assignment that the forward view would need to buffer the entire trajectory to compute.

### 3.6 Three Types of Eligibility Traces

Sections 3.2 through 3.4 all used the **accumulating trace**:

$$E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbb{I}_{\{S_t = s\}}$$

When a state is visited repeatedly, the trace has no upper bound (it can grow past $1$), which may cause gradient explosion under function approximation. In practice, there are two variants.

**Replacing trace**:

$$E_t(s) = \begin{cases} 1 & \text{if } S_t = s \\ \gamma\lambda E_{t-1}(s) & \text{otherwise} \end{cases}$$

On each revisit, the trace is **reset to 1** rather than accumulated. Bounded above by $1$, numerically more stable. PPO and GAE implementations typically use a replacing trace or compute offline (in which case the trace type is irrelevant, since GAE bypasses explicit traces via backward scanning).

**Dutch trace** (interpolating between the two, $\alpha \in [0,1]$):

$$E_t(s) = \begin{cases} \alpha \cdot \gamma\lambda E_{t-1}(s) + 1 & \text{if } S_t = s \\ \gamma\lambda E_{t-1}(s) & \text{otherwise} \end{cases}$$

$\alpha = 0$ degenerates to replacing (reset to $1$ on revisit), $\alpha = 1$ degenerates to accumulating (decay then $+1$ on revisit). Intermediate $\alpha$ controls how much of the old trace is retained.

### 3.7 Offline Equivalence to Online Approximation

The proof in section 3.5 relies on a critical assumption: $V$ remains **fixed** throughout the episode. In an online setting, $V$ changes after every update, and subsequent $\delta_t$ shift accordingly -- the forward and backward views are **no longer strictly equivalent**.

Seijen & Sutton (2014) proposed **true online TD(lambda)**, which applies an extra correction to the trace at each update to restore equivalence, at the cost of increasing per-step computation from $O(|S|)$ to $O(d)$ (where $d$ is the parameter dimension of $V$). This correction is rarely used in practice, because the online approximation error is negligible on most tasks, especially in algorithms like PPO that already perform offline batch updates after episodes finish.

---

## 4. GAE: From lambda-Return to Advantage

### 4.1 Definition

The original definition of advantage:

$$A_t = G_t - V(s_t)$$

"How much better the actual return is compared to what $V(s_t)$ predicted."

GAE's approach: **replace $G_t$ with $G_t^\lambda$**:

$$A_t^{\text{GAE}(\gamma,\lambda)} = G_t^\lambda - V(s_t)$$

Substituting $G_t^\lambda = V(s_t) + \sum (\gamma\lambda)^k \delta_{t+k}$:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}$$

**$V(s_t)$ cancels out.** The GAE advantage estimate does not directly depend on the absolute value of $V(s_t)$ -- it only depends on TD errors (the **differences** between $V$ at adjacent states). Part of the bias cancels in the subtraction, which is why GAE is more stable than directly using $G_t - V(s_t)$.

### 4.2 Finite-Length Truncation

In practice, trajectory length is $T$, and the sum is truncated:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

$$\delta_{t+l} = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$$

### 4.3 Two Extremes

| $\lambda$ | $A_t$ | Meaning |
|:---|:---|:---|
| $0$ | $\delta_t$ | TD(0) -- look one step ahead, fully trust $V$ |
| $1$ | $\sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l} = G_t - V(s_t)$ | Monte Carlo -- trust $V$ not at all, only real rewards |

### 4.4 Computation: $O(T)$ Backward Scan

The recurrence:

$$A_{T-1} = \delta_{T-1}, \qquad A_t = \delta_t + \gamma\lambda \cdot A_{t+1} \quad \text{for } t = T-2, \dots, 0$$

A single backward pass -- no need to explicitly sum over each $t$.

### 4.5 Implementation: Under 10 Lines of PyTorch

```python
def compute_gae(rewards, values, gamma, lam):
    """
    rewards: [T]   per-step immediate rewards
    values:  [T+1] per-step V estimates (includes terminal V(s_{T+1}))
    returns: advantages [T]
    """
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages
```

The line `delta + gamma * lam * gae` is a direct translation of $A_t = \delta_t + \gamma\lambda \cdot A_{t+1}$. The entire function is just this one recurrence; everything else is index alignment.

---

## 5. Derivation Chain Overview

$$\boxed{G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})}
\;\xrightarrow{\text{geometric mixture}}\;
\boxed{G_t^\lambda = \sum_{k=0}^{\infty} (\gamma\lambda)^k r_{t+k}}
\;\xrightarrow{\text{sec. 3.2 lemma}}\;
\boxed{G_t^\lambda = V(s_t) + \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}}
\;\xrightarrow{-V(s_t)}\;
\boxed{A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}}
\;\xrightarrow{\text{truncate}}\;
\boxed{A_t = \delta_t + \gamma\lambda \cdot A_{t+1}}$$

One line, five steps, from n-step return to the PPO implementation.

---

## 6. Application in RLHF

Taking [[Fine-Tuning GPT-2 from Human Preferences (2019)]] as an example, GAE's usage in RLHF training differs from standard RL in a few key ways.

### 6.1 RLHF as an MDP

Text generation is modeled as a token-level MDP:

- **State $s_t$**: the prompt $x$ plus the first $t-1$ generated tokens
- **Action $a_t$**: generating the $t$-th token (choosing from the vocabulary)
- **Reward**: the reward model $r_\phi(x, y)$ scores the **entire text** -- this means **sparse rewards**: only the final token receives a non-zero reward; all intermediate tokens get zero immediate reward

So in a sequence of length $T$, only $r_{T-1}$ (or $r_T$, depending on indexing convention) is non-zero. Every $r_0, r_1, \dots, r_{T-2}$ is exactly zero.

### 6.2 Why gamma = 1?

Text generation tasks typically set $\gamma = 1$. The reason isn't "text has no discounting" -- it is more specific:

- Text generation is a **finite-horizon** task (the episode naturally terminates at the EOS token), so there is no need for $\gamma < 1$ to keep the return bounded
- $\gamma < 1$ introduces positional bias -- earlier tokens get inherently less credit than later ones, but this makes no sense for text (the first sentence matters just as much as the last paragraph for overall quality)
- With $\gamma = 1$, the TD errors for all intermediate tokens simplify to:

$$\delta_t = \underbrace{0}_{r_t} + \underbrace{1}_{\gamma} \cdot V(s_{t+1}) - V(s_t) = V(s_{t+1}) - V(s_t)$$

Only the final step differs: $\delta_{T-1} = r_{T-1} + V(s_T) - V(s_{T-1})$, where $V(s_T) = 0$ (terminal state).

### 6.3 GAE's Credit Assignment Mechanism

With $\gamma = 1$, GAE reduces to:

$$A_t^{\text{GAE}} = \sum_{l=0}^{T-t-1} \lambda^l \, \delta_{t+l}$$

For intermediate tokens, $\delta_t = V(s_{t+1}) - V(s_t)$. For the final token, $\delta_{T-1} = r_{T-1} - V(s_{T-1})$. Working backward from the last token:

$$\begin{aligned}
A_{T-1} &= r_{T-1} - V(s_{T-1}) \\[2pt]
A_{T-2} &= \underbrace{V(s_{T-1}) - V(s_{T-2})}_{\delta_{T-2}} \;+\; \lambda \cdot \underbrace{(r_{T-1} - V(s_{T-1}))}_{A_{T-1}} \\[2pt]
A_{T-3} &= \delta_{T-3} + \lambda \cdot A_{T-2}
\end{aligned}$$

**Intuition**: the final reward $r_{T-1}$ "seeps" into earlier $\delta_t$ through the differences in $V(s_{T-1})$, then propagates to even earlier tokens with $\lambda^l$ decay. Lambda controls the propagation distance:
- $\lambda \approx 1$: $r_{T-1}$ is shared almost equally; every token gets credit
- $\lambda \approx 0$: only the token immediately before the reward gets non-zero advantage
- $\lambda \approx 0.95$ (a common default): $0.95^{20} \approx 0.36$ -- contribution drops to about 1/3 at 20 tokens away

### 6.4 Coupling with PPO Clipping

GAE's output $A_t$ feeds directly into PPO's clipped surrogate objective:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

Here is an easily overlooked detail: **GAE's lambda and PPO's clip epsilon are complementary**. If lambda is set too high (high variance), you can compensate with a tighter epsilon (more conservative policy updates). If lambda is set too low (high bias), the policy learns something that is fundamentally biased, and no amount of epsilon tuning will fix it. In practice, you typically tune lambda first (to control credit assignment quality), then tune epsilon (to control update stability).

---

## Related Notes

- [[Fine-Tuning GPT-2 from Human Preferences (2019)]] -- section 2.4.3 on GAE's concrete use in RLHF
- Sutton & Barto, Chapter 12 -- the full theory and convergence proofs for TD(lambda) and eligibility traces
- Schulman et al. (2015) -- the original GAE paper
