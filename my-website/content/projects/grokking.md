+++
date = '2026-07-29T00:00:00+02:00'
draft = false
title = 'A Spectral Theory of Grokking'
slug = 'grokking'
layout = 'grokking'
description = 'Weight decay induces grokking through evolution of the neural tangent kernel.'
subtitle = 'How a neural network reorganizes its learning dynamics around the Fourier structure of modular addition, work done under the supervision of Professor Rulands, LMU.'
authors = ['Lenz Pracher', 'Pascal de Jong', 'Oskar Liesaus', 'Alan Jeffares', 'Steffen Rulands']
+++

**Grokking** occurs when a neural network sharply transitions from a memorized to a generalized state. It first reaches almost perfect training accuracy while its validation accuracy remains near chance.{{< footnote >}}Throughout this post, “validation” refers to the held-out part of the training distribution.{{< /footnote >}} After many more updates, validation accuracy rises abruptly, even though the training data has not changed. The network appears to move from memorizing examples to learning the "rule" behind them. We study what changes inside the network during that delay. 

<details class="tldr" open>
  <summary>TLDR</summary>
  <div class="tldr__body">
    <p>In this post, I will present a theory of Grokking through time scale separation. Homogeneous neural networks perform the kernel ridge regression solution on fast scales, while operating feature learning on slow time scales controlled by Weight Decay. Grokking is then delayed generalization, where we reach a fast memorizing solution through regression on a fixed feature set and slowly increase task aligning features at the same pace producing a sharp generalization transition. The evolution of task aligned features is driven by the irreducible loss induced by L2 Weight Decay.</p>
  </div>
</details>

<figure class="grokking-figure">
  <img src="../../images/grokking/learning-curve.png" alt="Training accuracy reaches one long before validation accuracy rises during grokking.">
  <figcaption><strong>Figure 1.</strong> A typical grokking curve. The network fits the training set early leading to high accuracy, while generalization appears an order of magnitude of updates later. <a href="#setup-a">Setup A</a>.</figcaption>
</figure>

## 1. Grokking and Modular Addition
We start by reviewing some of the background and describing the setup to arrive at the phenomenon of grokking.

### 1.1 Short Literature Review

[Power et al.](#ref-power) first observed grokking as delayed generalization on small algorithmic datasets. Their experiments showed that long training and regularization can produce a sudden transition in validation accuracy well after overfitting. Later work aimed at describing the mechanistic and theoretical origin of the transition.

[Nanda et al.](#ref-nanda) mechanistically investigated modular-addition transformers. They found a Fourier circuit that forms gradually beneath the sharp accuracy transition. Memorizing components dominate first, Fourier components grow later, and cleanup removes the remaining memorizing structure.

The recent work by [Xu et al.](#ref-xu) analyzes a simple enough model to be solved exactly. Ridge regression can be exactly solved on an over-parameterized linear model, trained by gradient descent with weight decay. They bound the generalization time in terms of the training hyperparameters. Their model traces feature learning back to an ill conditioned initialization through a fixed set of features.

Other studies expanded the phase diagram under different hyperparameters. [Doshi et al.](#ref-doshi) separated comprehension, grokking, memorization, confusion, and forgetting on corrupted arithmetic data. We aim to add to the growing literature by providing an analytical perspective on the learning dynamics in ReLU MLPs.

### 1.2 Setup

Effective theories of Grokking can be reached by restriction of the model class. [Liu et al.](#ref-liu) work with a constrained model, where an embedding forces a group operation. The symbols $a$ and $b$ are mapped to trainable embedding vectors, the two vectors are added, and a small decoder network reads the sum. One is then able to write down an effective theory for the embedding vectors $E_i$, which replicates the empirical experiments of the training dynamics of the neural network. We take a similar approach by assuming a homogeneous neural network:

<div class="math-display">
\[
D\cdot f_{\theta}(x^\mu) = \nabla_\theta f_\theta\rvert_{x^\mu}\cdot \theta,
\]
</div>

where $\theta$ are the stacked weights of the neural network $f_\theta$ and $D$ is the degree of homogeneity. This assumption is satisfied by for example ReLU MLPs. This restriction allows us to analyze the gradient flow dynamics on the level of individual samples and integrating out weights directly.

### 1.3 Decomposition of Modular Addition

We achieve Grokking through modulo addition. Our learning task is therefore addition modulo a prime $p$:

<div class="math-display">
\[
(a,b)\longmapsto c=(a+b)\bmod p,
\qquad a,b,c\in\mathbb Z_p.
\]
</div>

The inputs and outputs are one-hot encoded. We also one-hot encode the output numeral $c$. Modulo addition is periodic with even frequencies $\frac{k}{p}$. The natural basis on this cyclic space consists of the characters

<div class="math-display">
\[
\chi_k(x)=\exp\!\left(\frac{2\pi i kx}{p}\right).
\]
</div>

One can therefore decompose the target into a linear combination of fourier features:

<div class="math-display">
\[
y_c(a,b)
=\frac1p\sum_{k=0}^{p-1}
\exp\!\left(-\frac{2\pi i kc}{p}\right)
\chi_k(a)\chi_k(b).
\]
</div>

Only products with the same frequency $k$ on both input sectors $a,b$ appear. We can therefore hypothesize, that learning Fourier features with the same frequency in both $a,b$ will be useful for generalization. A network encoding these features in its last layer, can linearly solve for the generalization rule. 

### 1.4 Experimental Setup and Gradient-Flow Dynamics

We use $p=23$ producing 529 input pairs, a bias free, one hidden-layer ReLU network of width $256$, trained with full-batch gradient descent on Regression Loss and Weight Decay. We stack the two one-hot encodings of the inputs $a$ and $b$ as an input and directly regress on the one-hot encoded label $c$. We structure the train-validation split by drawing from

<div class="math-display">
\[
\mathcal T_D=\{(a,b):a-b\pmod {23}\in D\},
\]
</div>

where we draw 16 values from $\mathbb Z_p$. {{< footnote >}}This split is designed to preserve circulant matrices, but is not necessarily needed for the empirical analysis.{{< /footnote >}}
To describe the learning dynamics without carrying an output-class index through every equation, first consider one scalar output $f_\theta(x)$. We define its prediction error by the residual:

<div class="math-display">
\[
\sigma^\mu=(f_\theta(x^\mu)-y^\mu)
\]
</div>

Together with Weight Decay, the loss objective becomes:

<div class="math-display">
\[
L=\frac12\sum_\mu(\sigma^\mu)^2+\frac{\lambda_W}{2}\lVert\theta\rVert^2.
\]
</div>

Our equations of motion are given by gradient descent.

<div class="math-display">
\[
\dot\theta
=-\sum_\mu\nabla_\theta f_\theta(x^\mu)\,\sigma^\mu
-\lambda_W\theta .
\]
</div>

The first term is given by the regression objective. It is described by the empirical [Neural Tangent Kernel](#ref-jacot), $K(x,x')=\nabla_\theta f_\theta(x)\cdot\nabla_\theta f_\theta(x')$. The second term pulls every weight toward the origin. For a homogeneous neural network this amounts to reducing the output magnitude. This relation allows us to rewrite the equations of motion on the level of function space:

<div class="math-display">
\[
\dot f_\theta(x^\mu)
=\nabla_\theta f_\theta(x^\mu)\cdot\dot\theta
=-\sum_\nu K(x^\mu,x^\nu)\,\sigma^\nu
-\lambda_W\cdot D\,f_\theta(x^\mu).
\]
</div>

For a degree-$D$ homogeneous network, Euler's relation gives $\nabla_\theta f_\theta(x)\cdot\theta=D\,f_\theta(x)$ and so the equations integrate out the weight contributions. This is remarkable, as it allows us to ignore weights in our analysis, which are the crux of most interpretability works. This allows us to directly interpret Gradient Flow as a  dynamical system between samples in the training and validation distribution. We are interested in the deviation of our neural network from the labels and therefore investigate the evolution of the residuals of each sample $\sigma^\mu$.

With $D=2$ for the ReLU one-hidden layer neural network, the equations become:

<div class="math-display">
\[
\dot{\boldsymbol\sigma}^\mu
=-\sum_\nu(K^{\mu\nu}+2\lambda_W\delta^{\mu\nu})\boldsymbol\sigma^\nu
-2\lambda_W\mathbf y^\mu .
\]
</div>

Weight Decay forces a relaxation of the prediction. For samples with $y^\mu=0$, this amounts to an increase in prediction accuracy. This is why we see an increase in the relaxation rate $(\Lambda_\alpha+2\lambda_W)$. For samples with $y^\mu\neq 0$, Weight Decay induces an irreducible loss, which is supported by the source $-2\lambda_W y^\mu$ in the equations of motion. For a constant Kernel $K$, the dynamics reach the kernel ridge regression solution:

<div class="math-display">
\[
\boldsymbol\sigma_*^\mu
=-2\lambda_W\,(K^{\mu\nu}+2\lambda_W I)^{-1}\mathbf y^\nu .
\]
</div>

Weight decay therefore leaves a residual behind, of size proportional to $\lambda_W$. One can also observe, that the only way to reduce the regression loss of this solution is to increase the alignment of $K$ with the label space $y^\mu$, while kernel modes independent of the target do not contribute. We therefore conclude, that for homogeneous neural networks, the dynamics are mainly driven by the empirical Neural Tangent Kernel and will in the following analyze the learning dynamics from this perspective.

## 2. Generalization with Tangent Features

We first inspect the NTK $K^{\mu\nu}$ throughout one grokking run. At initialization it has little visible organization. During memorization it develops broad local structure. Near the validation-accuracy transition, repeated diagonal bands appear and then sharpen into a fine grained periodic pattern. This first observation prompted the empirical and theoretical investigation of the NTK presented here.

<figure class="grokking-figure grokking-figure--wide">
  <video controls muted playsinline preload="metadata">
    <source src="../../videos/grokking/ntk-evolution.mp4" type="video/mp4">
    Your browser does not support the embedded video.
  </video>
  <figcaption><strong>Video 1.</strong> Evolution of the label-averaged empirical NTK projected onto the true label components during training. Fine structure emerges as validation accuracy begins to rise. Experimental <a href="#setup-a">Setup A</a>.</figcaption>
</figure>

<figure class="grokking-figure grokking-figure--wide">
  <img src="../../images/grokking/ntk-evolution.jpeg" alt="Label-averaged NTK at initialization, memorization, transition, and generalization, with the corresponding learning curve.">
  <figcaption><strong>Figure 2.</strong> The label-averaged NTK projected onto the true label components at four stages of learning. The fine grained pattern appears around the delayed rise in validation accuracy. Experimental <a href="#setup-a">Setup A</a>.</figcaption>
</figure>

### 2.1 Investigation of the Eigenspectrum

We want to understand, how this fine structure relates to generalization. To this end, we investigate the Kernel Eigenspectrum and how it relates to the label space. We plot the components of the eigenvectors with the 6 highest eigenvalues, which dominate the residual dynamics. Before generalization the eigenvectors are noisy. At the generalization boundary, we observe the emergence of standing waves in label space. These correspond directly to the characters with same frequency in both $a,b$ Fourier Sectors as described in [section 1.3](#13-decomposition-of-modular-addition). As training continues, the highest eigenvalue eigenvectors emerge as standing waves over the modulo labels. This makes sense as it allows the neural network to discover the data generating rule by a linear combination of these features, which is how a homogeneous neural network produces its prediction. These standing waves are aligned with the target and therefore reduce the loss reached in the ridge regression equilibrium.

<figure class="grokking-figure grokking-figure--wide">
  <img src="../../images/grokking/eigenvectors.png" alt="Leading empirical NTK eigenvectors become Fourier-like standing waves during grokking.">
  <figcaption><strong>Figure 3.</strong> Leading NTK eigenvectors across training. Fourier-like standing waves become clear before and during the validation transition. <a href="#setup-a">Setup A</a>.</figcaption>
</figure>

### 2.2 Evolution of Kernel Eigenmodes and the Neural Tangent Hierarchy

We now understand what structure emerges during the generalization transition in grokking modular arithmetic. However, the theoretical explanation for the delayed generalization through feature learning is still unclear. We investigate the evolution of the NTK $K$ during Gradient Flow. For this we employ the [Neural Tangent Hierarchy](#ref-huang) of Huang and Yau. The evolution of the NTK produces a second object $K^{(2)}(x,x';z)$, which then produces a fourth and onwards. The Neural Tangent Kernel and all higher order modes are again homogeneous functions allowing the integration of the Weight Decay terms. The coupled equations become:

<div class="math-display">
\[
\dot{\boldsymbol\sigma}
=-(K+2\lambda_W I)\boldsymbol\sigma-2\lambda_W\mathbf y,
\qquad
\dot K=-K^{(2)}\boldsymbol\sigma-2\lambda_W K,
\qquad
\dot K^{(2)} = \dots
\]
</div>

In the NTK initialization, the evolution of $K^{(2)}$ is suppressed by orders of the width $m$ with $K$ becoming constant in the infinite width limit. This gives us intuition that for finite width, we might be able to close the Neural Tangent Hierarchy and treat $K^{(2)}$ as constant. The $\boldsymbol\sigma$ term changes different kernel entries according to the current residuals. The weight Decay term instead exponentially decays every entry of $K$ directly. The growth of the NTK is then directly coupled to the remaining prediction residual.

We now decompose into NTK Fourier modes and write the same equations for each Fourier direction {{< footnote >}}At initialization, one can explicitly compute the infinite width Neural Tangent Kernel, which then only depends on differences of inputs. By the convolution theorem, this Kernel is then diagonal in Fourier Space, which allows us to approximately track the evolution of each of the Fourier modes independently over time. Additionally, each Fourier mode (besides constant sectors) has equal weight at initialization with one-hot encoded inputs and outputs. This leads to the memorization solution at the start of the training, where all modes are used equally and allows us to focus on a single mode analysis, which will hold for all leaving an effective diagonal $K^{(2)}$.{{< /footnote >}}. Let $\Lambda$ be the eigenvalue of $K$ along a chosen Fourier vector $(a,b)$, and let $\sigma$ and $y$ be the residual and target components along that vector. Applying $K^{(2)}$ to the same vector on its first two indices and to the mode residual on its third leaves a scalar coefficient, which we call $a$. We now apply the Neural Tangent Hierarchy and set $a$ approximately as a positive constant. This closure allows us to analytically investigate the dynamics. {{< footnote >}}Further empirical analysis has shown that $a$ evolves quite a bit, but qualitatively the results of the constant $a$ closure fit quite well. Expansions of $a$ in $K$ also seem promising. Using quadratic activation functions should also yield an analytical closure without approximations.{{< /footnote >}} Our canonical mode-wise equations become

<div class="math-display">
\[
\dot\sigma=-(\Lambda+2\lambda_W)\sigma-2\lambda_W y,
\qquad
\dot\Lambda=-a\sigma-2\lambda_W\Lambda.
\]
</div>

Structurally these equations are very simple! Surprisingly, they capture most of the qualitative observations during Grokking. Residuals $\sigma$ relax exponentially fast onto the ridge regression solution on the time scale of single Gradient Descent steps $t$. The evolution of features is driven by the remaining residual, which is induced by Weight Decay. This leads to a slow time scale $\tau = \lambda_W\cdot t$. Feature learning is driven through alignment with residuals, while weight decay removes eigenvalues through exponential decay $-2\lambda_W\Lambda$. A target aligned eigenvalue carries a persistent residual source. An unsupported mode does not and decays. We can move from first order differential equations to a closed second order equation for the eigenvalues $\Lambda$. {{< footnote >}}In Dynamical Systems, one usually does the opposite trick and moves from a single variable to a two dimensional phase space. However, in this case, we will describe the second order dynamics from the perspective of a potential to reach some insights and then go back to a Hamiltonian Formulation.{{< /footnote >}}

#### Second Order Equation for Eigenmodes

We can eliminate $\sigma$ exactly. Differentiate the equation for $\Lambda$, use the equation for $\dot\sigma$, and replace $\sigma$ by $-(\dot\Lambda+2\lambda_W\Lambda)/a$. The result closes entirely on the eigenvalue:

<div class="math-display">
\[
\ddot\Lambda
+(\Lambda+4\lambda_W)\dot\Lambda
+2\lambda_W\Lambda(\Lambda+2\lambda_W)
-2a\lambda_W y=0.
\]
</div>

This has the form of a damped Newton equation,

<div class="math-display">
\[
\ddot\Lambda+\Gamma(\Lambda)\dot\Lambda
+\frac{\partial V}{\partial\Lambda}=0,
\qquad
\Gamma(\Lambda)=\Lambda+4\lambda_W,
\]
\[
V(\Lambda)
=\frac{2\lambda_W}{3}\Lambda^3
+2\lambda_W^2\Lambda^2
-2a\lambda_W y\Lambda.
\]
</div>

For physical eigenvalues $\Lambda\geq0$, the friction $\Gamma$ is positive. The mechanical energy $\mathcal H=\dot\Lambda^2/2+V(\Lambda)$ therefore decreases as

<div class="math-display">
\[
\dot{\mathcal H}=-\Gamma(\Lambda)\dot\Lambda^2\leq0.
\]
</div>

Every Fourier mode follows this same potential equation. What differs is the source $a y$. A mode absent from the target has $a y=0$, so its potential has its physical minimum at $\Lambda=0$ and Weight Decay removes it. A target aligned mode has $a y>0$, which tilts the potential toward a positive minimum. Starting from similar eigenvalues, aligned and non aligned modes therefore move toward different endpoints. This also explains the abrupt transition from a memorized to a generalized state. As all eigenvalues start at the same initial condition, they all cross a critical boundary $\Lambda_{grok}$ at the same time, when they become useful for the ridge regression fit and the network suddenly generalizes. **Grokking is then just a phenomenon of an ill-conditioned initialization**, which evolves on a timescale of $\lambda_W\cdot t$ to the well-conditioned solution.

<figure class="grokking-figure grokking-figure--wide">
  <img src="../../images/grokking/theory-potential-trajectory.svg" alt="Potential and eigenvalue trajectories for target-aligned and non-aligned Fourier modes.">
  <figcaption><strong>Figure 4.</strong> Mechanical picture of spectral selection. Weight decay pulls unsupported modes toward zero, while the target residual tilts aligned modes toward a positive eigenvalue.</figcaption>
</figure>

The second order equation retains the full two-state dynamics. It is useful for visualizing why modes separate. To calculate the actual grokking time, we now return to the two first order equations and use the observed fact that their timescales are different.

#### Effective Adiabatic Approximation of Grokking

The residual relaxes faster than the eigenvalue changes. {{< footnote >}}This assumption is partly given empirically, but qualitatively the results fit the observations very well.{{< /footnote >}} Setting $\dot\sigma\simeq0$ gives

<div class="math-display">
\[
\sigma_*(\Lambda)
=-\frac{2\lambda_W y}{\Lambda+2\lambda_W}.
\]
</div>

Substituting this value as a time scale separation into the second equation leaves one slow equation:

<div class="math-display">
\[
\dot\Lambda
=2\lambda_W
\left(
\frac{a y}{\Lambda+2\lambda_W}-\Lambda
\right).
\]
</div>

The remaining residual pushes a target mode upward, while weight decay pulls every mode downward. The positive fixed point is

<div class="math-display">
\[
\Lambda_+(\lambda_W)
=-\lambda_W+\sqrt{\lambda_W^2+a y}.
\]
</div>
This is the same positive minimum shown in Figure 4. However the first order effective equation is integrable allowing us to calculate grokking time explicitly and reproduce a phase diagram.

### 2.3 Theory and the Empirical Phase Diagram

The equations above use continuous gradient-flow time $t$. A discrete run with learning rate $\eta$ and $s$ updates corresponds to $t=\eta s$. After including Weight Decay, the slow clock is $\tau=\eta\lambda_W s$. For the remainder of this section, we absorb the fixed target projection into the mode coupling and write $a$ for the product $a y$. For a budget of $T$ updates, the smallest learning rate for which $\Lambda(\eta\lambda_W T)$ reaches the threshold $\Lambda_{\rm grok}$ from initialization $\Lambda_0$ obeys

<div class="math-display">
\[
\eta_*(\lambda_W)
=\frac{1}{2\lambda_W T}
\int_{\Lambda_0}^{\Lambda_{\rm grok}}
\frac{\Lambda+2\lambda_W}
{a-\Lambda^2-2\lambda_W\Lambda}\,d\Lambda.
\]
</div>

To evaluate the integral, factor its denominator using the two fixed points

<div class="math-display">
\[
\Lambda_\pm
=\frac{-2\lambda_W\pm\sqrt{4\lambda_W^2+4a}}{2}
=-\lambda_W\pm\sqrt{\lambda_W^2+a}.
\]
</div>

The partial fraction decomposition gives

<div class="math-display">
\[
A=\frac{2\lambda_W+\Lambda_+}{\Lambda_+-\Lambda_-},
\qquad
B=\frac{2\lambda_W+\Lambda_-}{\Lambda_--\Lambda_+},
\]
</div>

so that

<div class="math-display">
\[
\frac{\Lambda+2\lambda_W}
{a-\Lambda^2-2\lambda_W\Lambda}
=-
\left(
\frac{A}{\Lambda-\Lambda_+}
+\frac{B}{\Lambda-\Lambda_-}
\right).
\]
</div>

Integration gives the explicit critical learning rate

<div class="math-display">
\[
\eta_*(\lambda_W)
=\frac{1}{2\lambda_W T}
\left[
A\log\!\left|
\frac{\Lambda_0-\Lambda_+}
{\Lambda_{\rm grok}-\Lambda_+}
\right|
+B\log\!\left|
\frac{\Lambda_0-\Lambda_-}
{\Lambda_{\rm grok}-\Lambda_-}
\right|
\right].
\]
</div>

This expression is finite only while the positive fixed point lies above the grokking threshold. Setting $\Lambda_+=\Lambda_{\rm grok}$ gives the vertical feature-learning boundary

<div class="math-display">
\[
\lambda_{\rm feature}
=\frac{a-\Lambda_{\rm grok}^2}
{2\Lambda_{\rm grok}}.
\]
</div>

These equations predict a U shaped boundary in the $(\lambda_W,\eta)$ plane. Its two limits have the forms:

<div class="math-display">
\[
\eta_*(\lambda_W)\propto\frac{1}{\lambda_W T}
\quad(\lambda_W\ \text{small}),
\qquad
\eta_*(\lambda_W)\propto
\frac{1}{\lambda_W T}
\log\!\frac{1}{\lambda_{\rm feature}-\lambda_W}
\quad(\lambda_W\to\lambda_{\rm feature}^{-}).
\]
</div>

Below the U-curve, training fits the observed examples but feature learning is too slow, so the network memorizes. Above the curve and below $\lambda_{\rm feature}$, the task modes cross $\Lambda_{\rm grok}$ within $T$ steps and the network groks. For $\lambda_W\geq\lambda_{\rm feature}$, no learning rate can raise the fixed point above the task threshold in the continuous theory, so the network remains in the memorizing or high-decay region.

A second, nearly vertical boundary comes from the fast kernel ridge regression fit. If a training mode with eigenvalue $\Lambda_{\rm fit}$ must retain a fraction $g_{\min}$ of its target, then

<div class="math-display">
\[
\frac{\Lambda_{\rm fit}}
{\Lambda_{\rm fit}+2\lambda_W}\geq g_{\min}
\quad\Longrightarrow\quad
\lambda_W\leq
\lambda_{\rm fit}
=\frac{\Lambda_{\rm fit}}{2}
\frac{1-g_{\min}}{g_{\min}}.
\]
</div>

Beyond $\lambda_{\rm fit}$, Weight Decay is too big even for the training fit. Between $\lambda_{\rm feature}$ and $\lambda_{\rm fit}$, the network can fit without sustaining the task modes. This is where memorization reaches a partly resolved forgetting/no-fitting region. Finally, discrete gradient descent requires the learning rate to remain below the edge of stability, approximately $\eta_{\rm cut}\simeq2/\alpha_{\max}$ for the largest effective curvature $\alpha_{\max}$. This adds the horizontal upper edge missing for approximating Gradient Flow.

We test this against a single $p=23$ empirical sweep. We cover an $84\times90$ grid in $(\eta,\lambda_W)$, with the learning rate running from $16$ to $1000$ and Weight Decay from $2\times10^{-7}$ to $10^{-4}$, both logarithmically spaced to observe the predicted linear Grokking boundary.{{< footnote >}}The comically large learning rates originate from averaging of the full batch regression loss producing small values.{{< /footnote >}} This gives us 7,560 runs of $T=24,000$ full-batch updates each.

Figure 5 shows the phase diagram and its **3.5 phases**. In the grokking phase, both training and validation accuracy exceed 90%. In memorization, only training accuracy does. At larger Weight Decay, some runs fit and then forget. The neighboring no-fitting label means that no fit was recorded. We therefore treat forgetting and no fitting as one partly resolved region rather than two firmly separated phases. We find remarkable agreement between our linear adiabatic Grokking boundary and the empirically observed phase diagram. Interestingly, there appears to be a small grokking island at very high values of learning rates, which has interesting training curves and should be investigated further. Both the u-shape and the linear grokking boundary fit very well. Larger time horizons would extend the grokking boundary further into the lower left corner of phase space. For a larger sweep, please view the Appendix.

<div class="phase-comparison">
  <figure class="grokking-figure">
    <img src="../../images/grokking/theory-phase-diagram.svg?v=2" alt="Theoretical phase diagram with a U-shaped grokking boundary and a forgetting or no-fitting region.">
    <figcaption><strong>Theory.</strong> Too little decay gives insufficient feature-learning time. Too much decay suppresses the task modes.</figcaption>
  </figure>
  <figure class="grokking-figure phase-experiment">
    <div class="phase-map" data-phase-map data-viewer="phase-trajectory-viewer">
      <img src="../../images/grokking/phase-diagram.svg" alt="Empirical p=23 phase diagram showing grokking, memorization, forgetting, no fitting, and a narrow high-learning-rate island.">
      <div class="phase-hotspots" role="group" aria-label="Representative empirical phase regions">
        <button type="button" class="phase-hotspot phase-hotspot--memorization" title="Show memorization trajectory" data-phase="memorization" data-trajectory-src="../../images/grokking/trajectory-memorization.svg" data-trajectory-title="Memorization" data-trajectory-alt="Training accuracy rapidly reaches one while validation accuracy remains near chance." data-trajectory-caption="At η ≈ 184 and λW ≈ 4.02×10⁻⁷, training fits almost immediately while validation remains near chance." aria-label="Show a representative memorization trajectory" aria-controls="phase-trajectory-viewer" aria-pressed="false"></button>
        <button type="button" class="phase-hotspot phase-hotspot--generalization" title="Show generalization trajectory" data-phase="generalization" data-trajectory-src="../../images/grokking/trajectory-generalization.svg" data-trajectory-title="Generalization" data-trajectory-alt="Training accuracy rises first and validation accuracy follows after a delay, with both reaching one." data-trajectory-caption="At η ≈ 184 and λW ≈ 7.04×10⁻⁶, training fits first and validation follows after a short grokking delay." aria-label="Show a representative generalization trajectory" aria-controls="phase-trajectory-viewer" aria-pressed="true"></button>
        <button type="button" class="phase-hotspot phase-hotspot--forgetting" title="Show forgetting trajectory" data-phase="forgetting" data-trajectory-src="../../images/grokking/trajectory-forgetting.svg" data-trajectory-title="Forgetting" data-trajectory-alt="Training accuracy initially reaches the phase threshold and then decays while validation remains near chance." data-trajectory-caption="At η ≈ 184 and λW ≈ 2.15×10⁻⁵, training briefly fits and then decays; validation never leaves chance." aria-label="Show a representative forgetting trajectory" aria-controls="phase-trajectory-viewer" aria-pressed="false"></button>
        <button type="button" class="phase-hotspot phase-hotspot--no-fitting" title="Show no-fitting trajectory" data-phase="no-fitting" data-trajectory-src="../../images/grokking/trajectory-no-fitting.svg" data-trajectory-title="No fitting" data-trajectory-alt="Training and validation accuracy remain close to chance throughout optimization." data-trajectory-caption="At η ≈ 184 and λW = 10⁻⁴, neither training nor validation accuracy approaches the fitting threshold." aria-label="Show a representative no-fitting trajectory" aria-controls="phase-trajectory-viewer" aria-pressed="false"></button>
        <button type="button" class="phase-hotspot phase-hotspot--high-rate" title="Show high-rate island trajectory" data-phase="high-rate-island" data-trajectory-src="../../images/grokking/trajectory-high-rate-island.svg" data-trajectory-title="High-rate island" data-trajectory-alt="Training and validation accuracy stay intermediate for most of training and rise sharply near 20000 steps." data-trajectory-caption="At η = 10³ and λW ≈ 6.12×10⁻⁶, both accuracies transition only after roughly 20,000 updates. Its mechanism remains open." aria-label="Show a representative high-rate island trajectory" aria-controls="phase-trajectory-viewer" aria-pressed="false"></button>
      </div>
    </div>
    <figcaption><strong>Experiment.</strong> Each pixel shows the outcome for one hyperparameter pair on the fixed difference split. Click on a section to show the training trajectory.</figcaption>
    <aside class="phase-trajectory" id="phase-trajectory-viewer" data-phase="generalization" aria-label="Selected representative trajectory">
      <p class="phase-trajectory__hint"><strong>Click a phase region in the map to show its trajectory.</strong></p>
      <div class="phase-trajectory__header">
        <span class="phase-trajectory__eyebrow">Trajectory</span>
        <strong class="phase-trajectory__title" data-trajectory-title>Generalization</strong>
      </div>
      <img src="../../images/grokking/trajectory-generalization.svg" data-trajectory-image alt="Training accuracy rises first and validation accuracy follows after a delay, with both reaching one.">
      <p class="phase-trajectory__caption" data-trajectory-caption aria-live="polite">At η ≈ 184 and λW ≈ 7.04×10⁻⁶, training fits first and validation follows after a short grokking delay.</p>
    </aside>
  </figure>
  <p class="phase-comparison__caption"><strong>Figure 5.</strong> Predicted and observed phase structure, compared qualitatively. Yellow represents grokking, purple memorization, and the gray high-decay region denotes forgetting/no fitting. The empirical panel uses a lighter gray to distinguish observed forgetting from no fitting. The experimental panel contains the complete fixed split grid. Experimental <a href="#setup-b">Setup B</a>.</p>
</div>

### 2.4 Inducing Grokking Through Feature Transfer

We want to test our theory by inducing the right Kernel structure from the start of the training and induce Grokking. To do this, we train a source network until it generalizes and record the label averaged and projected onto the true label space NTK, $\bar{K}_{\rm gen}$. This matrix is transferred to a fresh network through the temporary regularization

<div class="math-display">
\[
L_{\rm NTK}
=\lambda_{\rm NTK}
\lVert \bar K_\theta-\bar K_{\rm gen}\rVert_F.
\]
</div>

This encourages the new network to begin with stronger interactions along the target aligned directions. We extract the generalized NTK from a two-hidden-layer ReLU network with width 256. As a target we use a three-hidden layer network of width 128 trained on a different training-validation split. We activate the penalty only for the first 800 epochs to imitate initialization.

Across 50 different seeds, the biased networks generalize earlier. Their validation loss drops during the short regularization period and remains below the baseline after the penalty is removed. Transferring $\bar K_{\rm gen}$ therefore supports the theoretical analysis.

<figure class="grokking-figure">
  <img src="../../images/grokking/ntk-regularization.png" alt="NTK-regularized networks generalize earlier than unregularized networks across 50 runs.">
  <figcaption><strong>Figure 6.</strong> Mean and standard deviation across 50 runs. A temporary bias toward the generalized NTK produces a consistent speedup in validation accuracy and loss. <a href="#setup-a">Setup A</a>.</figcaption>
</figure>

## 3. Conclusion and Takeaways

We conclude that Grokking in homogeneous neural networks is sparked via a regularizing pressure of Weight Decay and is a gradual process where the Neural Tangent Kernel aligns with the training labels over time. In our experimental setup the abrupt transition from a memorized to a generalized state is induced via a conjoint movement of all features related to generalization yielding a sigmoid like transition. This depends on the one-hot encoding of our inputs, as it forces the same initial condition on all Fourier Modes with no bias to larger or smaller wave lengths. This is in contrast to other tasks (Image classification with CNNs), where inductive bias is helpful for generalization. Our analysis is also limited to algorithmic datasets. Grokking has however been observed in more general settings, where it might have a different origin. Our theoretical analysis is however independent of the exact group structure and could be extended to a more general theory of feature learning, where the canonical basis of the underlying task is less well known.

To summarize in three points:

- **Memorization and generalization use different NTK spectra.** A broad initial spectrum can fit observed examples, while the modular rule requires stronger equal-frequency Fourier modes.
- **Grokking Through Feature Learning.** Training accuracy saturates and Weight Decay induces a time scale on the order of $\lambda_W\cdot t$, under which the Tangent Features evolve. In these networks, Grokking occurs through actual feature learning.
- **We can enforce an inductive bias through regularization** Temporarily steering a new network toward a generalized NTK makes it generalize sooner.

## Compute

The phase sweep ran on NVIDIA A40 GPU slices.

| Sweep | Runs | Epochs per run | GPU time |
|---|---:|---:|---:|
| $p=23$ phase diagram | 7,560 | 24,000 | 46 hours |

## References

1. <span id="ref-power"></span>Alethea Power et al. [*Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*](https://arxiv.org/abs/2201.02177), 2022.
2. <span id="ref-xu"></span>Mingyue Xu, Gal Vardi, and Itay Safran. [*To Grok Grokking: Provable Grokking in Ridge Regression*](https://arxiv.org/abs/2601.19791), ICML 2026.
3. <span id="ref-nanda"></span>Neel Nanda et al. [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217), 2023.
4. <span id="ref-doshi"></span>Darshil Doshi et al. [*To Grok or Not to Grok: Disentangling Generalization and Memorization on Corrupted Algorithmic Datasets*](https://arxiv.org/abs/2310.13061), 2024.
5. <span id="ref-liu"></span>Ziming Liu et al. [*Towards Understanding Grokking: An Effective Theory of Representation Learning*](https://arxiv.org/abs/2205.10343), 2022.
6. <span id="ref-jacot"></span>Arthur Jacot, Franck Gabriel, and Clément Hongler. [*Neural Tangent Kernel: Convergence and Generalization in Neural Networks*](https://arxiv.org/abs/1806.07572), 2018.
7. <span id="ref-huang"></span>Jiaoyang Huang and Horng-Tzer Yau. [*Dynamics of Deep Neural Networks and Neural Tangent Hierarchy*](https://arxiv.org/abs/1909.08156), 2020.
---

## Cite This Work

Lenz Pracher, Pascal de Jong, Oskar Liesaus, Alan Jeffares, and Steffen Rulands. “A Spectral Theory of Grokking.” 2026.

```bibtex
@misc{pracher2026grokking,
  title  = {A Spectral Theory of Grokking},
  author = {Pracher, Lenz and de Jong, Pascal and Liesaus, Oskar and Jeffares, Alan and Rulands, Steffen},
  year   = {2026},
  url    = {https://lenzpracher.github.io/projects/grokking/}
}
```

---

## Comments

<div class="comments-section">
<script src="https://giscus.app/client.js"
        data-repo="lenzpracher/lenzpracher.github.io"
        data-repo-id="R_kgDON9MmFA"
        data-category="Announcements"
        data-category-id="DIC_kwDON9MmFM4DCT5u"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="dark_dimmed"
        data-lang="en"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>
</div>

---

## Appendix

### A.1 Experimental Setups

Two different models appear in this post. We used preliminary experiments (Setup A) that motivated the theory, which then promoted a simplified setup (Setup B). The phase maps use the simplified Setup B, while the rest of the figures used Setup A.

<span id="setup-a"></span>**Setup A — preliminary experiments.** Modular addition on $\mathbb Z_p$ with $p=97$. The symbols $a$ and $b$ are one-hot encoded and concatenated, $(e_a\oplus e_b)\in\mathbb R^{2p}$, and the target is the one-hot encoding of $f(a,b)$ in $\mathbb R^{p}$. Commutativity leaves $p(p+1)/2$ unique mappings, which make up the training and validation set. The training and validation sets are formed by drawing a fraction $r=0.5$ of those unique mappings. The model used is a bias-free ReLU MLP with two hidden layers of width 256 under HeNormal initialization, trained by full-batch gradient descent with Weight Decay for 20,000 epochs. The loss used was a weighted squared error in which the true label coordinate carries weight $d_O-1$ and each of the $d_O-1$ off-label coordinates carries weight 1, so that true and off-label terms contribute equal total mass. The kernel shown is the label-averaged projected NTK onto the true labels. The label averaging works by averaging the NTK samples, which have the same label. For example $(a_1,b_1) = (2,3)$ would be averaged with $(a_2,b_2)=(4,1)$. This was done to extract the label wise structure. The transfer experiment trains a teacher of two hidden layers of width 256 to generalization, then adds $\lambda_{\rm focus}\lVert\bar G_S-\bar G_T\rVert_F$ to the loss of a student with three hidden layers of width 128 for the first 800 epochs, with a different split, a different initialization distribution and different hyperparameters, repeated over 50 runs.

<span id="setup-b"></span>**Setup B — simplified model.** Modular addition on $\mathbb Z_{23}$, with both orders of every pair retained, so the kernel is evaluated on all $23^2=529$ input pairs. The symbols $a$ and $b$ are one-hot encoded and the target is the one-hot encoding from the sum, where the input is concatenated. The model used is a bias-free ReLU network with a single hidden layer of width 256, which makes it degree two homogeneous, trained by full-batch gradient descent on the plain regular squared loss with L2 Weight Decay for 24,000 updates. The training and validation split is structured by input differences. 16 residues $D\subset\mathbb Z_{23}$ are drawn once and training uses $\{(a,b):a-b\bmod 23\in D\}$, giving 368 training and 161 held-out pairs at $r=16/23$. One fixed split and one model seed are used at every grid point. 

The two differ mostly through choice of the loss. We found that regular regression loss also works, while Setup A chose a loss to imitate cross entropy loss. The train and validation split is also simplified to be compatible with the Fourier basis.

### A.2 The Expanded Phase Map

<figure class="grokking-figure grokking-figure--wide">
  <img src="../../images/grokking/phase-diagram-expanded.svg" alt="Expanded p=23 phase map covering learning rates from 0.16 to 100000 and weight decays from 2e-9 to 1e-2, with the main-text grid marked and the left edge of the grokking region fitted.">
  <figcaption><strong>Figure A1.</strong> Extended phase diagram. One can observe that the grokking island above the stability threshold extends further to the left and also produces a band.{{< footnote >}} This region could be investigated in further research and might depend on second order approximations of Gradient Flow.{{< /footnote >}} The white curve fits the left edge as η ∝ λW<sup>−α</sup> with α = 1.07. <a href="#setup-b">Setup B</a>.</figcaption>
</figure>
