# Generative Deep Learning

## Learning Objectives

- Understand the mathematical foundations of generative modeling.
- Implement autoencoders and variational autoencoders using PyTorch.
- Understand adversarial learning, GAN objectives, and training dynamics.
- Explain diffusion models and implement simplified diffusion pipelines.
- Connect probabilistic objectives and optimization algorithms to practical implementations.
- Evaluate generative models using reconstruction quality, sample diversity, and appropriate metrics.

## Mathematical Foundations

### 1. Autoencoders

An autoencoder learns a compressed representation of an input and reconstructs it.

**Encoder**

$$
z = f_{\theta}(x)
$$

**Decoder**

$$
\hat{x} = g_{\phi}(z)
$$

**Mean Squared Error (MSE)**

$$
L_{\mathrm{reconstruction}}
=
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
$$

The objective is to minimize reconstruction error while learning a useful latent representation.

### 2. Variational Autoencoders (VAEs)

A VAE learns a probability distribution over latent representations instead of mapping each input to a single deterministic vector.

**Approximate posterior**

$$
q_{\phi}(z \mid x)
=
\mathcal{N}
\left(
z;\mu_{\phi}(x),
\operatorname{diag}(\sigma_{\phi}^{2}(x))
\right)
$$

**Reparameterization trick**

$$
z = \mu + \sigma \odot \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

**Evidence Lower Bound (ELBO) objective**

$$
L_{\mathrm{VAE}}
=
-\mathbb{E}_{q_{\phi}(z\mid x)}
[\log p_{\theta}(x\mid z)]
+
D_{\mathrm{KL}}
\left(
q_{\phi}(z\mid x)
\parallel p(z)
\right)
$$

The first term encourages accurate reconstruction. The KL divergence regularizes the learned latent distribution toward the prior, commonly a standard normal distribution.

### 3. Generative Adversarial Networks (GANs)

A GAN trains two neural networks with competing objectives.

**Original minimax objective**

$$
\min_G \max_D
\left[
\mathbb{E}_{x\sim p_{\mathrm{data}}}
[\log D(x)]
+
\mathbb{E}_{z\sim p_z}
[\log(1-D(G(z)))]
\right]
$$

The discriminator distinguishes real samples from generated samples, while the generator learns to produce samples that the discriminator classifies as real.

### 4. GAN Training Stability

GAN optimization can be unstable because the generator and discriminator continuously adapt to one another.

**Wasserstein objective**

$$
\min_G \max_{D\in\mathcal{D}}
\left[
\mathbb{E}_{x\sim p_{\mathrm{data}}}[D(x)]
-
\mathbb{E}_{z\sim p_z}[D(G(z))]
\right]
$$

Here, the critic approximates a Wasserstein-distance objective under a 1-Lipschitz constraint. Practical implementations must enforce or encourage this constraint, for example through gradient penalty.

### 5. Diffusion Models

Diffusion models learn to reverse a gradual process of adding noise to data.

**Forward diffusion**

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right)
$$

Define the cumulative product:

$$
\bar{\alpha}_t
=
\prod_{s=1}^{t}(1-\beta_s)
$$

The noisy sample can be generated directly from the original sample:

$$
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

**Noise prediction objective**

$$
L_{\mathrm{diffusion}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_{\theta}(x_t,t)
\right\|_2^2
\right]
$$

The denoising network learns to predict the noise added at a randomly selected timestep. Sampling then uses the learned model to progressively generate data from noise.

## Core Concepts

| Concept                         | Key Topics                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| Autoencoders                    | Encoder-decoder architectures, latent representations, reconstruction loss         |
| Variational Autoencoders        | Probability distributions, reparameterization, ELBO, KL divergence                 |
| Generative Adversarial Networks | Generator, discriminator, minimax objective, adversarial optimization              |
| GAN Training Stability          | Mode collapse, gradient behavior, Wasserstein objectives, gradient penalty         |
| Diffusion Models                | Forward noising, noise schedules, reverse denoising, noise prediction              |
| Generative Evaluation           | Reconstruction quality, sample diversity, latent structure, evaluation limitations |

## Implementation and Experiments

| File                                    | Experiment                                                                       |
| --------------------------------------- | -------------------------------------------------------------------------------- |
| `01_autoencoder_fundamentals.py`        | Build and train an autoencoder for image reconstruction                          |
| `02_variational_autoencoder.py`         | Implement a VAE with latent sampling and a probabilistic loss                    |
| `03_gan_fundamentals.py`                | Build a generator and discriminator and train them adversarially                 |
| `04_gan_training_stability.py`          | Investigate GAN instability, gradient behavior, and Wasserstein-style objectives |
| `05_diffusion_fundamentals.py`          | Implement forward diffusion, noise scheduling, and noise prediction              |
| `06_diffusion_training_and_sampling.py` | Train a compact denoising model and generate samples through reverse diffusion   |

## Expected Outcome

After completing this module, you should be able to:

- Derive and explain the core objectives of autoencoders, VAEs, GANs, and diffusion models.
- Translate mathematical objectives into differentiable PyTorch loss functions.
- Explain latent-variable sampling, the reparameterization trick, and KL-divergence regularization.
- Understand adversarial optimization, Wasserstein objectives, and common GAN failure modes.
- Explain the forward and reverse processes used by diffusion models.
- Build reproducible experiments and evaluate generative model quality and limitations.
