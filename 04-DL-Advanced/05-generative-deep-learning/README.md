# Generative Deep Learning

## Learning Objectives

- Understand the fundamental principles of generative modeling.
- Implement autoencoders and variational autoencoders using PyTorch.
- Understand adversarial learning and GAN training dynamics.
- Explain diffusion models and implement simplified diffusion pipelines.
- Evaluate generative models using reconstruction quality, sample diversity, and appropriate metrics.
- Understand the optimization challenges and architectural trade-offs of generative models.

## Core Concepts

| Concept                         | Key Topics                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| Autoencoders                    | Encoder-decoder architectures, latent representations, reconstruction loss          |
| Variational Autoencoders        | Latent probability distributions, reparameterization trick, KL divergence, sampling |
| Generative Adversarial Networks | Generator, discriminator, adversarial objectives, alternating optimization          |
| GAN Training Stability          | Mode collapse, gradient behavior, Wasserstein objectives                            |
| Diffusion Models                | Forward noising, reverse denoising, noise prediction, sampling                      |
| Generative Evaluation           | Reconstruction quality, sample diversity, latent structure, evaluation limitations  |

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

- Explain how major generative modeling architectures work.
- Implement autoencoders, VAEs, GANs, and simplified diffusion models in PyTorch.
- Understand latent representations, probabilistic objectives, and adversarial optimization.
- Diagnose common training challenges and evaluate generative model behavior.
- Build reproducible experiments for training, sampling, and comparing generative models.
