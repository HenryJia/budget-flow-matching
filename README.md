# Henry's Diffusion Classroom

So, I previously worked at Midjoruney, but decided to do a PhD in something completely different for 3 years. As such, I'm a little rusty these days.

As always, a good way to polish off the rust and brush up on everything is to try and build stuff from scratch.

So, this repository contains a set of well commented implementations, as close to the reference implementation as is reasonable of various keystone diffusion papers.

In case anyone wants to know where this list came from, it was given to me by Sully, an old friend of mine who previously worked at OpenAI.

## List of papers
- [x] Deep Unsupervised Learning using Nonequilibrium Thermodynamics [ArXiv](https://arxiv.org/abs/1503.03585) [Author's implementation](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master)
    * Our code is in `nonequilibrium-thermodynamics/`
- [] Denoising Diffusion Probabilistic Models (DDPM) [ArXiv](https://arxiv.org/abs/2006.11239)
- [] Generative Modeling by Estimating Gradients of the Data Distribution [ArXiv](https://arxiv.org/abs/1907.05600)
- [] Score-Based Generative Modeling through Stochastic Differential Equations [ArXiv](https://arxiv.org/abs/2011.13456)
- [] Maximum Likelihood Training of Score-Based Diffusion Models [ArXiv](https://arxiv.org/abs/2101.09258)
- [] Improved Denoising Diffusion Probabilistic Models [ArXiv](https://arxiv.org/abs/2102.09672)
- [] Denoising Diffusion Implicit Models (DDIM) [ArXiv](https://arxiv.org/abs/2010.02502)
- [] DPM-Solver [ArXiv](https://arxiv.org/abs/2206.00927)
- [] Diffusion Models Beat GANs on Image Synthesis [ArXiv](https://arxiv.org/abs/2105.05233)
- [] Classifier-Free Diffusion Guidance [ArXiv](https://arxiv.org/abs/2207.12598)
- [] High-Resolution Image Synthesis with Latent Diffusion Models [ArXiv](https://arxiv.org/abs/2112.10752)

# Notes and Empirical Observations

## Deep Unsupervised Learning using Nonequilibrium Thermodynamics

This paper is interesting. The maths is very focused on figuring out a lowerbound kind of like the variational lowerbound in VAEs. But, there are a fair few problems. For starters, the architecture they use also feels quite overcomplicated for the toy problems which they train on (CIFAR and MNIST).

The training set up is also kind of hacky. There's a fair few calculations they use in the reference implementation which aren't mentioned in the paper. whilst they're not too complicated to derive by hand, it is a bit of a pain. There's also some tricks they use to make the thing train, which are mentioned in the paper but aren't really explained as to why.

Training this thing is still an absolute pain in the arse. Normally models on MNIST can train in a few epochs, but this one takes like 200 to be able to generate anything vaguely resembling a digit. It also seems to be quite particular about the learning rate. Too high and it seems to struggle to converge. You go from one epoch where it generates noisy trash to the next where it seems to be vaguely trying to generate a digit. Set the learning rate too low and it optimises very slowly.

A workaround seems to be to use a exponentially decaying learning rate, so that later on it finetunes the model.

My intuition of why this is the case is that the noise deviations the model is trying to learn are very small. So a larger update step will throw the model optimisation off so far that it might effectively have teleported somewhere alien in the loss landscape, and has to find a whole new local minima to converge to.

I think this also explains why training takes so long. The old reference implementation is in Theano and they run for around 850 epochs. I think the finetuning element of this diffusion model is just a slow and tedious process.

This being said, there is one interesting aspect of the loss function for this paper. It's clear that the KL term is the main driver which minimises the denoising error. But, the entropy terms are interesting. If we initialise the diffusion rate betas to be smaller initially, the term H_q(X_T | X_0), the entropy of the noise distribution at the end, will be smaller than a standard Gaussian. This means that the model will slowly increase the diffusion rate betas to eventually match the entropy of a standard Gaussian. This is interesting because it means that the model is effectively learning to increase the noise level over time, which is kind of like a curriculum learning strategy. The model starts off with an easier task of denoising less noisy images, and then gradually increases the difficulty as it learns.

All in all, implementing this thing is a neat exercise, but it kind of sucks. It's overcomplicated and kind of fragile to train. It's not practical, but it is useful for understanding the variational lowerbound.

Here's a sample of the generated MNIST digits after 1000 epochs: (TODO add sample)

