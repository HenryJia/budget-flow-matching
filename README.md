# Henry's Diffusion Classroom

So, I previously worked at Midjoruney, but decided to do a PhD in something completely different for 3 years. As such, I'm a little rusty these days.

As always, a good way to polish off the rust and brush up on everything is to try and build stuff from scratch.

So, this repository contains a set of well commented implementations, as close to the reference implementation as is reasonable of various keystone diffusion papers.

In case anyone wants to know where this list came from, it was given to me by Sully, an old friend of mine who previously worked at OpenAI.

## List of papers
- [x] [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](#deep-unsupervised-learning-using-nonequilibrium-thermodynamics) [ArXiv](https://arxiv.org/abs/1503.03585) [Author's implementation](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master)
- [x] [Denoising Diffusion Probabilistic Models (DDPM)](#denoising-diffusion-probabilistic-models-ddpm) [ArXiv](https://arxiv.org/abs/2006.11239)
- [ ] Generative Modeling by Estimating Gradients of the Data Distribution [ArXiv](https://arxiv.org/abs/1907.05600)
- [ ] Score-Based Generative Modeling through Stochastic Differential Equations [ArXiv](https://arxiv.org/abs/2011.13456)
- [ ] Maximum Likelihood Training of Score-Based Diffusion Models [ArXiv](https://arxiv.org/abs/2101.09258)
- [ ] Improved Denoising Diffusion Probabilistic Models [ArXiv](https://arxiv.org/abs/2102.09672)
- [ ] Denoising Diffusion Implicit Models (DDIM) [ArXiv](https://arxiv.org/abs/2010.02502)
- [ ] DPM-Solver [ArXiv](https://arxiv.org/abs/2206.00927)
- [ ] Diffusion Models Beat GANs on Image Synthesis [ArXiv](https://arxiv.org/abs/2105.05233)
- [ ] Classifier-Free Diffusion Guidance [ArXiv](https://arxiv.org/abs/2207.12598)
- [ ] High-Resolution Image Synthesis with Latent Diffusion Models [ArXiv](https://arxiv.org/abs/2112.10752)


# Implementation Notes and Empirical Observations

## Deep Unsupervised Learning using Nonequilibrium Thermodynamics
### Our code is in `nonequilibrium-thermodynamics/`

#### Henry's rating for this paper: 3/10 - honestly, absolutely awful paper to try and implement. But, the mathematical framework is extremely useful for getting a good grasp of the foundations of diffusion models, so it's worth it for that.

This paper is interesting. The maths is very focused on figuring out a lowerbound kind of like the variational lowerbound in VAEs. But, there are a fair few problems. For starters, the architecture they use also feels quite overcomplicated for the toy problems which they train on (CIFAR and MNIST).

The training set up is also kind of hacky. There's a fair few calculations they use in the reference implementation which aren't mentioned in the paper. whilst they're not too complicated to derive by hand, it is a bit of a pain. There's also some tricks they use to make the thing train, which are mentioned in the paper but aren't really explained as to why.

Training this thing is still an absolute pain in the arse. Normally models on MNIST can train in a few epochs, but this one takes like 200 to be able to generate anything vaguely resembling a digit. It also seems to be quite particular about the learning rate. Too high and it seems to struggle to converge. You go from one epoch where it generates noisy trash to the next where it seems to be vaguely trying to generate a digit. Set the learning rate too low and it optimises very slowly.

A workaround seems to be to use a exponentially decaying learning rate, so that later on it finetunes the model.

My intuition of why this is the case is that the noise deviations the model is trying to learn are very small. So a larger update step will throw the model optimisation off so far that it might effectively have teleported somewhere alien in the loss landscape, and has to find a whole new local minima to converge to.

I think this also explains why training takes so long. The old reference implementation is in Theano and they run for around 850 epochs. I think the finetuning element of this diffusion model is just a slow and tedious process.

This being said, there is one interesting aspect of the loss function for this paper. It's clear that the KL term is the main driver which minimises the denoising error. But, the entropy terms are interesting. If we initialise the diffusion rate betas to be smaller initially, the term H_q(X_T | X_0), the entropy of the noise distribution at the end, will be smaller than a standard Gaussian. This means that the model will slowly increase the diffusion rate betas to eventually match the entropy of a standard Gaussian. This is interesting because it means that the model is effectively learning to increase the noise level over time, which is kind of like a curriculum learning strategy. The model starts off with an easier task of denoising less noisy images, and then gradually increases the difficulty as it learns.

All in all, implementing this thing is a neat exercise, but it kind of sucks. It's overcomplicated and kind of fragile to train. It's not practical, but it is useful for understanding the variational lowerbound.

Here's a sample of the generated MNIST digits after 1000 epochs: (TODO add sample)

## Denoising Diffusion Probabilistic Models (DDPM)
### Our code is in `ddpm/`

#### Henry's rating for this paper: 10/10 - I fucking love this paper

This paper is honestly a lot better written than the nonequilibrium thermodynamics one as far as implementation is concerned. The maths is a lot more straightforward, and the training process is a lot more stable. Granted though, it doesn't cover the mathematical background as much.

The training set up is much more trivial, and the calculations for doing so are clear in the paper. I didn't even have to look at the reference implementation to figure out how to train the model, which is a rarity these days.

Training for this thing runs much faster than the nonequilibrium thermodynamics model. On MNIST, it seems to mostly converge within the first few epochs. It takes a handful of epochs to be able to generate vaguely digit like things, unlike the nonequilibrium thermodynamics model which takes over 100 epochs to do so.

My big intuition for the difference between this paper vs the nonequilibrium paper is that the nonequilibrium paper is trying to learn the reverse transition between x_t and x_{t-1}, whereas the DDPM paper is trying to learn the reverse transition between x_t and x_0. This means that the DDPM model is effectively learning to denoise all the way back to the original image. The DDPM paper takes advantage of the fact that there is a simple closed form method from converting the transition between x_t to x_0 to x_t to x_{t-1}.

One might think that learning the transition between x_t and x_0 is harder than learning the transition between x_t and x_{t-1}, but in practice this is the opposite. Years ago, when I worked with trying to do high FPS neural SLAM, we found that higher FPS were harder to train than lower FPS. This is because the difference between each frame becomes so small that the signal to noise ratio becomes very low. The model update process effectively has to have very high sensitivity and "noise filtering" ability. The same I believe applies here. Given that the trajectory length is 1000 or more, the difference between x_t and x_{t-1} is very small, possibly actually quite close to the noise floor in the image.

This is also compounded by the fact that we're recursively applying the model at each diffusion step. If the model transition is "off" by a small amount, this may be compounded over the diffusion trajectory to produce a nonsensical image.

If we want to put on an intuitive statistical lens, the transition between x_t and x_{t-1} might be a lot noisier. Since x_t to x_0 is inherently an aggregate of the former, by law of large numbers, some of the noise starts to disappear. This gives a higher signal to noise ratio for the model to learn from, which makes training more stable and faster.