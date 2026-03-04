# Henry's Diffusion Classroom

So, I previously worked at Midjourney, but decided to do a PhD in something completely different for 3 years. As such, I'm a little rusty these days.

As always, a good way to polish off the rust and brush up on everything is to try and build stuff from scratch.

So, this repository contains a set of well commented implementations, built to run reasonably efficiently whilst being as minimal as possible.

Please feel free to use this repository as a reference for building your own diffusion models, or just to understand how they work. The code should be readable and well commented, so it should be easy to follow along.

Do note that whilst the different implementations share various bits of code, they are all implemented independently to improve readability of each implementation. So, there will be a lot of code duplication.

Acknowledgements: I want to give a shoutout to Sully, an old friend and former researcher at OpenAI, who got me started and gave me a list of papers to read and implement. I was also guided in what papers to read by the good people of the Eleuther.AI discord and the Yannic Kilcher discord.


## Implementations
- [x] [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](#deep-unsupervised-learning-using-nonequilibrium-thermodynamics) )
- [x] [Denoising Diffusion Probabilistic Models (DDPM)](#denoising-diffusion-probabilistic-models-ddpm) 
- [ ] Flow Matching for Generative Modeling [ArXiv](https://arxiv.org/abs/2210.02747)
- [ ] Generative Modeling by Estimating Gradients of the Data Distribution [ArXiv](https://arxiv.org/abs/1907.05600)
- [ ] Score-Based Generative Modeling through Stochastic Differential Equations [ArXiv](https://arxiv.org/abs/2011.13456)
- [ ] Maximum Likelihood Training of Score-Based Diffusion Models [ArXiv](https://arxiv.org/abs/2101.09258)
- [ ] Improved Denoising Diffusion Probabilistic Models [ArXiv](https://arxiv.org/abs/2102.09672)
- [ ] Denoising Diffusion Implicit Models (DDIM) [ArXiv](https://arxiv.org/abs/2010.02502)
- [ ] DPM-Solver [ArXiv](https://arxiv.org/abs/2206.00927)
- [ ] Diffusion Models Beat GANs on Image Synthesis [ArXiv](https://arxiv.org/abs/2105.05233)
- [ ] Classifier-Free Diffusion Guidance [ArXiv](https://arxiv.org/abs/2207.12598)
- [ ] High-Resolution Image Synthesis with Latent Diffusion Models [ArXiv](https://arxiv.org/abs/2112.10752)


# Implementations

## Denoising Diffusion Probabilistic Models (DDPM)
### Our code is in `ddpm/` [Link](https://github.com/HenryJia/diffusion-classroom/tree/main/ddpm)
#### [ArXiv](https://arxiv.org/abs/2006.11239) [Author's implementation](https://github.com/hojonathanho/diffusion/tree/master)

## Deep Unsupervised Learning using Nonequilibrium Thermodynamics
### Our code is in `nonequilibrium-thermodynamics/` [Link](https://github.com/HenryJia/diffusion-classroom/tree/main/nonequilibrium-thermodynamics)
#### [ArXiv](https://arxiv.org/abs/1503.03585) [Author's implementation](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master)




This paper is honestly a lot better written than the nonequilibrium thermodynamics one as far as implementation is concerned. The maths is a lot more straightforward, and the training process is a lot more stable. Granted though, it doesn't cover the mathematical background as much.

The training set up is much more trivial, and the calculations for doing so are clear in the paper. I didn't even have to look at the reference implementation to figure out how to train the model, which is a rarity these days.

Training for this thing runs much faster than the nonequilibrium thermodynamics model. On MNIST, it seems to mostly converge within the first few epochs. It takes a handful of epochs to be able to generate vaguely digit like things, unlike the nonequilibrium thermodynamics model which takes over 100 epochs to do so.

My big intuition for the difference between this paper vs the nonequilibrium paper is that the nonequilibrium paper is trying to learn the reverse transition between x_t and x_{t-1}, whereas the DDPM paper is trying to learn the reverse transition between x_t and x_0. This means that the DDPM model is effectively learning to denoise all the way back to the original image. The DDPM paper takes advantage of the fact that there is a simple closed form method from converting the transition between x_t to x_0 to x_t to x_{t-1}.

One might think that learning the transition between x_t and x_0 is harder than learning the transition between x_t and x_{t-1}, but in practice this is the opposite. Years ago, when I worked with trying to do high FPS neural SLAM, we found that higher FPS were harder to train than lower FPS. This is because the difference between each frame becomes so small that the signal to noise ratio becomes very low. The model update process effectively has to have very high sensitivity and "noise filtering" ability. The same I believe applies here. Given that the trajectory length is 1000 or more, the difference between x_t and x_{t-1} is very small, possibly actually quite close to the noise floor in the image.

This is also compounded by the fact that we're recursively applying the model at each diffusion step. If the model transition is "off" by a small amount, this may be compounded over the diffusion trajectory to produce a nonsensical image.

If we want to put on an intuitive statistical lens, the transition between x_t and x_{t-1} might be a lot noisier. Since x_t to x_0 is inherently an aggregate of the former, by law of large numbers, some of the noise starts to disappear. This gives a higher signal to noise ratio for the model to learn from, which makes training more stable and faster.

Another key difference between this paper and the nonequilibrium thermodynamics paper is that the former has fixed beta diffusion rates whereas the latter attempts to learn them. As previously stated, the former proves superior in practice, even though the latter uses something that should make learning easier if anything. My best guess is just that the aged old rule of better engineering is less engineering applies. The fixed beta diffusion rates just mean there's less moving parts in the model, so things are easier to build.

Here is a sample of the generated faces from CelebA-HQ after 200 epochs: (TODO add sample)