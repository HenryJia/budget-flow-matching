# Henry's Diffusion Classroom

So, I previously worked at Midjourney, but decided to do a PhD in something completely different for 3 years. As such, I'm a little rusty these days.

As always, a good way to polish off the rust and brush up on everything is to try and build stuff from scratch.

So, this repository contains a set of well commented implementations, built to run reasonably efficiently whilst being as minimal as possible.

Please feel free to use this repository as a reference for building your own diffusion models, or just to understand how they work. The code should be readable and well commented, so it should be easy to follow along.

Do note that whilst the different implementations share various bits of code, they are all implemented independently to improve readability of each implementation. So, there will be a lot of code duplication.

Acknowledgements: I want to give a shoutout to Sully, an old friend and former researcher at OpenAI, who got me started and gave me a list of papers to read and implement. I was also guided in what papers to read by the good people of the Eleuther.AI discord and the Yannic Kilcher discord.


# Implementations

# Flow Matching for Generative Modeling
### Our code is in `cfm` [Link](https://github.com/HenryJia/diffusion-classroom/tree/main/cfm)
#### [ArXiv](https://arxiv.org/abs/2210.02747)

## Denoising Diffusion Probabilistic Models (DDPM)
### Our code is in `ddpm/` [Link](https://github.com/HenryJia/diffusion-classroom/tree/main/ddpm)
#### [ArXiv](https://arxiv.org/abs/2006.11239) and [Author's implementation](https://github.com/hojonathanho/diffusion/tree/master)

## Deep Unsupervised Learning using Nonequilibrium Thermodynamics
### Our code is in `nonequilibrium-thermodynamics/` [Link](https://github.com/HenryJia/diffusion-classroom/tree/main/nonequilibrium-thermodynamics)
#### [ArXiv](https://arxiv.org/abs/1503.03585) and [Author's implementation](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master)