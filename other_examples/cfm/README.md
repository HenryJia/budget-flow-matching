# Flow Matching for Generative Modeling
### [ArXiv](https://arxiv.org/abs/2210.02747)

## Sample on CelebA-HQ256

![CelebA-HQ256 Sample](https://github.com/HenryJia/diffusion-classroom/blob/main/cfm/samples/epoch_999.png?raw=true "CelebA-HQ256 Sample after 1000 epochs")

## Henry's Implementation Notes

This paper is interesting. In implementation, it's even simpler than the DDPM paper to implement, which is already relatively simple.

However, the mathematics gets more complicated. The paper goes very deep on the mathematical background and shows that the DDPM is actually a special case of flow models. The flow matching model they do use is a specific implementation of the general idea of flow matching, and is very simple.

It also seems to sample much faster. The first order differential equation solver seems to be able to sample it much quicker than DDPMs. My best guess is that DDPMs force you to take 1000 sampling steps. However, if the underlying flow is actually much simpler, the diffusion equation solver can take a much more efficient and aggressive sampling path to integrate the ODE.

It also seems to be much more stable to train. This is probably because again, it's simpler, and better engineering is less engineering. It's also much easier to debug as it has very few moving parts.