import torch
import math
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

from sentence_transformers import SentenceTransformer

from diffusers import SanaTransformer2DModel


class PromptEncoderWrapper(nn.Module):
    def __init__(self, encoder, tokeniser=None):
        super().__init__()
        self.tokeniser = tokeniser
        if self.tokeniser is not None:
            self.tokeniser.padding_side = "right"
            self.tokeniser.pad_token = self.tokeniser.eos_token

        self.encoder = encoder

    def forward(self, prompts):
        if self.tokeniser is not None:
            tokens = self.tokeniser(prompts, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)

            for k in tokens:
                if hasattr(tokens[k], "to"):
                    tokens[k] = tokens[k].to(device=self.encoder.device)

            encoding = self.encoder(**tokens)
            return encoding.last_hidden_state, tokens.attention_mask
        else:
            assert isinstance(self.encoder, SentenceTransformer), "If no tokeniser is provided, the encoder must be a SentenceTransformer"
            encoding = self.encoder.encode(prompts, convert_to_tensor=True)

            return encoding, None

class LatentDiffusionModel(L.LightningModule):
    def __init__(
            self, latent_dim, latent_channels, autoencoder,
            trajectory_length, lr, prompt_encoder=None, prompt_dim=768
            ):
        super(LatentDiffusionModel, self).__init__()
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.autoencoder = autoencoder
        self.trajectory_length = trajectory_length
        self.lr = lr
        self.prompt_encoder = prompt_encoder
        self.prompt_dim = prompt_dim

        # Load Sana 600M config
        config = SanaTransformer2DModel.load_config(
            "Efficient-Large-Model/Sana_600M_1024px_diffusers", 
            subfolder="transformer"
        )

        # Reduce depth
        config["num_layers"] = 12

        # Reduce width
        config["num_attention_heads"] = 12
        config["attention_head_dim"] = 64
        config["cross_attention_dim"] = 768
        config["num_cross_attention_heads"] = 12
        config["cross_attention_head_dim"] = 64

        config["caption_channels"] = prompt_dim

        self.reverse_diffusion_net = SanaTransformer2DModel.from_config(config)

        # Interestingly, unlike the nonequilibrium themodynamics paper, the betas are NOT learnable
        # We will use the same fixed beta schedule as described in section 4 of the paper
        self.beta = nn.parameter.Buffer(torch.linspace(start=1e-4, end=0.02, steps=trajectory_length))
        self.alpha = nn.parameter.Buffer(1 - self.beta)
        self.alpha_bar = nn.parameter.Buffer(torch.cumprod(self.alpha, dim=0))

    def forward_diffusion(self, x_0, t):
        alpha_bar = self.alpha_bar[t]

        x_0 = x_0.to(dtype=self.autoencoder.dtype)
        latent = self.autoencoder.encode(x_0).latent
        latent = latent.to(dtype=self.dtype)

        epsilon_forward = torch.randn_like(latent)
        latent = latent * torch.sqrt(alpha_bar)[:, None, None, None] + epsilon_forward * torch.sqrt(1 - alpha_bar)[:, None, None, None]

        return latent, epsilon_forward

    def reverse_diffusion(self, latent_t, t, prompt_embeddings, prompt_mask):
        # Huggingface will take care of generating the time embedding fo us
        out = self.reverse_diffusion_net(
            hidden_states=latent_t,
            encoder_hidden_states=prompt_embeddings,
            encoder_attention_mask=prompt_mask,
            timestep=t.to(dtype=latent_t.dtype)).sample

        return out

    def sample(self, x_t, t, prompt_embeddings, prompt_mask):
        # Sample a single step of the reverse diffusion process as described in Algorithm 2 of the paper
        epsilon_reverse = self.reverse_diffusion(x_t, t, prompt_embeddings, prompt_mask)

        alpha_bar = self.alpha_bar[t]
        alpha_t = self.alpha[t]
        beta_t = self.beta[t]

        # Described in section 3.2, we can either choose
        #beta_tilde = (1 - alpha_bar / alpha_t) / (1 - alpha_bar) * beta_t # equation 7
        #sigma2_t = beta_tilde
        sigma2_t = beta_t

        coef = torch.pow(alpha_t, -0.5)
        coef_eps = beta_t / torch.sqrt(1 - alpha_bar)
        out = coef[:, None, None, None] * (x_t - coef_eps[:, None, None, None] * epsilon_reverse)

        # Note: The paper doesn't mention clamping the output at all
        # But it is done in the reference implementation
        # From my experiments, it seems to be crucial otherwise the colour hue can drift in very weird ways
        z = (t > 0)[:, None, None, None] * torch.sqrt(sigma2_t)[:, None, None, None] * torch.randn_like(out)
        out = out + z
        return out

    def training_step(self, batch, batch_idx):
        x = batch[0]

        t = torch.randint(low=0, high=self.trajectory_length, size=(x.shape[0],), device=x.device)

        # The autoencoder in the forward diffusion process is not trained
        # It is also ungodly expensive, so we really don't want PyTorch to be tracking gradients through it
        with torch.no_grad():
            x_t, epsilon_forward = self.forward_diffusion(x, t)

        # Whilst the derivation to get here takes a bit more work, all we need is to predict the epsilons when running in reverse
        # This is based on Algorithm 1 in the ddpm paper
        with torch.no_grad():
            if self.prompt_encoder is not None:
                prompt = batch[1]
                prompt_embeddings, prompt_mask = self.prompt_encoder(prompt)
                prompt_embeddings = prompt_embeddings.to(dtype=self.dtype)
                if len(prompt_embeddings.shape) == 2:
                    prompt_embeddings = prompt_embeddings[:, None, :]
                if prompt_mask is not None:
                    prompt_mask = prompt_mask.to(dtype=self.dtype)
            else:
                prompt_embeddings = torch.zeros((x.shape[0], 1, self.prompt_dim), device=x.device)
                prompt_mask = None

            if len(batch) == 3:
                size = batch[2][:, None, :].expand((-1, prompt_embeddings.shape[1], -1))
                prompt_embeddings = torch.cat([prompt_embeddings, size.to(dtype=self.dtype)], dim=-1).detach()


        epsilon_reverse  = self.reverse_diffusion(x_t, t, prompt_embeddings, prompt_mask)

        # Based on equation 14 and its accompanying explanation, we can do this simple loss or the more complicated one in equation 12
        # The paper suggests that the simple one works better, so we have no reason to do the more complicated one
        loss = F.mse_loss(epsilon_reverse, epsilon_forward.detach(), reduction='mean')

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def forward(self, x, prompts=None, size=None):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        with torch.no_grad():
            # Step 1: Draw a sample from the prior distribution
            x_t = torch.randn_like(x, dtype=self.dtype)

            if prompts is not None:
                prompt_embeddings, prompt_mask = self.prompt_encoder(prompts)
                prompt_embeddings = prompt_embeddings.to(dtype=self.dtype)
                if len(prompt_embeddings.shape) == 2:
                    prompt_embeddings = prompt_embeddings[:, None, :]
                if prompt_mask is not None:
                    prompt_mask = prompt_mask.to(dtype=self.dtype)
            else:
                prompt_embeddings = torch.zeros((x.shape[0], 1, self.prompt_dim), device=x.device)
                prompt_mask = None

            if size is not None:
                size = size[:, None, :].expand((-1, prompt_embeddings.shape[1], -1))
                prompt_embeddings = torch.cat([prompt_embeddings, size.to(dtype=self.dtype)], dim=-1)

            # Step 2: Run the reverse diffusion process for the whole trajectory
            for t in range(self.trajectory_length - 1, -1, -1):
                x_t = self.sample(x_t, t * torch.ones(x_t.shape[0], device=x_t.device, dtype=torch.long), prompt_embeddings, prompt_mask)
            out = self.autoencoder.decode(x_t.to(dtype=self.autoencoder.dtype)).sample
        return out

    def configure_optimizers(self):
        # Just use Adam and call it a day
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer