import torch
import math
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

from sentence_transformers import SentenceTransformer

from diffusers import SanaTransformer2DModel
from diffusers.utils import apply_lora_scale

from torchdiffeq import odeint
from torchmetrics.image.fid import FrechetInceptionDistance

class REPATransformer2DModel(SanaTransformer2DModel):
    def __init__(self, repa_dim, repa_layer, num_attention_heads, attention_head_dim, *args, **kwargs):
        super().__init__(*args, **kwargs, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)

        self.repa_dim = repa_dim
        self.repa_layer = repa_layer

        inner_dim = num_attention_heads * attention_head_dim

        self.repa_attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=8, batch_first=True)
        self.repa_projection = nn.Sequential(
            nn.Linear(inner_dim, self.repa_dim * 4),
            nn.SiLU(),
            nn.Linear(self.repa_dim * 4, self.repa_dim)
        )

    # This is just copied from the original forward, but we need to output the REPA hidden states
    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

                if index_block == self.repa_layer:
                    repa_state = hidden_states

        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

                if index_block == self.repa_layer:
                    repa_state = hidden_states

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        # Do the REPA projection
        repa_state = self.repa_attn(repa_state, repa_state, repa_state, need_weights=False)[0]
        repa_state = torch.mean(repa_state, dim=1)
        repa_state = self.repa_projection(repa_state)

        return (output, repa_state)


class ViTWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, x):
        inputs = self.processor(x, return_tensors="pt").to(dtype=self.model.dtype)
        return self.model(**inputs).last_hidden_state


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
            tokens = self.tokeniser(prompts, padding='max_length', max_length=128, truncation=True, return_tensors="pt", return_attention_mask=True)

            for k in tokens:
                if hasattr(tokens[k], "to"):
                    tokens[k] = tokens[k].to(device=self.encoder.device)

            encoding = self.encoder(**tokens)
            return encoding.last_hidden_state, tokens.attention_mask
        else:
            assert isinstance(self.encoder, SentenceTransformer), "If no tokeniser is provided, the encoder must be a SentenceTransformer"
            encoding = self.encoder.encode(prompts, convert_to_tensor=True)

            return encoding, None


class REPAModel(L.LightningModule):
    def __init__(
            self, latent_dim, latent_channels, autoencoder,
            lr, prompt_encoder, prompt_dim=768,
            repa_dim=256, repa_layer=7, repa_weight=0.5,
            prompt_dropout=0.1
            ):
        super(REPAModel, self).__init__()
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.autoencoder = autoencoder
        self.lr = lr
        self.prompt_encoder = prompt_encoder
        self.prompt_dim = prompt_dim
        self.repa_weight = repa_weight
        self.prompt_dropout = prompt_dropout

        # Note: if we are using flash attention we need to ensure the model is on GPU before we run it to get the embedding for an empty prompt for classifier free guidance
        # A tad irritating but oh well
        with torch.no_grad():
            dev = self.prompt_encoder.encoder.device
            self.prompt_encoder.cuda()
            empty_prompt, empty_mask = self.prompt_encoder([""])
            empty_prompt = empty_prompt.to(dev) # And restore it back to its original device

        self.register_buffer("empty_prompt", empty_prompt.to(dev))
        self.register_buffer("empty_mask", empty_mask.to(dev))

        # Load Sana 600M config
        config = SanaTransformer2DModel.load_config(
            "Efficient-Large-Model/Sana_600M_1024px_diffusers", 
            subfolder="transformer"
        )

        # Reduce depth
        #config["num_layers"] = 20

        # Reduce width
        config["num_attention_heads"] = 12
        config["attention_head_dim"] = 64
        config["cross_attention_dim"] = 768
        config["num_cross_attention_heads"] = 12
        config["cross_attention_head_dim"] = 64

        config["caption_channels"] = prompt_dim

        config["dropout"] = 0.1 # A little bit of dropout is probably wise

        config["sample_size"] = latent_dim[0]
        config["in_channels"] = latent_channels
        config["out_channels"] = latent_channels

        #config["repa_dim"] = repa_dim
        #config["repa_layer"] = repa_layer

        # We're going to have to do this manually because we're not using from config
        config.pop('_class_name')
        config.pop('_diffusers_version')

        #self.flow_net = REPATransformer2DModel.from_config(config)
        self.flow_net = REPATransformer2DModel(repa_dim=repa_dim, repa_layer=repa_layer, **config)

        self.fid_metric = FrechetInceptionDistance(
            feature=2048, input_img_size=(3, self.latent_dim[0] * 32, self.latent_dim[1] * 32),
            normalize=True, compute_on_cpu=False, sync_on_compute=True, dist_sync_on_step=True).to(self.device)

    def flow(self, latent_t, t, prompt_embeddings, prompt_mask, return_repa=False):
        out, repa_state = self.flow_net(
            hidden_states=latent_t,
            encoder_hidden_states=prompt_embeddings,
            encoder_attention_mask=prompt_mask,
            timestep=t.to(dtype=latent_t.dtype) * 1000)

        if return_repa:
            return out, repa_state
        else:
            return out


    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            x_1 = batch['dcae_embedding'].to(dtype=self.dtype) # Precomputed DCAE embeddings are already scaled by the scaling factor

            # Unlike diffusion models, our timestep is continuous in [0, 1]
            t = torch.rand(size=(x_1.shape[0],), device=x_1.device, dtype=x_1.dtype)

            # Note: Unlike diffusion models, x_1 is the data and x_0 is our prior distribution (which is N(0, I))
            x_0 = torch.randn_like(x_1)

            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * x_1

            target_velocity = x_1 - x_0

            target_velocity = target_velocity.to(dtype=self.dtype)
            x_t = x_t.to(dtype=self.dtype)
            t = t.to(dtype=self.dtype)

            prompt_embeddings = batch['prompt_embedding'].to(dtype=self.dtype)
            prompt_mask = batch['prompt_mask']

            empty_prompt = self.empty_prompt.expand((prompt_embeddings.shape[0], -1, -1))
            empty_mask = self.empty_mask.expand((prompt_embeddings.shape[0], -1))

            prompt_dropout = torch.rand(size=(prompt_embeddings.shape[0], 1, 1), device=prompt_embeddings.device) < self.prompt_dropout

            prompt_embeddings = torch.where(prompt_dropout, empty_prompt, prompt_embeddings)
            prompt_mask = torch.where(prompt_dropout.squeeze(2), empty_mask, prompt_mask)

            size = batch['size'][:, None, :].expand((-1, prompt_embeddings.shape[1], -1))
            prompt_embeddings = torch.cat([prompt_embeddings, size.to(dtype=self.dtype)], dim=-1).detach()

        velocity, repa_state = self.flow(x_t, t, prompt_embeddings, prompt_mask, return_repa=True)

        velocity_loss = F.mse_loss(velocity, target_velocity.detach(), reduction='mean')
        self.log("velocity_loss", velocity_loss, prog_bar=True)

        if self.repa_weight > 0:
            with torch.no_grad():
                repa_target = batch['repa_embedding'].to(dtype=self.dtype)
    
            repa_loss = 1 - F.cosine_similarity(repa_state, repa_target.detach(), dim=-1)
            repa_loss = self.repa_weight * repa_loss.mean()

            self.log("repa_loss", repa_loss, prog_bar=True)

            loss = velocity_loss + repa_loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # Override the train function to ensure the other models we're using stay in eval mode
    def train(self, mode=True):
        super().train(mode)
        if self.prompt_encoder is not None:
            self.prompt_encoder.eval()
        self.autoencoder.eval()

    def forward(self, x, prompts=None, size=None, steps=1, cfg_scale=1.0):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        with torch.no_grad():
            x_0 = torch.randn_like(x)

            prompt_embeddings, prompt_mask = self.prompt_encoder(prompts)
            prompt_embeddings = prompt_embeddings.to(dtype=self.dtype)
            prompt_mask = prompt_mask.to(dtype=self.dtype)

            empty_prompt = self.empty_prompt.expand((prompt_embeddings.shape[0], -1, -1))
            empty_mask = self.empty_mask.expand((prompt_embeddings.shape[0], -1))

            size = size[:, None, :].expand((-1, prompt_embeddings.shape[1], -1)).to(dtype=self.dtype)
            prompt_embeddings = torch.cat([prompt_embeddings, size], dim=-1)
            empty_prompt = torch.cat([empty_prompt, size], dim=-1)

            # Note: The actual REPA paper actually uses a more sophisticated SDE solver
            # To do this, they had to make their own SDE solver, and convert the flow model from an ODE to SDE
            # I honestly, do not entirely see the point of this. Maybe it gives better sample quality, but we want speed and simplicity
            t = torch.linspace(0, 1, steps=steps + 1, device=x.device)

            def ode(t, x):
                v = self.flow(x, t.expand((x.shape[0],)), empty_prompt, empty_mask, return_repa=False)
                v_cond = self.flow(x, t.expand((x.shape[0],)), prompt_embeddings, prompt_mask, return_repa=False)
                return v + cfg_scale * (v_cond - v)

            trajectory = odeint(ode, x_0, t, atol=1e-5, rtol=1e-3, method='dopri5')

            # Again, note that Huggingface's diffusers library does not apply the scaling factor for us
            # We have to divide by it ourselves to ensure the magnitudes are correct for decoding with the autoencoder
            latent = trajectory[-1] / self.autoencoder.config.scaling_factor

            out = self.autoencoder.decode(latent.to(dtype=self.autoencoder.dtype)).sample
        return out

    def on_validation_epoch_start(self):
        with torch.no_grad():
            self.fid_metric.reset()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            idxs, images, captions, size = batch

            x = torch.randn((images.shape[0], self.latent_channels, *self.latent_dim), device=self.device, dtype=self.dtype)

            generated = self.forward(x, prompts=captions, size=size, cfg_scale=1.0)

            generated = (generated + 1.0) / 2.0 # Rescale from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0

            self.fid_metric.update(generated, real=False)
            self.fid_metric.update(images, real=True)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            fid = self.fid_metric.compute()
            self.log("COCO Validation FID", fid, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Just use Adam and call it a day
        optimizer = torch.optim.Adam(self.flow_net.parameters(), lr=self.lr)
        return optimizer

    # Note, we often want to pause and continue training with different learning rates
    # But lightning's checkpointing keeps overwriting our new learning rates with the old ones
    # Intercept the on_load_checkpoint callback to deal with this
    def on_load_checkpoint(self, checkpoint):
        if 'optimizer_states' in checkpoint:
            for optimizer_state in checkpoint['optimizer_states']:
                for param_group in optimizer_state['param_groups']:
                    assert 'lr' in param_group, "Learning rate not found in checkpoint optimizer state"
                    param_group['lr'] = self.lr