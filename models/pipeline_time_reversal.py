import torch
import torchvision.transforms as T
from diffusers import UNet3DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import torch.nn as nn

class TimeReversalPipeline:
    def __init__(self, unet, vae, scheduler, image_encoder, device=None):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.image_encoder = image_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        encoder_hidden_dim = getattr(self.unet.config, "encoder_hid_dim", 1024)
        clip_output_dim = self.image_encoder.config.projection_dim
        self.encoder_hidden_proj = nn.Linear(clip_output_dim, encoder_hidden_dim).to(self.device, dtype=torch.float32)

    def _encode_image(self, image):
        if isinstance(image, Image.Image):
            image = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(self.device, dtype=torch.float32)
        outputs = self.image_encoder(image)
        return outputs.image_embeds

    def __call__(self, image_1, image_2, s_churn=0.0, M=8, t0=5, decode_chunk_size=8, generator=None):
        device = self.device
        dtype = torch.float32

        # 1. Transform images
        resize_transform = T.Compose([T.Resize((320, 576)), T.ToTensor()])
        image_tensor_1 = resize_transform(image_1).unsqueeze(0).to(device, dtype=dtype)
        image_tensor_2 = resize_transform(image_2).unsqueeze(0).to(device, dtype=dtype)
        image_tensor_1 = 2.0 * image_tensor_1 - 1.0
        image_tensor_2 = 2.0 * image_tensor_2 - 1.0

        # 2. VAE encode
        latents_1 = self.vae.encode(image_tensor_1).latent_dist.sample().unsqueeze(2)
        latents_2 = self.vae.encode(image_tensor_2).latent_dist.sample().unsqueeze(2)

        # 3. Interpolate latents
        latent_sequence = [(1 - i / (M - 1)) * latents_1 + (i / (M - 1)) * latents_2 for i in range(M)]
        latents = torch.cat(latent_sequence, dim=2).to(dtype)

        # 4. Interpolate embeddings
        embedding_1 = self._encode_image(image_1)
        embedding_2 = self._encode_image(image_2)
        interpolated_embeddings = [(1 - i / (M - 1)) * embedding_1 + (i / (M - 1)) * embedding_2 for i in range(M)]
        encoder_hidden_states = torch.cat(interpolated_embeddings, dim=0)  # (M, D)
        encoder_hidden_states = self.encoder_hidden_proj(encoder_hidden_states).to(dtype)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)  # (1, M, D)

        # 5. Noise prediction
        t = torch.tensor([t0], dtype=torch.long, device=device)
        noise_pred = self.unet(
            latents, t.expand(latents.shape[0]), encoder_hidden_states=encoder_hidden_states
        ).sample

        # 6. Denoising step
        latents = self.scheduler.step(noise_pred, t0, latents).prev_sample

        # 7. Decode
        decoded = []
        for i in range(0, M, decode_chunk_size):
            chunk = latents[:, :, i:i+decode_chunk_size, :, :]  # (1, 4, chunk, H, W)
            vae_input = chunk.squeeze(0).permute(1, 0, 2, 3)  # (chunk, 4, H, W)
            recon = self.vae.decode(vae_input.to(dtype=torch.float32)).sample.to(dtype)
            recon = (recon / 2 + 0.5).clamp(0, 1)
            decoded.extend(recon)

        pil_images = [T.ToPILImage()(frame.cpu()) for frame in decoded]
        return type("Result", (), {"frames": [pil_images]})()
