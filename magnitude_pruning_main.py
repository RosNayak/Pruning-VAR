import sys
sys.path.append("VAR")

import os
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from models import build_vae_var
import torch.nn.utils.prune as prune
import glob
import numpy as np
import matplotlib.pyplot as plt

# def measure_model_size(model):
#     torch.save(model.state_dict(), "temp_model.pth")
#     size = os.path.getsize("temp_model.pth") / (1024 ** 2)
#     os.remove("temp_model.pth")
#     return size

# def effective_model_size(model):
#     total = 0
#     nonzero = 0
#     for p in model.parameters():
#         total += p.numel()
#         nonzero += p.nonzero().size(0)
#     density = nonzero / total
#     full_size_mb = measure_model_size(model)
#     return density * full_size_mb  # Approx effective size

# def measure_inference_time(model, input_tensor, runs=30):
#     model.eval()
#     with torch.no_grad():
#         for _ in range(10):  # warmup
#             _ = model(input_tensor)
#         start = time.time()
#         for _ in range(runs):
#             _ = model(input_tensor)
#         end = time.time()
#     return (end - start) / runs

def plot_weight_distributions(model,
                              layer_types=(torch.nn.Linear, torch.nn.Conv2d),
                              bins=100,
                              save_dir="weight_histograms"):
    """
    Plots (and saves) weight‐distribution histograms for each layer in `model`
    whose type is in `layer_types`.

    Args:
        model: a torch.nn.Module
        layer_types: tuple of layer classes to include (e.g. Linear, Conv2d)
        bins: number of histogram bins
        save_dir: directory where PNGs will be written
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and hasattr(module, 'weight'):
            weights = module.weight.detach().cpu().numpy().flatten()

            plt.figure(figsize=(6, 4))
            plt.hist(weights, bins=bins, alpha=0.7)
            plt.title(f"Weight Distribution: {name}")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()

            # sanitize name for filesystem
            filename = name.replace(".", "_").replace("/", "_")
            path = os.path.join(save_dir, f"{filename}.png")

            plt.savefig(path)
            plt.close()  # close the figure to free memory

def apply_magnitude_pruning(model, pruning_amount=0.2):
    """
    Perform naive magnitude pruning on the given model using the specified pruning ratio.

    Args:
    - model (torch.nn.Module): The VAR model to prune
    - pruning_ratio (float): The percentage of weights to prune (e.g., 0.2 means 20%)
    """
    print("Applying Magnitude Pruning...")

    for name, module in model.named_modules():
        # Prune only specific layer types: Linear or Conv2d
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)

            # Optionally prune the bias (if present)
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=pruning_amount)

    return model

def save_tensor_as_image(tensor, save_path):
    """Convert tensor to PIL Image and save"""
    # Ensure tensor is in range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to PIL Image
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    Image.fromarray(img).save(save_path)

def generate_class_samples(
    vae,
    var,
    num_samples_per_class=50,
    num_classes=1000,
    save_dir='generated_samples',
    device='cuda',
    batch_size=8,  # Process multiple images per batch for efficiency
    schedule=None
):
    """Generate samples for all ImageNet classes"""
    os.makedirs(save_dir, exist_ok=True)

    # Set models to eval mode
    vae.eval()
    var.eval()

    with torch.no_grad():
        # Loop through all classes
        for class_idx in tqdm(range(num_classes), desc="Generating class samples"):
            samples_remaining = num_samples_per_class

            while samples_remaining > 0:
                # Calculate batch size for this iteration
                current_batch_size = min(batch_size, samples_remaining)

                # Create class labels tensor
                labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)

                # Generate samples using autoregressive inference
                generated_images = var.autoregressive_infer_cfg_with_pruning(
                    B=current_batch_size,
                    label_B=labels,
                    sparsity_schedule=schedule,
                    pruning_method="magnitude",
                    g_seed=42,
                    top_k=900,  # Can be adjusted
                    top_p=0.96,  # Can be adjusted
                    cfg=1.5,  # Classifier-free guidance scale - adjust as needed
                    more_smooth=False
                )
                # generated_images = var.autoregressive_infer_cfg(
                #     B=current_batch_size,
                #     label_B=labels,
                #     top_k=900,  # Can be adjusted
                #     top_p=0.96,  # Can be adjusted
                #     cfg=1.5,  # Classifier-free guidance scale - adjust as needed
                #     more_smooth=False
                # )

                # Decode latents to images using VQVAE
                # generated_images = vae.decode(generated_images)

                # Save each image in the batch
                for i in range(current_batch_size):
                    image_path = os.path.join(
                        save_dir,
                        f'class_{class_idx:04d}sample{num_samples_per_class - samples_remaining + i:03d}.png'
                    )
                    save_tensor_as_image(generated_images[i], image_path)

                samples_remaining -= current_batch_size

def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path

def make_hook(layer_idx):
    def hook(self, x, attn_bias):
        # x: [B, L, C], self.mat_qkv: nn.Linear(C, 3*C)
        B, L, C = x.shape
        # compute Q,K exactly as SelfAttention does
        qkv = self.mat_qkv(x)             # [B, L, 3*C]
        q, k, _ = qkv.chunk(3, dim=-1)    # each [B, L, C]
        d_head = C // self.num_heads

        # reshape into (B, H, L, d_head)
        q = q.view(B, L, self.num_heads, d_head).transpose(1,2)
        k = k.view(B, L, self.num_heads, d_head).transpose(1,2)

        # scaled dot–product
        scores = (q @ k.transpose(-2,-1)) * self.scale   # [B, H, L, L]
        if attn_bias is not None:
            scores = scores + attn_bias

        weights = scores.softmax(dim=-1)  # [B, H, L, L]

        # accumulate per-head mean map over the batch
        for h in range(self.num_heads):
            head_map = weights[:, h].mean(dim=0).detach().cpu()  # [L, L]
            key = (layer_idx, h)
            if attn_sums[key] is None:
                attn_sums[key] = head_map
            else:
                attn_sums[key] += head_map

        # return the *real* output
        return torch.nn.functional.linear(  # mimic SelfAttention.forward
            torch.nn.functional.linear(x, self.mat_qkv.weight, bias=None)
            .view(B, L, self.num_heads, d_head)
            .transpose(1,2)
            .reshape(B, L, C),
            self.proj.weight, self.proj.bias
        )
    return hook

if __name__=="__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    vae, var = build_vae_var(
        device=device,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        V=4096,              # Vocabulary size
        Cvae=32,            # VAE latent channels
        ch=160,             # Base channels
        share_quant_resi=4  # Shared quantization residual
    )

    # Load checkpoints
    vae.load_state_dict(torch.load('vae_ch160v4096z32.pth', map_location='cpu'), strict=True)
    var_ckpt = torch.load('var_d16.pth', map_location='cpu')
    if 'model' in var_ckpt:
        var_ckpt = var_ckpt['model']
    var.load_state_dict(var_ckpt, strict=True)

    # plot_weight_distributions(var)

    #Move models to device
    vae = vae.to(device)
    var = var.to(device)

    #add the hooks
    hooks = []
    for layer_idx, block in enumerate(var.blocks):
        sa = block.attn
        hooks.append(sa.register_forward_hook(make_hook(layer_idx)))

    # var.prune_magnitude(prune_n=2, prune_m=4)

    # var_pruned = apply_magnitude_pruning(var, pruning_amount=0.2)

    # generate_class_samples(
    #     vae=vae,
    #     var=var_pruned,
    #     num_samples_per_class=10,
    #     num_classes=1000,
    #     save_dir='GenImagesPruned',
    #     device=device,
    #     batch_size=1  # Adjust based on your GPU memory
    # )

    # pruned_npz_path = create_npz_from_sample_folder('GenImagesPruned')

    #Generate samples
    generate_class_samples(
        vae=vae,
        var=var,
        num_samples_per_class=10,
        num_classes=1000,
        save_dir='NM_Pruning',
        device=device,
        batch_size=16,  # Adjust based on your GPU memory
        # schedule=[0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
    )

    unpruned_npz_path = create_npz_from_sample_folder('NM_Pruning')