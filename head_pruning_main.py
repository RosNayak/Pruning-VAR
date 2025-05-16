import math
import pickle
import os, glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from models import build_vae_var
import torch.nn.utils.prune as prune

import sys
sys.path.append("VAR/models")
sys.path.append("VAR")

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
    do_save=True
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
                generated_images = var.autoregressive_infer_cfg(
                    B=current_batch_size,
                    label_B=labels,
                    top_k=900,  # Can be adjusted
                    top_p=0.96,  # Can be adjusted
                    cfg=1.5,  # Classifier-free guidance scale - adjust as needed
                    more_smooth=False
                )

                # Decode latents to images using VQVAE
                # generated_images = vae.decode(latents)

                # Save each image in the batch
                if do_save:
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

def compute_avg(importance):
    sums   = defaultdict(float)
    counts = defaultdict(int)

    for (layer, head, L), score in importance.items():
      key = (layer, head)
      sums[key]   += score
      counts[key] += 1

    # 2) compute averages
    avg_importance = {
      (layer, head): sums[(layer, head)] / counts[(layer, head)]
      for (layer, head) in sums
    }

    return avg_importance

def main():
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
    vae.load_state_dict(torch.load('/content/drive/MyDrive/EfficientAI/vae_ch160v4096z32.pth', map_location='cpu'), strict=True)
    var_ckpt = torch.load('/content/drive/MyDrive/EfficientAI/var_d16.pth', map_location='cpu')
    if 'model' in var_ckpt:
        var_ckpt = var_ckpt['model']
    var.load_state_dict(var_ckpt, strict=True)

    # var = apply_magnitude_pruning(var, pruning_amount=0.4)

    # Move models to device
    vae = vae.to(device)
    var = var.to(device)

    hooks = []
    attn_sums = {}
    for layer_idx, block in enumerate(var.blocks):
        sa = block.attn
        hooks.append(sa.register_forward_hook(var.make_hook(layer_idx, attn_sums)))

    num_samples = 1
    batch_size = 8
    num_classes = 10
    n_batches = math.ceil(num_samples*num_classes / batch_size)

    important_scores_path = "/content/drive/MyDrive/EfficientAI/important_scores.pkl"
    if not os.path.exists(important_scores_path):
        # Generate samples
        generate_class_samples(
            vae=vae,
            var=var,
            num_samples_per_class=num_samples,
            num_classes=num_classes,
            save_dir='/content/drive/MyDrive/EfficientAI/GenImagesPruned',
            device=device,
            batch_size=batch_size,  # Adjust based on your GPU memory
            do_save=False
        )

        for h in hooks:
            h.remove()

        attn_avg = {}
        for key, total_map in attn_sums.items():
            attn_avg[key] = total_map / n_batches   # still a [L,L] tensor

        import torch.nn.functional as F

        def attention_entropy(attn_map):
            eps = 1e-8
            entropies = - (attn_map * (attn_map + eps).log()).sum(dim=-1)  # per query
            return entropies.mean()

        importance_scores = {}
        for key, attn in attn_avg.items():
            importance_scores[key] = attention_entropy(attn).item()

        importance_scores = compute_avg(importance_scores)
    else:
        with open(important_scores_path, 'rb') as f:
            importance_scores = pickle.load(f)

    sorted_heads = sorted(importance_scores.items(), key=lambda x: x[1])  # ascending
    prune_ratio = 0.1  # prune 30%
    num_heads_total = len(importance_scores)
    num_prune = int(prune_ratio * num_heads_total)
    heads_to_prune = sorted_heads[:num_prune]

    head_mask = {}  # mask 1 = keep, 0 = prune
    for (layer_idx, head_idx), _ in importance_scores.items():
        if layer_idx not in head_mask:
            head_mask[layer_idx] = [1] * var.blocks[layer_idx].attn.num_heads
    for (layer_idx, head_idx), _ in heads_to_prune:
        head_mask[layer_idx][head_idx] = 0

    for layer_idx, block in enumerate(var.blocks):
        mask_tensor = torch.tensor(head_mask[layer_idx], dtype=torch.float32)
        block.attn.set_head_mask(mask_tensor, device)

    num_samples = 1
    batch_size = 8

    generate_class_samples(
        vae=vae,
        var=var,
        num_samples_per_class=num_samples,
        num_classes=num_classes,
        save_dir='/content/drive/MyDrive/EfficientAI/zero_prune',
        device=device,
        batch_size=batch_size,  # Adjust based on your GPU memory
        do_save=True
    )


    # save_attention_heatmaps(attn_avg, "/content/drive/MyDrive/EfficientAI")

    pruned_npz_path = create_npz_from_sample_folder('path/to/generated_images_npz')
    unpruned_npz_path = create_npz_from_sample_folder('path/to/image_net_val_images')

    print(compute_fid_metrics('unpruned_npz_path', 'pruned_npz_path'))

    # # Example usage
    # compute_fid('/content/drive/MyDrive/EfficientAI/GenImagesPruned.npz', '/content/drive/MyDrive/EfficientAI/VIRTUAL_imagenet256_labeled.npz')
    # compute_fid('/content/drive/MyDrive/EfficientAI/GenImages.npz', '/content/drive/MyDrive/EfficientAI/VIRTUAL_imagenet256_labeled.npz')

if __name__ == '__main__':
    main()