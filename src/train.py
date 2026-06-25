import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from src.models import GeneratorRRDB, PatchGANDiscriminator

def parse_args():
    parser = argparse.ArgumentParser(description="Project IRIS - Multi-Objective GAN Training")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed .npy files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu/mps)")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 epochs on dummy/loaded data to check gradient flow")
    return parser.parse_args()

def normalize_for_deeplab(batch):
    """
    Normalize batch to ImageNet stats for DeepLabV3+ classification.
    Expects input batch shape [B, 3, H, W] in [0, 1] range.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std

def map_esa_to_voc(mask):
    """
    Map ESA WorldCover classes (10, 20, 30, etc.) to 21-class Pascal VOC indices for DeepLabV3+.
    """
    mapped = torch.zeros_like(mask, dtype=torch.long)
    # Trees (10), Shrubland (20), Grassland (30), Cropland (40) -> 16 (pottedplant/vegetation)
    mapped[mask == 10] = 16
    mapped[mask == 20] = 16
    mapped[mask == 30] = 16
    mapped[mask == 40] = 16
    # Built-up (50) -> 11 (diningtable / urban structure)
    mapped[mask == 50] = 11
    # Open water (80) -> 4 (boat/water-associated)
    mapped[mask == 80] = 4
    return mapped

def load_dataset(data_dir):
    """
    Load preprocessed NumPy arrays and wrap them into a PyTorch DataLoader.
    """
    input_path = os.path.join(data_dir, "delhi_ncr_input.npy")
    target_thermal_path = os.path.join(data_dir, "delhi_ncr_target_thermal.npy")
    target_rgb_path = os.path.join(data_dir, "delhi_ncr_target_rgb.npy")
    cover_mask_path = os.path.join(data_dir, "delhi_ncr_cover_mask.npy")
    
    if not (os.path.exists(input_path) and os.path.exists(target_rgb_path)):
        # Fallback to dummy tensors if preprocessed files are missing
        print("Processed files not found. Creating mock dataset tensors for training...")
        inputs = np.random.rand(4, 6, 77, 77).astype(np.float32)
        targets_rgb = np.random.rand(4, 3, 154, 154).astype(np.float32)
        targets_thermal = np.random.rand(4, 154, 154).astype(np.float32)
        cover_masks = np.random.choice([10, 20, 50, 80], size=(4, 154, 154)).astype(np.uint8)
    else:
        # Load from disk and add batch dimension (1 tile -> repeat to form a small training batch)
        inputs = np.load(input_path)[np.newaxis, ...] # [1, 6, 77, 77]
        targets_rgb = np.load(target_rgb_path)[np.newaxis, ...] # [1, 3, 154, 154]
        targets_thermal = np.load(target_thermal_path)[np.newaxis, ...] # [1, 154, 154]
        cover_masks = np.load(cover_mask_path)[np.newaxis, ...] # [1, 154, 154]
        
        # Duplicate to create a batch size of 2 or more for training loop stability
        inputs = np.repeat(inputs, 4, axis=0)
        targets_rgb = np.repeat(targets_rgb, 4, axis=0)
        targets_thermal = np.repeat(targets_thermal, 4, axis=0)
        cover_masks = np.repeat(cover_masks, 4, axis=0)

    # Convert to PyTorch tensors
    inputs_t = torch.tensor(inputs)
    targets_rgb_t = torch.tensor(targets_rgb)
    targets_thermal_t = torch.tensor(targets_thermal)
    cover_masks_t = torch.tensor(cover_masks)

    dataset = TensorDataset(inputs_t, targets_rgb_t, targets_thermal_t, cover_masks_t)
    return dataset

def main():
    args = parse_args()
    print(f"Training parameters: Device={args.device}, Epochs={args.epochs}, BatchSize={args.batch_size}, DryRun={args.dry_run}")

    # Set device (supporting CPU, MPS on Mac, and CUDA)
    device = torch.device(args.device)
    if "mps" in args.device or args.device == "cuda":
        # Check support
        if args.device == "mps" and not torch.backends.mps.is_available():
            print("MPS is not available, falling back to CPU.")
            device = torch.device("cpu")
        elif args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU.")
            device = torch.device("cpu")

    # 1. Initialize Generator and Discriminator
    net_G = GeneratorRRDB(in_channels=6, out_channels=3).to(device)
    net_D = PatchGANDiscriminator(in_channels=3).to(device)

    # 2. Load frozen Semantic Segmenter (DeepLabV3+ with ResNet50 backbone)
    print("Loading frozen DeepLabV3+ ResNet50 model for semantic consistency loss...")
    deeplab_weights = DeepLabV3_ResNet50_Weights.DEFAULT
    net_Sem = deeplabv3_resnet50(weights=deeplab_weights).to(device)
    for param in net_Sem.parameters():
        param.requires_grad = False
    net_Sem.eval()

    # 3. Setup optimizers and losses
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.9, 0.999))

    criterion_L1 = nn.L1Loss()
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_SEM = nn.CrossEntropyLoss()

    # 4. Load dataset
    dataset = load_dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    epochs_to_run = 2 if args.dry_run else args.epochs
    print(f"Starting training loop for {epochs_to_run} epochs...")

    for epoch in range(1, epochs_to_run + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for batch_idx, (inputs, targets_rgb, targets_thermal, cover_masks) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets_rgb = targets_rgb.to(device)
            cover_masks = cover_masks.to(device)
            
            # --- 4.1 Update Discriminator ---
            optimizer_D.zero_grad()
            
            # Real RGB forward pass
            pred_real = net_D(targets_rgb)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake RGB forward pass
            fake_rgb = net_G(inputs)
            pred_fake = net_D(fake_rgb.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()
            epoch_d_loss += loss_D.item()

            # --- 4.2 Update Generator ---
            optimizer_G.zero_grad()
            
            # Adversarial component
            pred_fake_G = net_D(fake_rgb)
            loss_GAN = criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
            
            # L1 pixel reconstruction component
            loss_L1 = criterion_L1(fake_rgb, targets_rgb)
            
            # Semantic consistency component (L_SEM)
            # Pass generated RGB through segmenter
            fake_rgb_norm = normalize_for_deeplab(fake_rgb)
            sem_logits = net_Sem(fake_rgb_norm)["out"] # Shape: [B, 21, H, W]
            
            # Map cover masks to VOC class space
            mapped_masks = map_esa_to_voc(cover_masks) # Shape: [B, H, W]
            loss_SEM = criterion_SEM(sem_logits, mapped_masks)
            
            # Multi-objective total generator loss
            loss_G = 10.0 * loss_L1 + 1.0 * loss_GAN + 5.0 * loss_SEM
            loss_G.backward()
            optimizer_G.step()
            epoch_g_loss += loss_G.item()
            
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs_to_run}] - Avg Gen Loss: {avg_g_loss:.4f} | Avg Disc Loss: {avg_d_loss:.4f}")

    print("Training finished successfully!")
    
    # Save the generator weights
    os.makedirs("models", exist_ok=True)
    save_path = "models/generator.pt"
    torch.save(net_G.state_dict(), save_path)
    print(f"Saved trained generator state dict to: {save_path}")

if __name__ == "__main__":
    main()
