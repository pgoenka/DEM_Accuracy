import os
import numpy as np
from pathlib import Path

# We'll use a try-except block to handle missing torch for now
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class UNetBlock(nn.Module if 'TORCH_AVAILABLE' in locals() and TORCH_AVAILABLE else object):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if not TORCH_AVAILABLE: return
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class LightweightUNet(nn.Module if 'TORCH_AVAILABLE' in locals() and TORCH_AVAILABLE else object):
    """A lightweight U-Net for spatial elevation error prediction."""
    def __init__(self, in_channels=13, out_channels=1):
        super().__init__()
        if not TORCH_AVAILABLE: return
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(32, 64)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(64, 32) # Skip connection adds 32
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(32, 16) # Skip connection adds 16
        
        # Final prediction map
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

class DeepRefiner:
    """Orchestrates training and inference for the Deep Learning refinement stage."""

    def train_model(self, context, cache, epochs=10, batch_size=16):
        print("\n=== DEEP REFINEMENT ENGINE (Phase 3) ===")
        
        if not TORCH_AVAILABLE:
            print("[!] PyTorch not found. Skipping Deep Learning training.")
            print("[!] Please run: venv\\Scripts\\python.exe -m pip install torch")
            return

        if not torch.cuda.is_available():
            print("[!] CUDA not available. U-Net training on CPU is prohibitively slow.")
            print("[!] Skipping Phase 3 Deep Refinement. Random Forest output will be used.")
            print("[!] To enable: install PyTorch with CUDA support.")
            return

        x_path = cache.path(context.aoi, "cnn_patches_X.npy")
        y_path = cache.path(context.aoi, "cnn_patches_y.npy")
        model_path = cache.path(context.aoi, "unet_refiner.pth")

        if not x_path.exists() or not y_path.exists():
            print("[!] Training patches not found. Run PatchGeneration stage first.")
            return

        # 1. Load Patches
        print("Loading training patches...")
        X = np.load(x_path) # (N, 256, 256, 13)
        y = np.load(y_path) # (N,) - Error at center pixel
        
        # 2. Prepare PyTorch Tensors
        # Convert (N, H, W, C) to (N, C, H, W) for PyTorch
        X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).float()
        y_tensor = torch.from_numpy(y).float()
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 3. Initialize Model
        device = torch.device("cuda")
        model = LightweightUNet(in_channels=13, out_channels=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 4. Training Loop
        print(f"Training U-Net on {len(X)} patches for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Model outputs a 256x256 map
                outputs = model(inputs)
                
                # We supervise the center pixel (128, 128) for point ground truth
                # Output shape: (Batch, 1, 256, 256)
                center_pred = outputs[:, 0, 128, 128]
                
                loss = criterion(center_pred, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")

        # 5. Save Model
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path.name}")
        context.outputs["unet_model"] = model_path
