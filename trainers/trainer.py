import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import requests
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from models.encoder.swinir import SwinIR
from utils.utils import load_model, verify_model, find_latest_checkpoint
from losses.swin_loss import VGGPerceptualLoss, PixelLoss, SSIMLoss
from datasets.DAFR_dataset import DAFRDataSet

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================
    # TensorBoard
    # =========================================================

    writer = SummaryWriter(log_dir="runs/swinir_experiment")

    # =========================================================
    # Download model if needed
    # =========================================================

    model_path = "/workspace/model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"

    if os.path.exists(model_path):
        print(f'loading model from {model_path}')

    else:
        os.makedirs("model_zoo", exist_ok=True)

        url = (
            'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
            f'{os.path.basename(model_path)}'
        )

        r = requests.get(url, allow_redirects=True)

        print('downloading pretrained model...')

        with open(model_path, 'wb') as f:
            f.write(r.content)

    # =========================================================
    # Model
    # =========================================================

    # model = SwinIR(upscale=1, in_chans=3, img_size=48, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
    model = SwinIR(upscale=1, in_chans=3, img_size=48, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='', resi_connection='1conv').to(device)

    pretrained_model = load_model(model)
    pretrained_model = pretrained_model.to(device)

    # =========================================================
    # Freeze / Unfreeze
    # =========================================================

    for param in pretrained_model.parameters():
        param.requires_grad = False

    for i in range(3, 6):
        for param in pretrained_model.layers[i].parameters():
            param.requires_grad = True

    # for module in [pretrained_model.norm, pretrained_model.conv_after_body, pretrained_model.conv_before_upsample, pretrained_model.upsample, pretrained_model.conv_last]:
    for module in [pretrained_model.norm, pretrained_model.conv_after_body, pretrained_model.conv_last]:
        for param in module.parameters():
            param.requires_grad = True

    verify_model(pretrained_model)

    # =========================================================
    # Dataset
    # =========================================================

    # dataset = SyntheticSRDataset(num_samples=500, hr_size=224, scale=1)
    dataset = DAFRDataSet(gt_dir="data/GT", aug_dir="data/dataset")
    # train / validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,[train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # =========================================================
    # Optimizer / Scheduler / Loss
    # =========================================================

    optimizer = AdamW(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    criterion1 = VGGPerceptualLoss().to(device)
    criterion2 = PixelLoss().to(device)
    criterion3 = SSIMLoss().to(device)
    epochs = 50

    global_step = 0
    
    
    # =========================================================
    # Resume training (optional)
    # =========================================================
    ckpt_dir = "/workspace/checkpoints"
    resume = True  # set False if you have not to resume
    resume_path = None  # if you have load specefic model set path here
    start_epoch = 0

    if resume:
        if resume_path is None:
            resume_path = find_latest_checkpoint(ckpt_dir)

        if resume_path is not None and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            pretrained_model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

            start_epoch = ckpt["epoch"] 
            global_step = ckpt.get("global_step", 0)

            print(f"Loaded epoch={start_epoch}, global_step={global_step}")
        else:
            print("No checkpoint found. Starting from scratch.")
            
    # =========================================================
    # Training loop
    # =========================================================

    # for epoch in tqdm(range(epochs)):
    for epoch in tqdm(range(start_epoch, epochs)):


        # =====================================================
        # TRAIN
        # =====================================================

        pretrained_model.train()

        train_loss = 0

        for step, (lr, hr) in enumerate(train_loader):

            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()

            pred = pretrained_model(lr)

            if pred.shape != hr.shape:
                pred = F.interpolate(pred, size=hr.shape[-2:], mode="bicubic", align_corners=False)

            loss_percept = criterion1(pred, hr)
            loss_pixel = criterion2(pred, hr)
            ssim_loss = criterion3(pred, hr)
            
            loss = 0.1 * loss_pixel + 0.3 * loss_percept + 0.8 * ssim_loss
            
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            # =================================================
            # TensorBoard Scalars
            # =================================================

            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)

            # =================================================
            # Gradient Histograms
            # =================================================

            for name, param in pretrained_model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, global_step)

            global_step += 1

            if step % 10 == 0:

                print(f"Epoch [{epoch+1}/{epochs}]", f"Step [{step}/{len(train_loader)}]", f"Loss: {loss.item():.5f}")

        avg_train_loss = train_loss / len(train_loader)

        # =====================================================
        # VALIDATION
        # =====================================================

        pretrained_model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)

                pred = pretrained_model(lr)

                if pred.shape != hr.shape:
                    pred = F.interpolate(pred, size=hr.shape[-2:], mode="bicubic", align_corners=False)

                loss_percept = criterion1(pred, hr)
                loss_pixel = criterion2(pred, hr)
                ssim_loss = criterion3(pred, hr)
                
                loss = 0.1 * loss_pixel + 0.3 * loss_percept + 0.8 * ssim_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # =====================================================
        # Scheduler
        # =====================================================

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # =====================================================
        # TensorBoard Epoch Scalars
        # =====================================================

        writer.add_scalars("Loss",{"train": avg_train_loss, "validation": avg_val_loss}, epoch)


        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # =====================================================
        # Image logging
        # =====================================================

        writer.add_images("Images/LR", lr[:4], epoch)

        writer.add_images("Images/Prediction", pred[:4].clamp(0, 1), epoch)

        writer.add_images("Images/HR", hr[:4], epoch)

        # =====================================================
        # Weight histograms
        # =====================================================

        for name, param in pretrained_model.named_parameters():

            writer.add_histogram(f"Weights/{name}", param, epoch)

        # =====================================================
        # Console output
        # =====================================================

        print(f"\nEpoch {epoch+1}/{epochs}", f"\nTrain Loss: {avg_train_loss:.5f}", f"\nVal Loss: {avg_val_loss:.5f}", f"\nLR: {current_lr:.8f}")

        # =====================================================
        # Save checkpoint
        # =====================================================

    
        ckpt_dir = "/workspace/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = f"{ckpt_dir}/checkpoint_epoch_{epoch+1}.pth"

        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model": pretrained_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()}, ckpt_path)

        print(f"Checkpoint saved to {ckpt_path}")
    
    
    writer.close()


if __name__ == "__main__":
    main()
