import sys
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import load_model, verify_model
from models.encoder.swinir import SwinIR

def get_inference(device: str='cuda', 
                  save_result: str="output_result.png", 
                  input_image: str="/workspace/data/dataset/0_opal-1_2023-07-21_19-12-43_Track_ID_4292_Last__SaltAndPepperNoise_0.1_0.1.jpg",
                  model_path: str="/workspace/checkpoints/checkpoint_epoch_5.pth") -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")

    model = SwinIR(
        upscale=1, 
        in_chans=3, 
        img_size=48, 
        window_size=8, 
        img_range=1., 
        depths=[6, 6, 6, 6, 6, 6], 
        embed_dim=180, 
        num_heads=[6, 6, 6, 6, 6, 6], 
        mlp_ratio=2, 
        upsampler='pixelshuffle', 
        resi_connection='1conv'
    ).to(device)

    if os.path.exists(model_path):
        print(f'Loading weights from {model_path}...')
        model = load_model(model, weight_src=model_path)
    else:
        print(f"Warning: Checkpoint {model_path} not found! Using default/pretrained weights.")

    model.eval()
    
    img_lq = Image.open(input_image).convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(img_lq).unsqueeze(0).to(device) 

    window_size = 8
    _, _, h_old, w_old = input_tensor.size()
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    input_tensor = torch.nn.functional.pad(input_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    print(f"Input shape: {input_tensor.shape}")

    with torch.no_grad():
        output = model(input_tensor)

    output = output[..., :h_old, :w_old]

    output_img = output.squeeze().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img, (1, 2, 0)) 
    output_img = (output_img * 255.0).round().astype(np.uint8)

    result_pil = Image.fromarray(output_img)
    result_pil.save(save_result)
    print(f"Result saved to: {save_result}")

if __name__ == "__main__":
    get_inference()
