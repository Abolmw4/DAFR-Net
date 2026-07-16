import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import load_model, verify_model
from torchvision import transforms
from models.encoder.swinir import SwinIR
from PIL import Image

def get_inference(device: str='cuda', 
                  save_result: str="output_result.png", 
                  input_image: str="/workspace/data/dataset/0_opal-1_2023-07-21_19-12-43_Track_ID_4292_Last__SaltAndPepperNoise_0.1_0.1.jpg",
                  model_path: str="/workspace/checkpoints/checkpoint_epoch_3.pth") -> None:
    
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
        upsampler='', 
        resi_connection='1conv'
    ).to(device)

    if os.path.exists(model_path):
        print(f'Loading weights from {model_path}...')
        model = load_model(model, weight_src=model_path)
    else:
        print(f"Warning: Checkpoint {model_path} not found! Using default/pretrained weights.")

    # --- Input validation ---
    # Check if input exists
    if not input_image:
        raise ValueError("Input image path is required!")
    
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"Input image not found: {input_image}")
    
    # Check if save result exists
    if not save_result:
        raise ValueError("Save result file name is required!")

    # Set up evaluation mode
    model.eval()
    
    # Parse input image
    img_lq = Image.open(input_image).convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(img_lq).unsqueeze(0).to(device) 
    
    # Get original dimensions
    h_old, w_old, _ = input_tensor.size()
    
    # --- Padding ---
    window_size = 8
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    input_tensor = torch.nn.functional.pad(input_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    # --- Inference ---
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