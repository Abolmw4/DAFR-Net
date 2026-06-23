import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.encoder.swinir import SwinIR
import torch
import requests
from utils.utils import load_model, verify_model
from torch.optim import AdamW
import cv2
from torchvision.transforms import transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists("model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"):
        print(f'loading model from model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    
    else:
        os.makedirs(os.path.dirname("/workspace/model_zoo"), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename("001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model model_zoo')
        open('/workspace/model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth', 'wb').write(r.content)


    model = SwinIR(upscale=1, in_chans=3, img_size=48, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                          mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
    model.train()
    model = model.to(device)
    pretrained_model = load_model(model)
        
    # Freeze everything
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # Unfreeze layers 3,4,5
    for i in range(3, 6):
        for param in pretrained_model.layers[i].parameters():
            param.requires_grad = True

    # Unfreeze remaining reconstruction modules
    for module in [pretrained_model.norm, pretrained_model.conv_after_body, pretrained_model.conv_before_upsample, pretrained_model.upsample, pretrained_model.conv_last,]:
        for param in module.parameters():
            param.requires_grad = True
        
    verify_model(pretrained_model)

    optimizer = AdamW(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=1e-4)
    
    image = cv2.imread("/workspace/datasets/opal-1_2023-07-21_19-14-25_Track_ID_4471_First_.jpg")

    tr = transforms.Compose([
        transforms.ToTensor(),
        ])
    torch_image = tr(image)
    final = torch_image.reshape(1, 3, 255, 186)
    output = pretrained_model(final.to(device))
    print(output.shape)
    
if __name__ == "__main__":
    main()