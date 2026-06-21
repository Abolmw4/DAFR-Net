from models.encoder.swinir import SwinIR
import os
import requests
import torch
import unittest

class TestNetowk(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        
    # def test_implimetation(self):
    #     device = "cuda" if torch.cuda.is_available else "cpu"
        
    #     swin_net = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
    #                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    #                       mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
        
    
    def test_load_model(self):
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
        
        model.eval()
        model = model.to(device)

        pretrained_model = torch.load("/workspace/model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth")
        model.load_state_dict(pretrained_model["params"] if "params" in pretrained_model.keys() else pretrained_model, strict=False)
        print(model)
        