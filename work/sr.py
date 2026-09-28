"""Real-ESRGAN (SRVGGNetCompact, realesr-general-x4v3) on CPU, with denoise blending (dni) like the official repo."""
import os, torch, numpy as np, cv2
import torch.nn as nn, torch.nn.functional as F

S = os.environ.get('S', '/tmp/claude-0/-home-user-Video/f8f0d442-cac9-55ab-9a30-19dc4913e7e5/scratchpad')
torch.set_num_threads(int(os.environ.get('SR_THREADS', '4')))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


_model = {}


def get_model(denoise=0.5):
    key = round(denoise, 2)
    if key in _model:
        return _model[key]
    a = torch.load(f'{S}/models/realesr-general-x4v3.pth', map_location='cpu')['params']
    b = torch.load(f'{S}/models/realesr-general-wdn-x4v3.pth', map_location='cpu')['params']
    # dni: weight on the "no-denoise" model = denoise strength inverse, as in inference_realesrgan.py
    w = [denoise, 1 - denoise]
    st = {k: w[0] * a[k] + w[1] * b[k] for k in a}
    m = SRVGGNetCompact()
    m.load_state_dict(st, strict=True)
    m.eval()
    _model[key] = m
    return m


@torch.inference_mode()
def upscale4(bgr, denoise=0.5):
    """bgr uint8 or float01 -> float01 bgr, 4x."""
    m = get_model(denoise)
    x = bgr.astype(np.float32)
    if bgr.dtype == np.uint8:
        x = x / 255.0
    t = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1)))[None]
    y = m(t)[0].clamp(0, 1).numpy().transpose(1, 2, 0)[:, :, ::-1]
    return np.ascontiguousarray(y)
