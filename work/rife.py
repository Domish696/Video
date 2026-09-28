"""Minimal inference wrapper for Practical-RIFE v4.26 (hzwer, MIT). Arbitrary-timestep frame interpolation on CPU."""
import os, sys, types
import numpy as np, torch
import torch.nn.functional as F

S = os.environ.get('S', '/tmp/claude-0/-home-user-Video/f8f0d442-cac9-55ab-9a30-19dc4913e7e5/scratchpad')
torch.set_num_threads(int(os.environ.get('RIFE_THREADS', '4')))
_grid = {}


def warp(inp, flow):
    k = (str(flow.device), str(flow.size()))
    if k not in _grid:
        B, _, H, W = flow.shape
        hor = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        ver = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        _grid[k] = torch.cat([hor, ver], 1)
    fl = torch.cat([flow[:, 0:1] / ((inp.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((inp.shape[2] - 1.0) / 2.0)], 1)
    g = (_grid[k] + fl).permute(0, 2, 3, 1)
    return F.grid_sample(inp, g, mode='bilinear', padding_mode='border', align_corners=True)


# the upstream network file imports `from model.warplayer import warp`; provide it without the training package
_m = types.ModuleType('model'); _w = types.ModuleType('model.warplayer'); _w.warp = warp
sys.modules.setdefault('model', _m); sys.modules.setdefault('model.warplayer', _w)
sys.path.insert(0, S + '/models/rife')
from IFNet_HDv3 import IFNet  # noqa: E402

_net = None


def net():
    global _net
    if _net is None:
        n = IFNet()
        sd = torch.load(S + '/models/rife/flownet.pkl', map_location='cpu', weights_only=True)
        sd = {k.replace('module.', ''): v for k, v in sd.items() if 'module.' in k or True}
        n.load_state_dict(sd, strict=False)
        n.eval()
        _net = n
    return _net


def _to_t(img):
    x = img.astype(np.float32)
    if img.dtype == np.uint8:
        x = x / 255.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


@torch.inference_mode()
def flows(a, b, t, scale=1.0):
    """Return (flow_t->a [H,W,2], flow_t->b [H,W,2], mask [H,W]) in pixel units of the inputs: the frame at time t is
    mask * a(p + f_a) + (1 - mask) * b(p + f_b). Lets us transport any higher-resolution version of a/b."""
    h, w = a.shape[:2]
    m = max(128, int(128 / scale))
    ph, pw = ((h - 1) // m + 1) * m, ((w - 1) // m + 1) * m
    A = F.pad(_to_t(a), (0, pw - w, 0, ph - h), mode='replicate'); B_ = F.pad(_to_t(b), (0, pw - w, 0, ph - h), mode='replicate')
    sl = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    fl, mask, _ = net()(torch.cat((A, B_), 1), float(t), sl)
    f = fl[-1][0, :, :h, :w].numpy()
    msk = torch.sigmoid(mask)[0, 0, :h, :w].numpy()
    return (np.ascontiguousarray(f[0:2].transpose(1, 2, 0)), np.ascontiguousarray(f[2:4].transpose(1, 2, 0)), np.ascontiguousarray(msk))


@torch.inference_mode()
def interpolate(a, b, t, scale=1.0):
    """a, b: HxWx3 (uint8 or float01, any channel order); t in (0,1). Returns float01 HxWx3."""
    h, w = a.shape[:2]
    m = max(128, int(128 / scale))
    ph, pw = ((h - 1) // m + 1) * m, ((w - 1) // m + 1) * m
    A = F.pad(_to_t(a), (0, pw - w, 0, ph - h), mode='replicate'); B_ = F.pad(_to_t(b), (0, pw - w, 0, ph - h), mode='replicate')
    sl = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    _, _, merged = net()(torch.cat((A, B_), 1), float(t), sl)
    out = merged[-1][0, :, :h, :w].clamp(0, 1).numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(out)
