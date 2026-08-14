"""RIFE frame interpolation.

The network definition in `ifnet.py` is vendored from Practical-RIFE
(https://github.com/hzwer/Practical-RIFE), MIT licensed. See NOTICE.

RIFE is not an image-to-image model, so spandrel cannot load it and the
architecture has to travel with the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ifnet import IFNet

RIFE_DIR = Path("models/rife")

# block0 (the coarsest IFBlock) combines its scale_list entry (16) with two
# internal stride-2 convs (4x), for a total downsample of 64x. Padding to a
# multiple of 32 is not enough: rounding in that block's internal
# F.interpolate calls then disagrees with the other blocks' feature maps and
# the concatenation in IFNet.forward fails on a spatial-size mismatch.
# Upstream's own inference_img.py pads to a multiple of 64 for this reason.
ALIGNMENT = 64

# Matches RIFE_HDv3.Model.inference()'s scale_list at scale=1.0: one entry
# per IFBlock (block0..block4), coarsest first.
SCALE_LIST = [16.0, 8.0, 4.0, 2.0, 1.0]


class RifeModel:
    """Callable adapter matching the flow-model contract in `interpolate.py`."""

    def __init__(self, net: IFNet, device: torch.device) -> None:
        self.net = net
        self.device = device

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape[:2]
        ph = (-h) % ALIGNMENT
        pw = (-w) % ALIGNMENT

        def prepare(frame: np.ndarray) -> torch.Tensor:
            x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            x = x.unsqueeze(0).float().div_(255.0).to(self.device)
            return F.pad(x, (0, pw, 0, ph), mode="replicate")

        # The vendored IFNet.forward takes a single channel-concatenated
        # tensor (img0, img1) rather than two separate arguments, and a
        # 5-entry scale_list (one per IFBlock) rather than a bare timestep.
        # It returns (flow_list, mask, merged) — merged[-1] is the final
        # full-resolution synthesized frame; the upstream Model.inference()
        # wrapper does the same.
        imgs = torch.cat((prepare(a), prepare(b)), dim=1)

        # The postprocessing (slicing, clamp, scale, round) has to stay
        # inside inference_mode: tensors produced under it reject in-place
        # ops once the context has exited.
        with torch.inference_mode():
            _, _, merged = self.net(imgs, timestep=float(t), scale_list=SCALE_LIST)
            out = merged[-1]
            out = out[:, :, :h, :w].clamp_(0.0, 1.0).mul_(255.0).round_()
            out = out.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

        return out


def load_rife(path: str | Path, device: str | torch.device = "cuda") -> RifeModel:
    """Load RIFE weights into the vendored IFNet."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RIFE weights not found: {path}")

    device = torch.device(device)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    net = IFNet()
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return RifeModel(net, device)
