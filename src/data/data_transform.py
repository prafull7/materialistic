import sys
sys.path.append('../../')
import torch
import torch.nn as nn
import numpy as np

def white_balance(im, red_gain, blue_gain):
    out = torch.clone(im)
    assert out.size(1) == 3, "expected a color image."
    out[:, 0:1] = im[:, 0:1] * red_gain
    out[:, 2:3] = im[:, 2:3] * blue_gain
    return out

class RandomWhiteBalance():
    def __init__(self, min_gain=0.8, max_gain=1.2):
        super().__init__()
        assert max_gain >= min_gain, "exposure range is invalid"
        self.min_gain = min_gain
        self.max_gain = max_gain

    def _random_gain(self, bs):
        gain = torch.rand(bs, 1, 1, 1)
        gain *= self.max_gain - self.min_gain
        gain += self.min_gain
        return gain * 0 + 1.

    def __call__(self, im):
        red_gain = self._random_gain(im.size(0)).type_as(im) 
        blue_gain = self._random_gain(im.size(0)).type_as(im)
        # print("gain:", red_gain.mean(), blue_gain.mean())
        return white_balance(im, red_gain, blue_gain)

def exposure_correction(im, exposure):
    return im * exposure

class RandomExposure():
    def __init__(self, min_stops=-1, max_stops=1):
        super().__init__()
        self.min_stops = min_stops
        self.max_stops = max_stops
        assert max_stops >= min_stops, "exposure range is invalid"

    def _random_gain(self, bs):
        gain = torch.clamp(torch.randn(bs, 1, 1, 1), -1, 1)
        gain *= self.max_stops - self.min_stops
        gain += self.min_stops
        gain = torch.pow(2.0, gain*0)
        return gain

    def __call__(self, im):
        exposure = self._random_gain(im.size(0)).type_as(im)
        im = im * exposure
        return im

def gamma_correction(im, gamma, eps=1e-8):
    gamma = 1/gamma
    sign = (im >= 0).float()
    sign = 1.0 * sign - 1.0 * (1 - sign)
    im = im.abs()
    im = torch.pow(im + eps, gamma) - torch.pow(
                    torch.Tensor([eps]).type_as(im), gamma
                )
    return im * sign

class RandomGammaCorrection():
    def __init__(self, min_gamma=1.8, max_gamma=2.6):
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.eps = 1e-4

    def __call__(self, im):
        gamma = torch.rand(im.size(0), 1, 1, 1)
        gamma *= self.max_gamma - self.min_gamma
        gamma += self.min_gamma
        gamma = gamma.type_as(im)
        im = gamma_correction(im, 2.2, eps=self.eps)
        return im

class ISPDataAugmentation(nn.Module):
    def __init__(self, no_jpeg_compress=False):
        super().__init__()
        self.WhiteBalance = RandomWhiteBalance()
        self.ExposureAdjustment = RandomExposure()
        self.GammaCorrection = RandomGammaCorrection()
    
    def forward(self, img):
        white_balanced_im = self.WhiteBalance(img.unsqueeze(0))
        exposure_adjusted_im = self.ExposureAdjustment(white_balanced_im)
        gamma_corrected_im = self.GammaCorrection(exposure_adjusted_im)
        clipped = np.clip(gamma_corrected_im, 0, 1)
        return clipped[0]

