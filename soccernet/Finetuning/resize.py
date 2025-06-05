from torchvision import transforms
from torchvision.transforms import functional as F
class ResizeWithPadding:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Redimensionne en gardant le ratio
        img = F.resize(img, self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        # Puis pad pour arriver à 640x640
        img = F.pad(img, self._get_padding(img))
        return img

    def _get_padding(self, img):
        w, h = img.size
        pad_w = self.size[0] - w
        pad_h = self.size[1] - h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return (left, top, right, bottom)