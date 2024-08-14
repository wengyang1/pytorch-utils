import random
from torchvision import transforms


class RandomScale():
    """
    Randomly scale the image between a given min and max scale factor.
    This transform does not keep the aspect ratio.

    Args:
        scale (tuple): Range of scale factors. (min_scale, max_scale)
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        # 获取图像的原始尺寸
        original_width, original_height = img.size

        # 计算随机缩放因子
        scale_factor = random.uniform(self.scale[0], self.scale[1])

        # 计算新的尺寸
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # 应用缩放
        img = transforms.Resize((new_height, new_width))(img)

        return img
