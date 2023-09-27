import os.path

from torchvision.datasets import ImageFolder

from .dataset import register_dataset


@register_dataset
class ddsm(ImageFolder):
    gray_images = True
    task = 'binary'
    num_labels = None

    def __init__(self, root='', mode='train', transform=None, **kwargs):
        super(ddsm, self).__init__(os.path.join(root, mode), transform)


@register_dataset
class vindr:
    def __init__(self):
        pass