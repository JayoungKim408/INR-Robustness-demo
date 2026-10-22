
import torchvision.transforms as transforms
try:
    from torchvision.transforms import v2
except ImportError:
    import torchvision.transforms as v2
from torchvision.transforms.functional import InterpolationMode


TO_TENSOR_RESCALE = transforms.Compose([transforms.ToTensor(), lambda x: (x * 2 - 1)])

def transform_train(image_size, first_resize, flip, rescale):
    assert rescale, 'rescale must be True'
    
    H, W = image_size
    ls = []
    if first_resize:
        ls += [ transforms.Resize((H, W), interpolation=InterpolationMode.BILINEAR) ]
    ls += [ v2.RandomCrop(size=(H, W), padding=4, padding_mode='reflect') ]
    if flip:
        ls += [ v2.RandomHorizontalFlip(p=0.5) ]
    ls += [ transforms.ToTensor(), lambda x: (x * 2 - 1) ]
    return transforms.Compose(ls)
