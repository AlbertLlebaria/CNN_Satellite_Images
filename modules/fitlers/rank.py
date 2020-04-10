from scipy.ndimage.filters import rank_filter
from skimage.exposure import equalize_adapthist


def apply_f1(image, rank, size):
    C = equalize_adapthist(image=image, kernel_size=size)
    r = rank_filter(input=C, rank=rank, size=size)
    return r
