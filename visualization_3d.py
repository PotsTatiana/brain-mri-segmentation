import numpy as np
from vedo import Volume, show


def visualize_tumor_3d(masks_3d, spacing=(3.0, 1.0, 1.0), threshold=0.5):
    """
    Строит 3D визуализацию опухоли по стеку бинарных масок.

    Параметры:
        masks_3d: numpy array (num_slices, height, width) – бинарные маски (0 или 1)
        spacing: (z, y, x) размер вокселя в мм
        threshold: порог для изоповерхности (0.5 для бинарных масок)
    """
    masks_3d = (masks_3d > 0).astype(np.uint8)
    if masks_3d.sum() == 0:
        print("Нет данных для 3D визуализации")
        return None

    vol = Volume(masks_3d.astype(np.float32), spacing=spacing)
    vol.cmap('Reds').alpha([0, 0.8])

    # Убираем '__doc__' и другие устаревшие аргументы
    show(vol, axes=1, bg='blackboard', interactive=True)
    return vol