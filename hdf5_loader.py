import h5py

def load_hdf5(path):
    """Загрузка HDF5 с MRI и масками"""
    with h5py.File(path, "r") as f:
        images = f["images"][:]   # (N, 2, 256, 256)
        masks  = f["masks"][:]    # (N, 1, 256, 256)
    return images, masks