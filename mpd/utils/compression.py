# Using gzip (usually faster than bz2)
import pickle
import gzip
import bz2


def save_compressed_pickle(data, filepath):
    with gzip.open(filepath, "wb", compresslevel=9) as f:  # 9 is maximum compression
        pickle.dump(data, f)


def load_compressed_pickle(filepath):
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)


# Using bz2 (usually better compression than gzip but slower)
def save_compressed_pickle_bz2(data, filepath):
    with bz2.BZ2File(filepath, "wb", compresslevel=9) as f:  # 9 is maximum compression
        pickle.dump(data, f)


def load_compressed_pickle_bz2(filepath):
    with bz2.BZ2File(filepath, "rb") as f:
        return pickle.load(f)
