import numpy as np


def numpy_monkey_patch():
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "double"):
        np.double = float  # or np.float64 if you prefer specifying precision
