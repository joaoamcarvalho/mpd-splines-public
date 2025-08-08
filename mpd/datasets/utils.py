import io
import pickle

import torch


class CPU_Unpickler(pickle.Unpickler):
    # https://stackoverflow.com/a/68992197
    # Unpickler that can load a GPU pickled object on the CPU
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
