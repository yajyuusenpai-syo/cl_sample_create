import numpy as np
import random
import torch
import os

class RIRDataset:
    def _load(self, fname):
        return torch.from_numpy(np.load(fname)).float()

    def __init__(self, subset):
        assert subset in ["train", "val", "test"]

        start_room_id, end_room_id = {
            "train": (0, 336),
            "val": (336, 378),
            "test": (378, 420),
        }[subset]

        self.data = [
            self._load(os.path.join(
                os.path.dirname(__file__),
                "data/rir_generator/",
                f"output/rir/{str(n).zfill(4)}.npy"
            ))
            for n in range(start_room_id, end_room_id)
        ]
        self.room_num = end_room_id - start_room_id

    def get(self):
        """ランダムにRIRを1つ返す"""
        rid = random.randint(0, self.room_num-1)
        doaid = random.randint(0, 359)
        return self.get_impl(rid, doaid)

    def get_impl(self, rid, doaid):
        doa = doaid/180
        if doa > 1:
            doa = 2 - doa
        doa = 2 * doa - 1 
        
        return (self.data[rid][doaid], doa)

