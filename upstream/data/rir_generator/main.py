import numpy as np
from tqdm import tqdm

from acoustic_simulation import simulate, SourceInfo, RoomParam
from util import load_csv

if __name__ == '__main__':
    degs = list(range(0, 360, 1))
    assert len(degs) == 360
    source = [SourceInfo(deg=deg) for deg in degs]

    rooms = load_csv('room_list.csv')
    for param in tqdm(rooms):
        size = [float(param[dim]) for dim in ["x", "y", "z"]]
        room = RoomParam(
                name=param["id"],
                room_size=np.array(size),
                absorption=float(param["absorption"])
        )
        simulate(room, source)

