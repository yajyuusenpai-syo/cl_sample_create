import pyroomacoustics as pa
import numpy as np
from typing import List, Tuple
import torchaudio
import os
from dataclasses import dataclass

DEG_TO_RAD = np.pi / 180

RIR_DIR = "output/rir"
RT60_DIR = "output/rt60"
for d in [RIR_DIR, RT60_DIR]:
    os.makedirs(d, exist_ok=True)

@dataclass
class RoomParam:
    name: str
    fs: int = 16_000
    mic_distance: float = 0.15
    seed: int = 765
    room_size: np.ndarray = np.array([9., 7.5, 3.5]) 
    snr: float = float("inf")
    absorption: float = 0.5

@dataclass
class SourceInfo:
    deg: int
    r: float = 2.0

def pad_and_transpose(rir_list):
    mic_num = len(rir_list)
    src_num = len(rir_list[0])
    max_t = max(len(h) for mic_rirs in rir_list for h in mic_rirs)
    ret = np.zeros((src_num, mic_num, max_t), dtype=rir_list[0][0].dtype)
    for s in range(src_num):
        for m in range(mic_num):
            h = rir_list[m][s]
            t = len(h)
            ret[s,m,:t] = h
    return ret

def simulate(room_param: RoomParam, sources: List[SourceInfo]):
    # 乱数初期化
    np.random.seed(room_param.seed)

    # マイク位置
    mic_center = room_param.room_size/2 
    mic_relative_pos = np.array([[x*room_param.mic_distance, 0., 0.] for x in [-0.5, 0.5]])
    mic_pos = mic_relative_pos.T + mic_center[:, None]

    # 音源位置
    r = np.array([source.r for source in sources])
    theta = DEG_TO_RAD * np.array([source.deg for source in sources])
    source_relative_pos = r * np.array([np.cos(theta), np.sin(theta), np.zeros(len(theta))])
    source_pos = source_relative_pos + mic_center[:, None]

    # 部屋の作成
    room = pa.ShoeBox(room_param.room_size, fs=room_param.fs, max_order=17, absorption=room_param.absorption)
    room.add_microphone_array(pa.MicrophoneArray(mic_pos, fs=room.fs))
    for pos in source_pos.T:
        room.add_source(pos)
    
    """
    # RIRを保存する
    room.compute_rir()
    rir = pad_and_transpose(room.rir)
    np.save(os.path.join(RIR_DIR, room_param.name+".npy"), rir)
    """

    # RT60を保存する
    rt60 = room.rt60_theory()
    np.save(os.path.join(RT60_DIR, room_param.name+".npy"), rt60)

if __name__ == '__main__':
    degs = list(range(0, 360, 1))
    assert len(degs) == 360

    room = RoomParam(name="test")
    source = [SourceInfo(deg=deg) for deg in degs]

    simulate(room, source)

