import torch
import torch.nn.functional as F
import copy

def conv_rir(dry_sources, rirs, ch_num, device):
    batch_size = len(dry_sources)
    assert len(rirs) == batch_size

    # それぞれ最大長を取得
    max_dry_len = max(w.size(0) for w in dry_sources)
    max_rir_len = max(rir.shape[-1] for rir in rirs)

    # --- dry_sourcesパディング＆ch展開 ---
    padded_dry_sources = []
    for w in dry_sources:
        pad = max_dry_len - w.size(0)
        w = torch.nn.functional.pad(w, (0, pad))  # (time,)
        w = w.unsqueeze(0).repeat(ch_num, 1)  # (ch_num, time)
        padded_dry_sources.append(w)
    dry_sources = torch.cat(padded_dry_sources, dim=0).unsqueeze(0)  # (1, batch*ch_num, time)

    # --- rirパディング＆展開 ---
    padded_rirs = []
    for rir in rirs:
        pad = max_rir_len - rir.size(-1)
        rir = torch.nn.functional.pad(rir, (0, pad))  # (ch_num, rir_time)
        rir = rir.flip(-1)
        rir = rir.unsqueeze(1)  # (ch_num, 1, rir_time)
        padded_rirs.append(rir)
    rirs = torch.cat(padded_rirs, dim=0)  # (batch*ch_num, 1, rir_time)

    # --- torch tensor ---
    dry_sources = dry_sources.to(device)
    rirs = rirs.to(device)
    assert dry_sources.shape == (1, batch_size * ch_num, max_dry_len)
    assert rirs.shape == (batch_size * ch_num, 1, max_rir_len)

    # --- convolution ---
    mix = F.conv1d(dry_sources, rirs, padding=max_rir_len-1, groups=rirs.size(0))

    # --- 整形 ---
    mix = mix.squeeze(1)  # (batch*ch_num, time)
    mix = mix.view(batch_size, ch_num, -1)  # (batch, ch_num, time)

    return mix

class CollateFN:
    def __init__(self, rir_dataset, device, ch_num=2, enable_scl=True):
        self.rir_dataset = rir_dataset
        self.device = device
        self.ch_num = ch_num
        self.enable_scl = enable_scl
        
    def make_batch(self, mixed, metas):
        assert (len(mixed) % 4) == 0
        n = len(mixed) // 2

        out_wavs = []
        out_metas = []

        # 単一音源としてそのまま追加
        for i in range(n // 2):
            out_wavs.append(mixed[i])         # (2, time)
            out_metas.append([metas[i]])      # メタデータはリストでラップ

        # 2. 後半: 2つの音源をミックスして追加
        lis = list(range(n // 2, n, 2)) + \
                list(range(n + n//2, 2*n, 2))
        assert len(lis) == (n//2)
        for i in lis:
            m1 = mixed[i]
            m2 = mixed[i + 1]
            mix = m1 + m2                      # ステレオのまま加算
            mix = mix / 2                      # 正規化（音量の過大を防ぐ）
            out_wavs.append(mix)
            out_metas.append([metas[i], metas[i + 1]])

        # --- tensor化 ---
        out_wavs = torch.stack(out_wavs, dim=0)

        assert len(out_wavs) == len(out_metas)
        if self.enable_scl:
            assert out_wavs.shape == (n, 2, mixed.shape[2])
        else:
            assert out_wavs.shape == ((3*n)//4, 2, mixed.shape[2])
        
        return (out_wavs, out_metas)

    def __call__(self, batch):
        n = len(batch)

        # --- input collection ---
        dry_sources = []
        rirs = []
        metas = []

        rir_candidates = [self.rir_dataset.get() for _ in range(n)]

        # --- main ---
        for i, (waveform, meta_) in enumerate(batch):
            rir, doa = rir_candidates[i]
            assert len(rir) == self.ch_num

            meta = copy.deepcopy(meta_)
            meta["doa"] = doa

            dry_sources.append(waveform)
            rirs.append(rir)
            metas.append(meta)

        # --- sub ---
        if self.enable_scl:
            for i, (waveform, meta_) in enumerate(batch):
                if i%2 == 0:
                    j = i + 1
                else:
                    j = i - 1
                rir, doa = rir_candidates[j]

                meta = copy.deepcopy(meta_)
                meta["doa"] = doa

                dry_sources.append(waveform)
                rirs.append(rir)
                metas.append(meta)

        # --- convolution ---
        mixed = conv_rir(dry_sources, rirs, self.ch_num, self.device)
        
        # --- output prep ---
        return self.make_batch(mixed, metas)

if __name__ == '__main__':
    import os
    import torchaudio
    
    from audio_dataset import AudioCapsDataset
    from collate import CollateFN
    from rir_dataset import RIRDataset
    from util import set_seed

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    config = {
        "wav_dir": "data/wav/",
        "csv_path": "data/fixed_audiocaps2.0/train.csv",
    }

    # Train Dataset & Dataloader
    rir_dataset = RIRDataset("train")
    audio_dataset = AudioCapsDataset(config)
    collate_fn = CollateFN(rir_dataset, device)

    batch = [audio_dataset[i] for i in range(batch_size)]
    wavs, metas = collate_fn(batch)

    out_dir = "output/debug/collate"
    os.makedirs(out_dir, exist_ok=True)

    for i, (wav, meta) in enumerate(batch):
        torchaudio.save(os.path.join(out_dir, f"input_{i}.wav"), wav.unsqueeze(0), audio_dataset.sample_rate)
    for i, wav in enumerate(wavs):
        torchaudio.save(os.path.join(out_dir, f"output_{i}.wav"), wav.cpu(), audio_dataset.sample_rate)
