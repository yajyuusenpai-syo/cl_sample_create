import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_ch=2, n_fft=1024, hop_length=512):
        super(FeatureExtractor, self).__init__()
        self.input_ch = input_ch
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        """
        x: (batch, channels=2, time)
        returns: (batch, channels=2, time_frames, freq_bins)
        """
        batch_size, channels, time_len = x.shape
        assert channels == self.input_ch, "Input must have 2 channels!"

        # (batch, channels, time) -> (batch * channels, time)
        x = x.view(batch_size * channels, time_len)

        stft_output = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True
        )  # (batch, channels, freq_bins, time_frames)
        stft_output = stft_output.view(batch_size, channels, *stft_output.shape[-2:])  # (batch, channels, freq_bins, time_frames)

        # Separate magnitude and phase
        magnitude = torch.abs(stft_output)  # (batch, channels, freq_bins, time_frames)
        phase = torch.angle(stft_output)    # (batch, channels, freq_bins, time_frames)

        # Permute to (batch, time_frames, freq_bins, channels)
        magnitude = magnitude.permute(0, 3, 2, 1)  # (batch, time_frames, freq_bins, channels)
        phase = phase.permute(0, 3, 2, 1)

        # Concatenate magnitude and phase along channel axis
        features = torch.cat([magnitude, phase], dim=-1)  # (batch, time_frames, freq_bins, 2*channels)

        return features
