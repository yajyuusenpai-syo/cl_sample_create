import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=2, cnn_channels=64, middle_features=128, output_features=256, n_fft=1024):
        """
        input_channels: 入力チャンネル数（ここでは4）
        cnn_channels (P): CNNの中間フィルタ数
        output_features (Q): 最終的な特徴量次元
        """
        super(Encoder, self).__init__()
        assert (output_features % 2) == 0
        self.output_features = output_features

        self.cnn_block = nn.Sequential(
            nn.Conv2d(input_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # CNN出力のfreq次元の縮小を正しく計算
        self.freq_after_cnn = (n_fft // 2) // (2 ** 3)  # (n_fft/2)を3回半分にする
        self.middle_linear = nn.Linear(cnn_channels * self.freq_after_cnn, middle_features)

        self.gru = nn.GRU(
            input_size=middle_features,
            hidden_size=output_features // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        """
        x: (batch, time_frames, freq_bins, channels=2)
        returns: (batch, time_frames, output_features=Q)
        """
        batch_size, time_frames, freq_bins, channels = x.shape

        # Prepare for CNN: permute to (batch, 2 * channels, time, freq)
        x = x.permute(0, 3, 1, 2)  # (batch, 2 * channels, time, freq)

        # CNN
        x = self.cnn_block(x)  # (batch, cnn_channels, time, reduced_freq)

        batch_size, cnn_channels, time_steps, freq_bins_reduced = x.size()

        # Prepare for linear layer
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time_steps, channels, freq_bins_reduced)
        x = x.view(batch_size, time_steps, -1)  # (batch, time_steps, channels * freq_bins_reduced)

        # Project to output_features dimension (Q)
        x = self.middle_linear(x)  # (batch, time_steps, middle_features)

        x, _ = self.gru(x)          # (batch, time_steps, output_features)
        x = torch.mean(x, dim=1)

        return x

