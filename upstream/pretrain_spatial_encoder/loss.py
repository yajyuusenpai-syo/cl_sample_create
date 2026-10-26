import torch
import torch.nn as nn

class SELDLoss:
    def __init__(self, num_classes, doa_weight, device="cuda"):
        self.num_classes = num_classes
        self.device = device

        self.bce_loss_fn = nn.BCELoss()
        self.mse_loss_fn = nn.MSELoss()
        self.doa_weight = doa_weight

    def __call__(self, sed_output, doa_output, metas):
        """
        sed_output: (batch, N)
        doa_output: (batch, N)
        metas: list of dict
        returns:
            sed_loss: Tensor
            doa_loss: Tensor
        """
        batch_size, _ = sed_output.shape

        sed_target = torch.zeros((batch_size, self.num_classes), device=self.device)
        doa_target = torch.zeros((batch_size, self.num_classes), device=self.device)

        for b in range(batch_size):
            meta_list = metas[b]  # 例: [meta1, meta2]
            for meta in meta_list:
                events = meta["event"]  # [2, 40]みたいなリスト
                doa = meta["doa"]        # scalar値
                
                for event_id in events:
                    event_id = event_id
                    sed_target[b, event_id] = 1.0
                    doa_target[b, event_id] = doa
        
        sed_loss = self.bce_loss_fn(sed_output, sed_target)
        doa_loss = self.mse_loss_fn(doa_output, doa_target)

        loss = sed_loss + self.doa_weight * doa_loss

        return sed_loss, doa_loss, loss
