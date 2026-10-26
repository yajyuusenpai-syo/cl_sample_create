import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy

from seldnet import SELDModel
from audio_dataset import AudioCapsDataset
from loss import SELDLoss
from collate import CollateFN
from rir_dataset import RIRDataset
from util import set_seed

def train(
    model, 
    dataloader, 
    optimizer, 
    loss_fn, 
    collate_fn,
    epoch=0,
    log_f=None,
):
    model.train()

    total_loss = 0
    total_sed_loss = 0
    total_doa_loss = 0
    loss_num = 0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        (waveforms, metas) = collate_fn(batch)

        optimizer.zero_grad()

        # Forward
        sed_output, doa_output = model(waveforms)

        # Compute loss
        sed_loss, doa_loss, loss = loss_fn(sed_output, doa_output, metas)

        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sed_loss += sed_loss.item()
        total_doa_loss += doa_loss.item()
        loss_num += 1

        if batch_idx % 10 == 0:
            log_line = f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, SED: {sed_loss.item():.4f}, DOA: {doa_loss.item():.4f}"
            print(log_line)
            if log_f is not None:
                log_f.write(log_line + "\n")

    avg_loss = total_loss / loss_num
    avg_sed_loss = total_sed_loss / loss_num
    avg_doa_loss = total_doa_loss / loss_num

    return avg_loss, avg_sed_loss, avg_doa_loss

def validate(model, dataloader, loss_fn, collate_fn):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_sed_loss = 0
        total_doa_loss = 0
        loss_num = 0

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            (waveforms, metas) = collate_fn(batch)


            # Forward
            sed_output, doa_output = model(waveforms)

            # Compute loss
            sed_loss, doa_loss, loss = loss_fn(sed_output, doa_output, metas)

            total_loss += loss.item()
            total_sed_loss += sed_loss.item()
            total_doa_loss += doa_loss.item()
            loss_num += 1
            
        avg_loss = total_loss / loss_num
        avg_sed_loss = total_sed_loss / loss_num
        avg_doa_loss = total_doa_loss / loss_num


    return avg_loss, avg_sed_loss, avg_doa_loss

def main():
    batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()

    # Config
    out_dir = "output"
    config = {
        "wav_dir": "../data/wav/",
        "csv_path": "../data/fixed_audiocaps2.0/train.csv",
        "val_csv_path": "../data/fixed_audiocaps2.0/val.csv",
        "batch_size": batch_size,
        "num_classes": 344,
        "epochs": 50,
        "path": {
            "log_dir": os.path.join(out_dir, "log"),
            "ckpt_dir": os.path.join(out_dir, "ckpt"),
        },
        "doa_weight": 50,
    }
    for d in config["path"].values():
        os.makedirs(d, exist_ok=True)
    val_config = copy.deepcopy(config)
    val_config["csv_path"] = val_config["val_csv_path"]

    # Train Dataset & Dataloader
    train_rir_dataset = RIRDataset("train")
    train_audio_dataset = AudioCapsDataset(config)
    train_collate_fn = CollateFN(train_rir_dataset, device)
    train_dataloader = DataLoader(
        train_audio_dataset, batch_size=config["batch_size"], 
        shuffle=True, num_workers=16, drop_last=True, collate_fn=lambda x:x)

    # Val Dataset & Dataloader
    val_rir_dataset = RIRDataset("val")
    val_audio_dataset = AudioCapsDataset(val_config)
    val_collate_fn = CollateFN(val_rir_dataset, device)
    val_dataloader = DataLoader(
        val_audio_dataset, batch_size=val_config["batch_size"], 
        shuffle=True, num_workers=16, drop_last=True, collate_fn=lambda x:x)
    
    # Model
    model = SELDModel(num_classes=config["num_classes"]).to(device)

    # Loss & Optimizer
    loss_fn = SELDLoss(num_classes=config["num_classes"], doa_weight=config["doa_weight"], device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Open log file
    train_log_f = open(os.path.join(config["path"]["log_dir"], "train_log.txt"), "w")
    val_log_f = open(os.path.join(config["path"]["log_dir"], "val_log.txt"), "w")

    # Training loop
    for epoch in range(config["epochs"]):
        avg_loss, avg_sed_loss, avg_doa_loss = train(
            model, train_dataloader, optimizer, loss_fn, train_collate_fn, epoch, train_log_f
        )
        val_avg_loss, val_avg_sed_loss, val_avg_doa_loss = validate(
            model, val_dataloader, loss_fn, val_collate_fn
        )
        log_line = f"===> Epoch {epoch} Avg Loss: {avg_loss:.4f}, SED: {avg_sed_loss:.4f}, DOA: {avg_doa_loss:.4f}\n" \
                +  f"  Val Loss: {val_avg_loss:.4f}, SED: {val_avg_sed_loss:.4f}, DOA: {val_avg_doa_loss:.4f}"
        print(log_line)
        train_log_f.write(log_line + "\n")

        log_line = f"Epoch {epoch} Val Loss: {val_avg_loss:.4f}, SED: {val_avg_sed_loss:.4f}, DOA: {val_avg_doa_loss:.4f}"
        val_log_f.write(log_line + "\n")

        # Save model
        save_path = os.path.join(config["path"]["ckpt_dir"], f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

if __name__ == "__main__":
    main()
