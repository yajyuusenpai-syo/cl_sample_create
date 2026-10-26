import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy

from model import CLAPEncoder
from audio_dataset import AudioCapsDataset
from loss import CLAPLoss
from collate import CollateFN
from rir_dataset import RIRDataset
from util import set_seed
from generate_caption import meta_to_caption

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
    loss_num = 0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        (waveforms, metas) = collate_fn(batch)
        captions = [meta_to_caption(meta) for meta in metas]

        optimizer.zero_grad()

        # Forward
        clap_output = model(audio=waveforms, text=captions)

        # Compute loss
        loss = loss_fn(clap_output, metas, model.logit_scale)

        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_num += 1

        if batch_idx % 10 == 0:
            log_line = f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
            print(log_line)
            if log_f is not None:
                log_f.write(log_line + "\n")

    avg_loss = total_loss / loss_num

    return avg_loss

def validate(model, dataloader, loss_fn, collate_fn):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        loss_num = 0

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            (waveforms, metas) = collate_fn(batch)
            captions = [meta_to_caption(meta) for meta in metas]

            # Forward
            clap_output = model(audio=waveforms, text=captions)

            # Compute loss
            loss = loss_fn(clap_output, metas, model.logit_scale)

            total_loss += loss.item()
            loss_num += 1
            
        avg_loss = total_loss / loss_num

    return avg_loss

def main():
    batch_size = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()

    # Config
    out_dir = "output"
    config = {
        "wav_dir": "data/wav/",
        "csv_path": "data/fixed_audiocaps2.0/train.csv",
        "val_csv_path": "data/fixed_audiocaps2.0/val.csv",
        "batch_size": batch_size,
        "epochs": 50,
        "path": {
            "log_dir": os.path.join(out_dir, "log"),
            "ckpt_dir": os.path.join(out_dir, "ckpt"),
        },
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
    model = CLAPEncoder().to(device)
    model.load_default_state_dict()

    # Loss & Optimizer
    loss_fn = CLAPLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Open log file
    train_log_f = open(os.path.join(config["path"]["log_dir"], "train_log.txt"), "w")
    val_log_f = open(os.path.join(config["path"]["log_dir"], "val_log.txt"), "w")

    # Training loop
    for epoch in range(config["epochs"]):
        avg_loss = train(
            model, train_dataloader, optimizer, loss_fn, train_collate_fn, epoch, train_log_f
        )
        val_avg_loss = validate(
            model, val_dataloader, loss_fn, val_collate_fn
        )
        logit_scale_value = model.logit_scale.exp().item()
        log_line = f"===> Epoch {epoch} Avg Loss: {avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, Temperature: {logit_scale_value:.4f}"
        print(log_line)
        train_log_f.write(log_line + "\n")

        log_line = f"Epoch {epoch} Val Loss: {val_avg_loss:.4f}, Temperature: {logit_scale_value:.4f}"
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
