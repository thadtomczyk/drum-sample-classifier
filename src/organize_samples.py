import os
import glob
import shutil
import numpy as np
import librosa
import torch

from model import DrumCNN
from dataset import CLASS_NAMES

def load_mel(path, sample_rate=22050, n_mels=64, max_duration=0.5):
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    max_len = int(max_duration * sample_rate)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        pad = max_len - len(y)
        y = np.pad(y, (0, pad))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db + 80.0) / 80.0
    S_db = S_db.astype(np.float32)
    S_db = np.expand_dims(S_db, axis=(0, 1))  # [1,1,n_mels,time]
    return torch.from_numpy(S_db)

def organize_folder(input_folder, model_path, out_root, conf_thresh=0.6, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = DrumCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(out_root, exist_ok=True)
    for cls in CLASS_NAMES + ["unknown"]:
        os.makedirs(os.path.join(out_root, cls), exist_ok=True)

    wavs = glob.glob(os.path.join(input_folder, "*.wav"))
    for path in wavs:
        mel = load_mel(path).to(device)
        with torch.no_grad():
            logits = model(mel)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            conf = conf.item()
            pred_idx = pred_idx.item()

        if conf >= conf_thresh:
            cls_name = CLASS_NAMES[pred_idx]
        else:
            cls_name = "unknown"

        dst = os.path.join(out_root, cls_name, os.path.basename(path))
        print(f"{os.path.basename(path)} -> {cls_name} ({conf:.2f})")
        shutil.copy2(path, dst)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(base_dir, "data", "raw")
    model_path = os.path.join(base_dir, "drum_cnn.pth")
    out_root = os.path.join(base_dir, "data", "sorted")

    organize_folder(input_folder=input_folder, model_path=model_path, out_root=out_root)

