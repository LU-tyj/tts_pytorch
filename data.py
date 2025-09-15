import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display


def show_mel(mel_db, sr):
    """看看mel spectrogram"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=256,
                            x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram")
    plt.tight_layout()
    plt.show()

def load_pairs(wav_dir, use="text"):
    """
    加载 (wav, text) 对
    wav_dir: 存放 wav 文件的目录
    use: "text", "pinyin", "phone" 选择使用哪一行作为文本
    返回: [(wav_path, text), ...]
    """
    pairs = []
    datapath = "database/data/"
    for fname in os.listdir(wav_dir):
        if not fname.endswith(".wav"):
            continue
        wav_path = os.path.join(datapath, fname)
        trn_path = os.path.join(datapath, fname + ".trn")
        if not os.path.exists(trn_path):
            continue
        with open(trn_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # 假设第一行是真正的转写
            if use == "text":
                text = lines[0].strip()
            elif use == "pinyin":
                text = lines[1].strip()
            else:
                use = "phone"
                text = lines[2].strip()
        pairs.append((wav_path, text))
    return pairs

def preprocess_wavs(pairs, out_dir, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
    os.makedirs(out_dir, exist_ok=True)
    """预处理所有 wav，保存 mel 到 out_dir"""
    mel_paths = []

    for wav_path, text in pairs:
        # 读 wav
        y, _ = librosa.load(wav_path, sr=sr)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 保存为 .npy
        mel_name = os.path.splitext(os.path.basename(wav_path))[0] + ".npy"
        mel_path = os.path.join(out_dir, mel_name)
        np.save(mel_path, mel_db.astype(np.float32))
        mel_paths.append((mel_path, text))

    return mel_paths
 
class TextMelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.char2idx = {ch: idx for idx, ch in enumerate(
            " abcdefghijklmnopqrstuvwxyz'.,!?-1234567890")}

    def __len__(self):
        return len(self.pairs)

    def text_to_sequence(self, text: str):
        return [self.char2idx[ch] for ch in text.lower() if ch in self.char2idx]

    def __getitem__(self, idx):
        mel_path, text = self.pairs[idx]
        mel = np.load(mel_path)
        text_seq = self.text_to_sequence(text)
        return torch.tensor(text_seq, dtype=torch.long), torch.tensor(mel.T, dtype=torch.float32)

def collate_fn(batch):
    """batch 是一个 list，里面每个元素是 __getitem__ 的返回值
    这里需要把它们拼成一个 batch
    """
    # 按文本长度排序，长的在前面
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    texts, mels = zip(*batch)
    text_lengths = [t.size(0) for t in texts]
    mel_lengths = [m.size(0) for m in mels]

    # 填充文本
    max_text_len = max(text_lengths)
    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, :t.size(0)] = t

    # 填充 mel
    max_mel_len = max(mel_lengths)
    mel_dim = mels[0].size(1)
    padded_mels = torch.zeros(len(mels), max_mel_len, mel_dim, dtype=torch.float32)
    for i, m in enumerate(mels):
        padded_mels[i, :m.size(0), :] = m

    return padded_texts, torch.tensor(text_lengths, dtype=torch.long), padded_mels, torch.tensor(mel_lengths, dtype=torch.long)

