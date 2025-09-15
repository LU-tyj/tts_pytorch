import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from data import load_pairs, preprocess_wavs, TextMelDataset, collate_fn
from train import train
from module.tacotron import tacotron

def main():
    # 超参数
    batch = 16
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    wav_dir = "database/train/"
    test_dir = "database/test/"
    train_pairs = load_pairs(wav_dir, use="pinyin")
    test_pairs = load_pairs(test_dir, use="pinyin")

    train_mel_dir = "output/train/"
    test_mel_dir = "output/test/"
    
    print("Preprocessing audio files...")
    train_mel_paths = preprocess_wavs(train_pairs, train_mel_dir)
    test_mel_paths = preprocess_wavs(test_pairs, test_mel_dir)
    print("Preprocessing completed.")

    train_dataset = TextMelDataset(train_mel_paths)
    test_dataset = TextMelDataset(test_mel_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=collate_fn)

    model = tacotron(n_vocab=len(train_dataset.char2idx), embedding_dim=256, mel_dim=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train(model, optimizer, criterion, train_loader, test_loader, num_epochs, device)

if __name__ == "__main__":
    print("Starting run...")
    main()
    print("Run finished.")