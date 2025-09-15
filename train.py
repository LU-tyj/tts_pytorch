from tqdm import tqdm
import torch
import torch.nn as nn

def train(model, optimizer, criterion, train_loader, val_loader=None, 
          num_epochs=10, device='cuda', save_path="tacotron.pth"):
    """
    训练 Tacotron 模型
    
    参数:
        model: Tacotron 模型实例
        optimizer: 优化器
        criterion: 损失函数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 (可选)
        num_epochs: 训练轮数
        device: 训练设备
        save_path: 模型保存路径
    """
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_loop:
            # 解包批次数据
            texts, text_lengths, mels, mel_lengths = batch
            texts, mels = texts.to(device), mels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            mel_outputs, linear_outputs, alignments = model(texts, mels, text_lengths)
            
            # 计算损失 (只使用mel输出计算损失) 限定mel长度一致
            min_len = min(mel_outputs.size(1), mels.size(1))
            loss = criterion(mel_outputs[:, :min_len, :], mels[:, :min_len, :])

            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss = {avg_train_loss:.4f}")
        
        # 验证阶段 (如果有验证数据)
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_loop:
                    texts, text_lengths, mels, mel_lengths = batch
                    texts, mels = texts.to(device), mels.to(device)
                    
                    # 前向传播
                    mel_outputs, linear_outputs, alignments = model(texts, mels, text_lengths)
                    
                    # 计算损失
                    loss = criterion(mel_outputs, mels)
                    
                    # 更新统计信息
                    total_val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())
            
            # 计算平均验证损失
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss = {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            # 如果没有验证集，只保存最终模型
            torch.save(model.state_dict(), save_path)
    
    print("训练完成!")
