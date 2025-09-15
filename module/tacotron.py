import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionWrapper, BahdanauAttention, get_mask_from_lengths


# encoder -------
class prenet(nn.Module):
    def __init__(self, in_dim):
        super(prenet, self).__init__()
        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        inputs = self.dropout(self.relu(self.layer1(inputs)))
        inputs = self.dropout(self.relu(self.layer2(inputs)))
        return inputs

class batch_normal(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, activation=None):
        super(batch_normal, self).__init__()
        self.conv1d = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, 
                                    stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.activation = activation

    def forward(self, x):
        outputs = self.conv1d(x)
        outputs = self.batch_norm(outputs)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs
    
class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()
        self.nonlinear = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        self.nonlinear.bias.data.zero_().normal_(0, 0.01)
        self.gate.bias.data.zero_().normal_(0, 0.01)

    def forward(self, x):
        H = F.relu(self.nonlinear(x))
        T = torch.sigmoid(self.gate(x))
        return H * T + x * (1 - T)

class CBHG(nn.Module):
    def __init__(self, input_size, K=16, projections=[128, 128]):
        super().__init__()
        self.input_size = input_size
        self.relu = nn.ReLU()

        self.conv1d_banks = nn.ModuleList(
            [batch_normal(input_size, input_size, kernel_size=k, stride=1, 
                          padding=k//2, activation=self.relu) for k in range(1, K+1)]
        )
        
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        in_sizes = [K * input_size] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        
        self.conv1d_projections = nn.ModuleList(
            [batch_normal(in_size, out_size, kernel_size=3, stride=1, padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(in_sizes, projections, activations)]
        )
        
        self.pre_highway = nn.Linear(projections[-1], input_size, bias=False)
       
        self.highways = nn.ModuleList(
            [Highway(input_size) for _ in range(4)])
        
        self.gru = nn.GRU(input_size, input_size, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths=None):
        residual = x

        # 转换维度以便进行卷积
        if x.size(-1) == self.input_size:
            x = x.transpose(1, 2)
        
        T = x.size(-1)
        
        # 通过卷积banks
        x = torch.cat([conv(x)[:, :, :T] for conv in self.conv1d_banks], dim=1)
        x = self.max_pool1d(x)[:, :, :T]
        
        # 通过投影层
        for conv in self.conv1d_projections:
            x = conv(x)
        
        # 转换回原始维度
        x = x.transpose(1, 2)
        
        # 如果需要，调整维度
        if x.size(-1) != self.input_size:
            x = self.pre_highway(x)
        
        # 残差连接
        x += residual
        
        # 通过highway网络
        for highway in self.highways:
            x = highway(x)
        
        # 如果有输入长度信息，打包序列
        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False)
        
        # 通过双向GRU
        outputs, _ = self.gru(x)
        
        # 如果有输入长度信息，解包序列
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)
        
        return outputs

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.prenet = prenet(input_size)
        self.cbhg = CBHG(input_size=128, K=16, projections=[128, 128])

    def forward(self, x, input_lengths=None):
        x = self.prenet(x)
        output = self.cbhg(x, input_lengths=input_lengths)
        return output


# decoder -------
class Decoder(nn.Module):
    def __init__(self, input_size, r):
        super(Decoder, self).__init__()
        self.input_size = input_size  # mel_dim 
        self.r = r  # 每次输出 r 帧
        # prenet 的输入是 mel_dim * r
        self.prenet = prenet(input_size * r)

        # attention / memory 
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),  # 256: attn_rnn hidden, 128: prenet 输出
            BahdanauAttention(256)
        )
        self.memory_layer = nn.Linear(256, 256, bias=False)
        self.project_to_decoder_in = nn.Linear(512, 256)

        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)]
        )

        self.proj_to_mel = nn.Linear(256, input_size * self.r)  # 输出 r 帧 (mel_dim * r)
        self.max_decoder_steps = 200

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
        B = encoder_outputs.size(0)
        processed_memory = self.memory_layer(encoder_outputs)
        mask = get_mask_from_lengths(memory_lengths) if memory_lengths is not None else None
        greedy = inputs is None

        if inputs is not None:
            # inputs: (B, T, mel_dim)
            if inputs.size(-1) == self.input_size:
                T = inputs.size(1)
                if T % self.r != 0:
                    T = T - (T % self.r)
                    inputs = inputs[:, :T, :]
                # reshape -> (B, T//r, mel_dim * r)
                inputs = inputs.contiguous().view(B, T // self.r, -1)
            assert inputs.size(-1) == self.input_size * self.r, \
                f"Expected input size {self.input_size * self.r}, but got {inputs.size(-1)}"
            T_decoder = inputs.size(1)
        else:
            T_decoder = None

        device = encoder_outputs.device
        dtype = encoder_outputs.dtype

        # 初始化 current_input：注意是 mel_dim * r
        current_input = torch.zeros(B, self.input_size * self.r, device=device, dtype=dtype)
        attention_rnn_hidden = torch.zeros(B, 256, device=device, dtype=dtype)
        decoder_rnn_hiddens = [
            torch.zeros(B, 256, device=device, dtype=dtype)
            for _ in range(len(self.decoder_rnns))
        ]
        current_attention = torch.zeros(B, 256, device=device, dtype=dtype)

        if inputs is not None:
            inputs = inputs.transpose(0, 1)  # (time, batch, mel_dim * r)

        outputs = []
        alignments = []
        t = 0

        while True:
            if t > 0:
                # Greedy: 用上一步 decoder 输出的完整 r 帧向量；teacher forcing: 用 inputs
                current_input = outputs[-1] if greedy else inputs[t - 1]

            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(
                current_input, current_attention, attention_rnn_hidden,
                encoder_outputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            output = self.proj_to_mel(decoder_input)  # (B, mel_dim * r)
            outputs.append(output)
            alignments.append(alignment)

            t += 1

            if greedy:
                if t > 1 and is_end_of_frames(output):
                    break
                elif t > self.max_decoder_steps:
                    print("Warning! doesn't seems to be converged")
                    break
            else:
                if t >= T_decoder:
                    break

        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments

def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()

# tacotron ------
class tacotron(nn.Module):
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super().__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.use_memory_mask = use_memory_mask
        self.embedding = nn.Embedding(n_vocab, embedding_dim,
                                      padding_idx=padding_idx)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    def forward(self, inputs, targets=None, input_lengths=None):
        B = inputs.size(0)

        inputs = self.embedding(inputs)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None
        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        # Post net processing below

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments