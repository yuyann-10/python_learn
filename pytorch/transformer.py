import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
#Transformer 的不同组件:Multi-Head Attention, feed-forward转换注意力输出，增加复杂性 ,positional encoding为嵌入添加位置信息,
#layer normalization将输入正规化到每个子层,residual connections帮助梯度流动有助于通过最小化梯度问题来训练更深层的网络
#dropout用于防止过拟合。
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):# d_model : 输入的维度。

        #num_heads : 将输入分成多少个注意力头。
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):#缩放点积注意力
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #注意力分数是通过将查询（ Q ）和键（ K ）进行点积计算，然后除以键维度的平方根（ d_k ）来缩放的
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        #如果提供了掩码，它将应用于注意力分数以遮蔽特定值。
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        #注意力机制的最终输出是通过将注意力权重与值（ V ）相乘来计算的。
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        #这种方法将输入 x 重塑为形状 ( batch_size , num_heads , seq_length , d_k )。它使模型能够同时处理多个注意力头，从而实现并行计算。
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        #在分别对每个头应用注意力后，该方法将结果重新组合成一个形状为 ( batch_size , seq_length , d_model ) 的单一张量。这为后续处理做好了准备。
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        #应用线性变换：查询（ Q ）、键（ K ）和值（ V ）首先通过初始化中定义的权重进行线性变换。
        #通过 split_heads 方法，将转换后的 Q 、 K 、 V 拆分成多个头。
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    '''MultiHeadAttention 类封装了 Transformer 模型中常用的多头注意力机制。
    它负责将输入拆分为多个注意力头，对每个头应用注意力，然后将结果合并。
    通过这种方式，模型能够在不同尺度上捕获输入数据中的各种关系，提高模型的表达能力。'''

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        ''' d_model : 模型的输入和输出维度。
        d_ff : 前馈网络中内层维度。
        self.fc1 和 self.fc2 ：两个全连接（线性）层，其输入和输出维度由 d_model 和 d_ff 定义。'''


    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    '''linear -> relu -> linear 总而言之， PositionWiseFeedForward 类定义了一个位置感知的前馈神经网络，
    该网络由两个线性层组成，中间包含一个 ReLU 激活函数。在 Transformer 模型的上下文中，
    这个前馈网络会分别且相同地应用于每个位置。它有助于转换 Transformer 内部注意力机制学习
    到的特征，作为注意力输出的额外处理步骤。'''    

class PositionalEncoding(nn.Module):
    #位置编码用于向输入序列中的每个标记注入位置信息。它使用不同频率的正弦和余弦函数来生成位置编码。
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        ''' d_model : 模型输入的维度。
max_seq_length : 用于预计算位置编码的序列的最大长度。
pe : 一个用零填充的张量，将用位置编码填充。
position : 一个包含序列中每个位置的位置索引的张量。
div_term : 一种用于以特定方式缩放位置索引的术语。
对 pe 的偶数索引应用正弦函数，对奇数索引应用余弦函数。
最后， pe 被注册为一个 buffer，这意味着它将作为模块状态的一部分，但不会被考虑为可训练的参数。

前向方法只是将位置编码加到输入 x 上。
它使用 pe 的前 x.size(1) 个元素来确保位置编码与 x 的实际序列长度相匹配。'''
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):


    '''  参数：

    d_model : 输入的维度。
    num_heads : 多头注意力机制中的注意力头数。
    d_ff : 位置前馈网络中内层的维度。
    dropout : 用于正则化的 dropout 率。

    组件：

    self.self_attn ：多头注意力机制。
    self.feed_forward : 位置前馈神经网络。
    self.norm1 和 self.norm2 ：层归一化，用于平滑层的输入。
    self.dropout : Dropout 层，在训练过程中通过随机将一些激活值设为零来防止过拟合。

    EncoderLayer 类定义了 Transformer 编码器的一个单层。它封装了多头自注意力机制，
    随后是位置感知的前馈神经网络，并根据需要应用了残差连接、层归一化和 dropout。
    这些组件共同使编码器能够捕获输入数据中的复杂关系，并将它们转换为下游任务的有用表示。
    通常，多个这样的编码器层被堆叠起来，形成 Transformer 模型的完整编码器部分。
'''
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)#多头注意力机制，用于关注编码器的输出。
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)# 位置感知的前馈神经网络。
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)# 层归一化组件。
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))#多头自注意力机制的输出经过残差连接和层归一化。
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    '''输入 :

    x: 解码器层的输入。
    enc_output: 对应编码器的输出（用于交叉注意力步骤）。
    src_mask: 源掩码，用于忽略编码器输出的一部分。
    tgt_mask: 目标掩码，用于忽略解码器输入的一部分。

    处理步骤：

    目标序列上的自注意力 : 输入 x 通过自注意力机制进行处理。
    添加和归一化（自注意力之后）: 自注意力的输出与原始 x 相加，然后进行 dropout 和 norm1 归一化。
    与编码器输出的交叉注意力 : 上一步的归一化输出通过一个交叉注意力机制进行处理，该机制关注编码器输出 enc_output。
    添加和归一化（交叉注意力之后）: 交叉注意力的输出与本阶段的输入相加，然后进行 dropout 和 norm2 归一化。
    前馈网络 ：前一步的输出被传递到前馈网络中。
    添加和归一化（前馈后）：前馈输出被加到这一阶段的输入中，然后进行 dropout 和 norm3 的归一化处理。
    输出 : 处理后的张量作为解码器层的输出返回。

    DecoderLayer 类定义了 Transformer 解码器的一个单层。它封装了多头自注意力机制、
    交叉注意力机制（用于关注编码器的输出），以及位置感知的前馈神经网络。
    这些组件共同使解码器能够生成与输入数据相关的输出表示，通常用于生成任务。'''

class Transformer(nn.Module):
    '''构造函数接受以下参数：

    src_vocab_size: 源词汇表大小。
    tgt_vocab_size: 目标词汇表大小。
    d_model: 模型嵌入的维度。
    num_heads: 多头注意力机制中的注意力头数。
    num_layers: 编码器和解码器中层的数量。
    d_ff: 前馈网络中内层的维度。
    max_seq_length: 位置编码的最大序列长度。
    dropout: 正则化时的 Dropout 比率。

    它定义了以下组件：

    self.encoder_embedding: 源序列的嵌入层。
    self.decoder_embedding: 目标序列的嵌入层。
    self.positional_encoding: 位置编码组件。
    self.encoder_layers: 编码器层列表。
    self.decoder_layers: 解码器层列表。
    self.fc: 映射到目标词汇大小的最终全连接（线性）层。
    self.dropout: Dropout 层。
    '''
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):#这种方法用于为源序列和目标序列创建掩码，确保忽略填充标记，并在训练目标序列时使未来的标记不可见。
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        #输入嵌入和位置编码 ：源序列和目标序列首先使用各自的嵌入层进行嵌入，然后加上它们的位置编码。
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        #编码器层 ：源序列通过编码器层，最终的编码器输出代表处理后的源序列。
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        #解码器层 ：目标序列和编码器的输出通过解码器层，产生解码器的输出。
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        #最终线性层 ：解码器的输出通过一个全连接（线性）层映射到目标词汇量大小。
        output = self.fc(dec_output)
        return output

#样本数据准备   
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1
'''超参数：

这些值定义了 Transformer 模型的架构和行为：

    src_vocab_size, tgt_vocab_size: 源序列和目标序列的词汇表大小，均设置为 5000。
    d_model: 模型嵌入的维度，设置为 512。
    num_heads: 多头注意力机制中的注意力头数量，设置为 8。
    num_layers: 编码器和解码器中的层数，设置为 6。
    d_ff: 前馈网络中内部层的维度，设置为 2048。
    max_seq_length: 位置编码的最大序列长度，设置为 100。
    dropout: 正则化使用的 Dropout 比率，设置为 0.1。

作为参考，下表描述了 Transformer 模型中最常见的超参数及其值：
超参数 	典型值 	对性能的影响
d_model 	256, 512, 1024 	更高的值会增加模型容量，但需要更多的计算
num_heads 	8, 12, 16 	更多的注意力头可以捕捉数据的多样性，但计算成本较高
num_layers 	6, 12, 24 	更多层可以提高表示能力，但可能导致过拟合
d_ff 	2048, 4096 	更大的前馈网络增加模型鲁棒性
dropout 	0.1, 0.3 	正则化模型以防止过拟合
学习率 	0.0001 - 0.001 	影响收敛速度和稳定性
批大小 	32, 64, 128 	较大的批量大小可以提高学习稳定性，但需要更多内存
'''
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
'''损失函数和优化器：

    criterion = nn.CrossEntropyLoss(ignore_index=0): 将损失函数定义为交叉熵损失。参数 ignore_index 设置为 0，表示损失计算将忽略索引为 0 的目标（通常用于填充标记）。
    optimizer = optim.Adam(...)：将优化器定义为学习率为 0.0001 且具有特定 beta 值的 Adam 优化器。
'''
#训练模型
transformer.train()
'''代码片段使用典型的训练循环对模型进行 100 个 epoch 的训练：

    for epoch in range(100): 遍历 100 个训练周期。
    optimizer.zero_grad(): 清除前一次迭代的梯度。
    output = transformer(src_data, tgt_data[:, :-1]) : 将源数据与目标数据（每个序列中排除最后一个标记）通过 transformer 传递。这在目标序列比源序列延迟一个标记的序列到序列任务中很常见。
    loss = criterion(...): 计算模型预测与目标数据（每个序列中排除第一个标记）之间的损失。通过将数据重塑为一维张量并使用交叉熵损失函数来计算损失。
    loss.backward(): 计算损失相对于模型参数的梯度。
    optimizer.step(): 使用计算出的梯度更新模型的参数。
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}") : 打印当前轮数和该轮的损失值。

    '''
for epoch in range(100):
    optimizer.zero_grad()#清除之前的梯度
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
#Transformer 模型性能评估
'''评估模式：

    transformer.eval(): 将 transformer 模型置于评估模式。这很重要，因为它会关闭某些仅在训练时使用的功能，如 dropout。

生成随机验证数据：

    val_src_data: 介于 1 和 src_vocab_size 之间的随机整数，表示形状为(64, max_seq_length)的验证源序列批次。
    val_tgt_data: 1 到 tgt_vocab_size 之间的随机整数，表示形状为(64, max_seq_length)的验证目标序列批次。

验证循环：

    with torch.no_grad(): 禁用梯度计算，因为在验证过程中不需要计算梯度。这可以减少内存消耗并加快计算速度。
    val_output = transformer(val_src_data, val_tgt_data[:, :-1]) : 将验证源数据（不包括每个序列的最后一个标记）和验证目标数据（不包括每个序列的第一个标记）通过 transformer 传递。
    val_loss = criterion(...): 计算模型预测与验证目标数据（不包括每个序列的第一个标记）之间的损失。通过将数据重塑为一维张量，并使用先前定义的交叉熵损失函数来计算损失。
    print(f"Validation Loss: {val_loss.item()}") : 打印验证损失值。

    '''
transformer.eval()

# Generate random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")