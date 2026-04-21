from importlib import metadata
import numpy as np
import scipy as sp
import torch
import random
import os
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import Conformer
import math
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import csv
# 从 notebook 导入 tqdm 用于命令行环境
from tqdm import tqdm


def set_seed(seed):
    """设置随机种子以确保结果可复现 (Set random seed for reproducibility)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class myDataset(Dataset):  # 自定义数据集类，用于处理训练和验证数据。(Custom Dataset class)
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len
        # Load mapping.json
        mapping_path = Path(data_dir) / "mapping.json"
        # 使用 mapping_path.open() 来打开文件
        mapping = json.load(mapping_path.open()) 
        self.speaker2id = mapping["speaker2id"]

        # Load metadata.json
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]
        self.speaker_num = len(metadata.keys())
        self.data = []
        # 构建数据列表 (feature path, speaker id)
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"],self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # load preprocesed mel-spectrogram
        mel = torch.load(os.path.join(self.data_dir , feat_path))
        
        # 随机切割或保持不变 (Random segment or keep full length)
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        
        speaker = torch.LongTensor([speaker]).long()
        return mel, speaker
    
    def get_speaker_number(self):
        return self.speaker_num
    

#对一个批次内的所有梅尔频谱图进行零填充 (padding)，使其具有相同的长度。(Padding Mel-spectrograms)
def collate_batch(batch):
    """自定义的collate函数，用于DataLoader中对批次数据进行处理 (Custom collate function)"""
    mel, speaker = zip(*batch)
    # 使用 -20 作为 padding value (通常是 mel 频谱图中的最小值，代表静音)
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)

    return mel, torch.FloatTensor(speaker).long()

#创建 myDataset 实例和 DataLoader。(Create myDataset instance and DataLoader)
def get_dataloader(data_dir, batch_size, n_workers):
    '''
    将数据集按 90% 训练集, 10% 验证集 的比例进行随机分割 (random_split)。
    创建 DataLoader 实例，用于批量迭代数据，并使用 collate_batch 进行批次处理。
    '''

    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    trainlen = int (0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)
    
    # 训练集 DataLoader
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch,
                              pin_memory=True, num_workers=n_workers)
    # 验证集 DataLoader
    valid_loader = DataLoader(validset, batch_size=batch_size, drop_last=True, collate_fn=collate_batch,
                              pin_memory=True, num_workers=n_workers)
                              
    return train_loader, valid_loader, speaker_num


#模型架构与损失函数（Model & Loss）
class SelfAttentionPooling(nn.Module):
    '''实现了自注意力池化 (Self-Attention Pooling, SAP) 机制。
    (Self-Attention Pooling (SAP) mechanism for sequence to utterance level feature aggregation.)
    '''
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)# 线性层用于学习原始注意力得分 (Linear layer for raw attention scores)

    def forward(self, batch_rep):
        softmax = nn.functional.softmax
        
        # 1. 计算原始得分，并通过 Softmax 归一化得到注意力权重 (N, T, 1)
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        
        # 2. 加权求和得到话语级表示 (Weighted sum to get utterance representation)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        
        return utter_rep
        
#一个带有标签平滑 (Label Smoothing) 的交叉熵损失函数。(Cross Entropy Loss with Label Smoothing)
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer."""

    def __init__(self, num_classes=600, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes # K (总类别数)
        self.epsilon = epsilon # epsilon (平滑系数)
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)# PyTorch 的 log(Softmax) 函数。

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: 模型的预测结果（logits），尺寸为 (batch_size,num_classes)
            targets: 真实标签，尺寸为 (batch_size)
        """
        # 1. 计算 log(Softmax)
        log_probs = self.logsoftmax(inputs)
        
        # 2. 生成 One-Hot 目标向量 (Generate One-Hot target vector)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        
        if self.use_gpu: 
             # 将独热码目标向量转移到 GPU (Move one-hot target to GPU)
             targets = targets.to(log_probs.device) 
             
        # 3. 标签平滑 (Label Smoothing)
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            
        # 4. 计算损失 (Calculate loss)
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
#减少过拟合，增强泛化能力的 AMSoftmax Loss (AMSoftmax Loss to enhance generalization)
class AMSoftmaxLoss(nn.Module):
    def __init__(self, m=0.35, s=30, d=512, num_classes=600, use_gpu=True , epsilon=0.1):
        super(AMSoftmaxLoss, self).__init__()
        self.m = m              # 加性边际 (Additive Margin)
        self.s = s              # 缩放因子 (Scale Factor)
        self.num_classes = num_classes
        # 使用前面定义的带有标签平滑的交叉熵作为基底损失 (Base loss is CrossEntropy with Label Smoothing)
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes , use_gpu=use_gpu, epsilon=epsilon)

    def forward(self, features, labels , classifier):
        '''
        features : feature vector : (b x  d) b= batch size d = dimension
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC)
        '''
        # 1. 对特征向量进行 L2 归一化 (L2 normalize features)
        features = nn.functional.normalize(features, p=2, dim=1)

        # 2. 对分类器权重进行 L2 归一化 (L2 normalize classifier weights)
        with torch.no_grad():
            classifier.weight.div_(torch.norm(classifier.weight, dim=1, keepdim=True))

        # 3. 计算余弦相似度（归一化后的点积）(Calculate cosine similarity)
        cos_angle = classifier(features)
        
        # 4. 限制余弦相似度范围以防止数值不稳定 (Clamp cosine similarity)
        cos_angle = torch.clamp( cos_angle , min = -1 , max = 1 )
        
        # 5. 在正确类别上应用加性边际 (Additive Margin)
        b = features.size(0)
        # 优化：使用 scatter_ 代替 for 循环以提高效率
        # 创建一个掩码，只在目标类别的位置减去 m
        one_hot = torch.zeros_like(cos_angle)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        cos_angle = cos_angle - one_hot * self.m
            
        # 6. 应用缩放因子 (Scale Factor)
        weighted_cos_angle = self.s * cos_angle
        
        # 7. 使用带有标签平滑的交叉熵计算损失
        log_probs = self.CrossEntropy(weighted_cos_angle , labels, use_label_smoothing=True)
        return log_probs
    
class Classifier(nn.Module):
    def __init__(self, d_model=512, n_spks=600, dropout=0.2):
        super().__init__()
        # 1. Mel特征投影层 (Prenet: Project Mel features from 40 to d_model)
        self.prenet = nn.Linear(40, d_model)

        # 2. Conformer 编码器 (Conformer Encoder)
        self.encoder = Conformer(
                dim = d_model,
                depth = 2,# 层数 (Number of blocks)
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                conv_expansion_factor = 2,
                conv_kernel_size = 31,
                attn_dropout = dropout,
                ff_dropout = dropout,
                conv_dropout = dropout
            )
            
        # 3. 自注意力池化层 (Self-Attention Pooling)
        self.sap = SelfAttentionPooling(d_model)

        # 4. 分类层 (Classification Layer - used as weights in AMSoftmax)
        self.pred_layer = nn.Linear(d_model, n_spks) 

    def forward(self, mels):
        """
        mels: (batch, length, 40)
        out: (batch, n_spks)
        """
        # (B, L, 40) → (B, L, D)
        out = self.prenet(mels)

        # (B, L, D) → (L, B, D) for Conformer input
        out = out.permute(1, 0, 2)

        # 送入 Conformer (Pass through Conformer)
        # out: (L, B, D)
        out = self.encoder(out)

        # (L, B, D) → (B, L, D) 转回批次优先格式 (Permute back to Batch-first)
        out = out.transpose(0, 1)

        # 池化成固定长度向量 (B, D) (Pool to fixed-length vector)
        out = self.sap(out)

        # 返回说话人嵌入特征 (Return speaker embedding feature)
        # 注意: pred_layer(out) 放在 model_fn/AMSoftmaxLoss 中执行
        return out
        
#带 warmup 的余弦退火学习率调度 (Cosine schedule with warmup)
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Create a schedule with a learning rate that decreases following the values of the cosine function."""
    def lr_lambda(current_step):
        # Warmup (线性上升)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Decadence (余弦下降)
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

#model_fn 封装了前向传播 + 损失计算 + 预测 + 精度统计 (Function to wrap forward pass, loss, and accuracy)

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)
    
    # 得到说话人嵌入特征 (Get speaker embedding feature)
    outs_feat = model(mels)
    
    # 计算损失 (Calculate loss using AMSoftmax)
    loss = criterion(outs_feat, labels, model.pred_layer)

    # 计算准确率: 需要通过 pred_layer 得到最终 logits (Calculate accuracy using logits)
    outs_logits = model.pred_layer(outs_feat)
    preds = outs_logits.argmax(1)
 
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy

# Validate (验证)
def valid(dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    # 使用 tqdm 进度条
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)

def parse_args_train():
    """训练参数 (Training Arguments)"""
    config = {
        "data_dir": "/home/user1/Downloads/ml2023springhw4/Dataset", # 数据集路径
        "save_path": "model.ckpt", # 模型保存路径
        "batch_size": 32,
        "n_workers": 4, # DataLoader 工作线程数
        "valid_steps": 2000, # 验证间隔步数
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 70000,
    }

    return config


"""主训练函数 (Main Training Function)"""
def train_main(
    data_dir,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    model = Classifier(n_spks=speaker_num).to(device)
    # 使用 AMSoftmaxLoss
    criterion = AMSoftmaxLoss(m=0.4, s=30, num_classes=speaker_num)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!",flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


# Inference推断 (Inference Block)

class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        # Load testdata.json
        testdata_path = Path(data_dir) / "testdata.json"
        # 使用 testdata_path.open() 来打开文件
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)

## main function of inference


def parse_args_inference():
    """推断参数 (Inference Arguments)"""
    config = {
        "data_dir": "/home/user1/Downloads/ml2023springhw4/Dataset", # 数据集路径
        "model_path": "./model.ckpt", # 加载的模型路径
        "output_path": "./output.csv", # 结果输出路径
    }

    return config


def inference_main(
    data_dir,
    model_path,
    output_path,
):
    """主推断函数 (Main Inference Function)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1, # 推断时通常使用较小的批次大小，这里为 1
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!",flush = True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    # 加载训练好的模型权重 (Load trained model state)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!",flush = True)

    results = [["Id", "Category"]]
    # 使用 tqdm 进度条
    for feat_paths, mels in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            mels = mels.to(device)
            outs_feat = model(mels)
            # 经过 pred_layer 得到最终 logits
            outs_logits = model.pred_layer(outs_feat)  
            preds = outs_logits.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    # 将结果写入 CSV 文件 (Write results to CSV)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
        
    print(f"[Info]: Inference results saved to {output_path}", flush=True)


if __name__ == "__main__":
    # 为了方便在命令行运行，这里需要手动判断是执行训练还是推断
    # 由于原始代码中 parse_args() 冲突，我们假设默认执行训练。
    # 如果要执行推断，需要将 train_main(**parse_args_train()) 替换为 inference_main(**parse_args_inference())
    # 或者通过命令行参数传入不同配置来区分。
    
    # 默认执行训练 (Default execution: Training)
    # train_main(**parse_args_train())

    # 如果需要执行推断，请取消下一行注释并注释掉上一行
    inference_main(**parse_args_inference())