import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bi_lstm import DomainGenerator, save_model
import pickle
from torch.autograd import Variable
from torch.nn.parallel import replicate
from torch.autograd import Function, grad

# 超参数
class Args:
    def __init__(self):
        self.batch_size = 128
        self.hidden_dim = 1024
        self.window = 32
        self.learning_rate = 0.00001
        self.num_epochs = 200
        self.sequences = list()
        self.targets = list()
        
args = Args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ids = [0,1,2,3]

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

# 读取输入文件并提取域名
def load_data(file_path):
    with open(file_path, 'r') as f:
        domains = f.read().strip().splitlines()
    # 在每个域名后面加一个空格，并将它们连接成一个字符串
    concatenated_domains = ' '.join(domain.strip() + ' ' for domain in domains).strip()
    return concatenated_domains

# 将字符映射为数字，以用于模型输入
def create_char_mapping(domains):
    char2idx = dict()
    idx2char = dict()
    idx =0
    for char in domains:
        if char not in char2idx.keys():
            char2idx[char] = idx
            idx2char[idx] = char
            idx += 1
    return char2idx, idx2char

def build_sequences(text, char2idx, args):
    x = list()
    y = list()
    window = args.window
    
    for i in range(len(text)):
        try:
            sequence = text[i:i+window]
            sequence = [char2idx[char] for char in sequence]
            
            target = text[i+window]
            target = char2idx[target]
            
            x.append(sequence)
            y.append(target)
        except:
            pass
    
    x = np.array(x)
    y = np.array(y)
    
    args.sequences = x
    args.targets = y

history = {}

def train(args, model):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # 优化器初始化
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 定义batch数
    num_batches = int(len(args.sequences) / args.batch_size)
    # 训练模型
    model.train()
    # 训练阶段
    for epoch in range(args.num_epochs):
        history['tr_loss'] = []
        model.module.init_hidden(args.batch_size) # 每个epoch开始时重置隐藏状态
        for i in range(num_batches):
        # Batch 定义
            try:
                x_batch = args.sequences[i * args.batch_size : (i + 1) * args.batch_size]
                y_batch = args.targets[i * args.batch_size : (i + 1) * args.batch_size]
            except:
                x_batch = args.sequences[i * args.batch_size :]
                y_batch = args.targets[i * args.batch_size :]
                #model.module.reset_hidden_state(args)
            
            # 转换 numpy array 为 torch tensors
            x = torch.tensor(x_batch, dtype=torch.long).cuda()
            y = torch.tensor(y_batch, dtype=torch.long).cuda()
            
            #x= x.reshape((x.shape[0], x.shape[1], 1))
            # 输入数据
            y_pred = model(x)
            # loss计算
            loss = criterion(y_pred, y.squeeze())
            # 清除梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            history['tr_loss'].append(loss.cpu().data.numpy())
            
        print("Epoch: %d ,  loss: %.5f " % (epoch, np.mean(history['tr_loss'])))

def main(input_file):
    domains = load_data(input_file)
    char2idx, idx2char = create_char_mapping(domains)
    vocab_size = len(char2idx)
    
    build_sequences(domains, char2idx, args)

    model = DomainGenerator(args, vocab_size)
    model = nn.DataParallel(model, device_ids=ids).cuda()
    train(args, model)

    save_model(model, 'model/banjori/banjori_3_generator_final.pth')
    
    with open('pickle/banjori/banjori_3_char_mappings.pkl', 'wb') as f:
        pickle.dump((char2idx, idx2char, args), f)
        
    print(f'Model saved to model/banjori/banjori_3_generator_final.pth')

if __name__ == "__main__":
    main("data/banjori/banjori1.txt")
