import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

'''
使用bert模型训练外卖评价数据集，进行情感分类
'''

#1.数据准备
data = pd.read_csv('./02-sentiment-classify/assets/dataset/waimai.csv',sep=',')
label = data['label'].to_list()
text = data['review'].to_list()
#数据太大电脑跑不动，截取500条数据
text_new = text[:250]+text[-250:]
label_new = label[:250]+label[-250:]

x_train,x_test,lable_train,lable_test = train_test_split(
    text_new,
    label_new,
    test_size = 0.02,
    stratify = label_new
)

#加载Bert预训练的分词器
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

#print(test_encoding.items())
#print(test_encoding['input_ids'][0:2])
#input_ids token的ID序列
#token_type_ids 区分句子1和句子2 
#attention_mask 注意力掩码

#2.数据集和数据装载器
class WmaiDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        #从编码字典提取input_ids,token_type_ids,attention_mask
        item = {key : torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = WmaiDataset(train_encoding,lable_train)
test_dataset = WmaiDataset(test_encoding,lable_test)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16,shuffle=True)

#3.模型和优化器
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese',num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optim = torch.optim.AdamW(model.parameters(),lr=2e-5)

#4.训练和验证函数
def get_accuracy(logits,labels):
    predicts = np.argmax(logits,axis=1).flatten()
    real_labels = labels.flatten()
    return np.sum(predicts == real_labels) / len(real_labels)

def train():
    model.train()
    total_train_loass = 0
    iter_num = 0
    total_iter = len(train_loader)

    for batch_data in train_loader:
        #清除上一轮梯度
        optim.zero_grad()

        #将批次数据移动到指定设备
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = batch_data['label'].to(device)

        #前向传播
        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        loss = outputs[0]
        total_train_loass += loss.item()

        #反向传播计算梯度
        loss.backward()
        #梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0)

        #更新参数
        optim.step()

        iter_num += 1
        if (iter_num % 10 == 0):
            print("epoch: %d,iter_num: %d, loss: %.4f ,%.2f%%" % (epoch,iter_num,loss.item(),iter_num/total_iter*100))

    #平均训练损失
    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loass / len(train_loader)))

def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch_data in test_loader:
        with torch.no_grad():
            #将批次数据移动到指定设备
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['label'].to(device)

            #前向传播
            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]  #[batch_size,num_labels]

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        #正确率
        total_eval_accuracy += get_accuracy(logits, labels)

    # 计算平均准确率
    avg_val_accuracy = total_eval_accuracy / len(test_loader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))

    #保存最佳模型
    global best_accuracy
    if avg_val_accuracy > best_accuracy:
        best_accuracy = avg_val_accuracy
        torch.save(model.state_dict(),'./02-sentiment-classify/assets/weights/best_model.pt')
        print("Saved new best model with accuracy: %.4f" % avg_val_accuracy)

best_accuracy = 0

#5.主训练循环
for epoch in range(4):
    print("-----------------Epoch: %d-------------------" % epoch)
    #训练
    train()
    #验证
    validation()
