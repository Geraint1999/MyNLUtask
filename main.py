import pandas
from transformers import BertTokenizer,BertModel
import numpy as np
import torch

train_set = pandas.read_csv(r'data/gt_train_set.csv',sep='|')
Tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese',from_tf=True)


#将标签从文本映射到数字
def create_label(train_set):
    #pandas -> dict
    index = 0
    dict = {}
    for i in range(len(train_set)):
        if train_set.loc[i]['label'] not in dict:
            dict[train_set.loc[i]['label']] = index
            index += 1
    return dict
label_map = create_label(train_set)
# print(len(label_map))

#创建数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        labels = dataset[:,2]
        for i in range(len(labels)):
            labels[i] = label_map[labels[i]]
        labels = np.array(labels, dtype=np.compat.long)
        records = dataset[:,3]
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)
        self.texts = [Tokenizer(text,padding = 'max_length',max_length = 16,truncation=True,return_tensors='pt') for text in records]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y  
# print(instance.labels[0:10])
# print(instance.texts[0:10])


#创建分类器类
from torch import nn 
class BertClassifier(nn.Module):
    def __init__(self,dropout=0.5):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-chinese',from_tf=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768,len(label_map))
        self.relu = nn.ReLU()

    def forward(self,input_id,mask):
        _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


#模型训练函数
from torch.optim import Adam
from tqdm import tqdm

def train(model, train_data, val_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
      # 进度条函数tqdm
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
                output = model(input_id, mask)
                # 计算损失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
        # 模型更新
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0
        # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for val_input, val_label in val_dataloader:
          # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''') 

def evaluate(model, test_data, label_map):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size = 1)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    total_acc_test = 0
    total_f1 = 0

    if use_cuda:
        model = model.cuda()
        
    #创建一张各个标签对应正确数量的表
    label_acc = {}
    label_actual_count = 0
    for i in range(len(label_map)):
        label_acc[i] = [0,0,0,None,None,None] #TP FP FN 精确率P 召回率R F1
    
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            # print(test_input,test_label)
            test_label = test_label.to(device)
            test_label = test_label.item()
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            output = output.argmax(dim=1).item()
            if output == test_label:
                label_acc[output][0] += 1
                total_acc_test += 1
            else:
                label_acc[output][1] += 1 #对应output标签中，错误数量加一
                label_acc[test_label][2] += 1 #对应test_label标签中，未召回数量加一
    for label_num in label_acc:
        #准确率
        if label_acc[label_num][0] + label_acc[label_num][1] != 0:
            P = label_acc[label_num][3] = label_acc[label_num][0] / (label_acc[label_num][0] + label_acc[label_num][1])
        else: P = 0
        #召回率
        if label_acc[label_num][0] + label_acc[label_num][2] != 0:
            R = label_acc[label_num][4] = label_acc[label_num][0] / (label_acc[label_num][0] + label_acc[label_num][2])
        else: R = 0
        #F1
        if P+R != 0:
            F1 = label_acc[label_num][5] = 2*(P*R)/(P+R)
            label_actual_count += 1
            total_f1 += F1
        
    print(f'''Test Accuracy: {total_acc_test / len(test_data): .3f}
    macro_F1: {total_f1 / label_actual_count: .3f}''')

    
def data_cleaning(train_set):
    train_set_pro = None
    index = 0
    current_text = train_set[0,3]
    current_label = train_set[0,2]
    current_callid = train_set[0,0]
    while index <= len(train_set)-1:
        if index < len(train_set)-1 and train_set[index,0] == train_set[index+1,0]:
            if train_set[index,3] != 'nann' and train_set[index,3] != 'rcgerrorr':
                current_text += train_set[index,3]
            index += 1
            continue
        new_raw = np.array([[current_callid,1.0,current_label,current_text]])
        if train_set_pro is None:
            train_set_pro = np.array(new_raw)
        else:
            train_set_pro = np.append(train_set_pro,new_raw,axis=0)
        index += 1
        if index <= len(train_set)-1:
            current_callid, current_label, current_text = train_set[index,0], train_set[index,2], train_set[index,3]
    return train_set_pro

def main():
    task = input('train or test\n')
    state_dict = torch.load(r'my_bert_model.pth',map_location=torch.device('cpu'))
    df = data_cleaning(np.array(train_set))
    df_train, df_val, df_test = np.split(df,[int(0.8*len(df)),int(0.9*len(df))])
    EPOCHS = 1
    model = BertClassifier()
    model.load_state_dict(state_dict['model'])
    print('load_over')
    LR = 1e-6
    if task == 'train':
        train(model, df_train, df_val, LR, EPOCHS)
        torch.save({'model':model.state_dict()},r'my_bert_model.pth')
    elif task == 'test':
        evaluate(model, df_test, label_map)
    else:
        print('输入错误')

main()
