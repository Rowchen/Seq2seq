import json
import pandas as pd
from tqdm import tqdm
import gc
import numpy as np
import numpy as np
import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init


PAD = 0
UNK = 1
START = 2
END = 3

from tqdm import tqdm
def get_title_mask(data):
    '''title mask'''
    mask=np.ones(data.shape,dtype=np.int)
    for row in tqdm(range(data.shape[0])):
        for i in range(1,data.shape[1]):
            '''leave 1 END to calculate loss,other END is masked'''
            if data[row,i]==0 and data[row,i-1]==0:
                mask[row,i]=0
    return mask

def set_title_end(data):
    for row in tqdm(range(data.shape[0])):
        for i in range(data.shape[1]):
            if data[row,i]==0:
                data[row,i]=END
    return data

def sort_title_len(title_mask):
    title_sum=np.sum(title_mask,axis=1)
    ite=0
    idx=[]
    while ite<title_sum.shape[0]:
        idx.extend(list(ite+np.argsort(title_sum[ite:ite+1000])))
        ite+=1000
    return np.array(idx)

def get_training_data(cnt):
    
    train_title_embed=np.load('../data/title_embed%d.npy'%(cnt%6))
    
    '''remove the title len less than 3'''
    tit_sum=np.sum((train_title_embed!=0),axis=1)
    filter_=((tit_sum>2))
    
    train_title_embed=train_title_embed[filter_]
    
    title_mask=get_title_mask(train_title_embed)
    
    train_title_embed=set_title_end(train_title_embed)
    
    title_len_index=sort_title_len(title_mask)
    print (train_title_embed.shape)
    print ('loading training train_embed%d.npy'%(cnt%6))
    return train_title_embed,title_mask,title_len_index

val_title_embed=np.load('../data/title_embed6.npy')[:4000]
val_title_mask=get_title_mask(val_title_embed)
val_title_embed=set_title_end(val_title_embed)


f = open("../data/vocab2.txt")
line = f.readline()   
vocab=[]
while line:
    w=line.strip('\n')
    if not w.isdigit():
        vocab.append(w)
    line = f.readline()
f.close()

vocab=['PADDING','UNK','START','END','DIGIT']+vocab+['UNK%d'%i for i in range(50)]#UNK没有用，不去掉是为了不影响其他网络
vocab2=set(vocab)
print (len(vocab2))


class elmo_param(object):
    vocab_size=78864
    embedding_size=300
    seq_len=20
    num_layers=1
    lstm_dim=512
    project_dim=512
    dropout=0.3
    
arg=elmo_param()


embed_matrix=np.load('../data/embedding.npy')
embed_matrix[0]=np.zeros(embed_matrix.shape[1])


class dec_elmo(nn.Module):
    def __init__(self, arg):
        super(dec_elmo, self).__init__()
        
        self.embeddings=nn.Embedding(arg.vocab_size, arg.embedding_size).cuda()
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embeddings.weight.requires_grad = True
        
        self.for_lstm1= nn.LSTM(arg.embedding_size,
                            arg.lstm_dim,
                            arg.num_layers,
                            batch_first=True,
                            bidirectional=False)
        
        self.for_lstm2= nn.LSTM(arg.lstm_dim,
                            arg.lstm_dim,
                            arg.num_layers,
                            batch_first=True,
                            bidirectional=False)
        
        self.projection1 = nn.Linear(arg.embedding_size, arg.lstm_dim, bias=False)
        self.projection2 = nn.Linear(arg.lstm_dim, arg.vocab_size, bias=False)
        self.opt=optim.Adam(self.parameters(), lr=0.001)
        self.dropout1=nn.Dropout(p=0.2)
        self.dropout2=nn.Dropout(p=0.3)
        self.dropout3=nn.Dropout(p=0.4)
        self.start=START
        
        '''init'''
        init.xavier_uniform_(self.projection1.weight.data)
        init.xavier_uniform_(self.projection2.weight.data, gain=np.sqrt(6))
        for i in range(len(self.for_lstm2.all_weights)):
            for j in range(len(self.for_lstm2.all_weights[i])):
                try:
                    init.xavier_uniform_(self.for_lstm1.all_weights[i][j])
                    init.xavier_uniform_(self.for_lstm2.all_weights[i][j])
                except:
                    pass
                
        
    def forward(self, inputs ,train_embed):
        '''if fix the embed'''
        self.embeddings.weight.requires_grad = train_embed
        
        '''dropout embedding'''
        START=self.start*torch.ones(inputs.size(0),1,dtype=torch.long).cuda()
        inputs_embed=self.dropout1(self.embeddings(torch.cat((START,inputs),dim=1)))
        
        '''first layer forward'''
        forward1 =self.dropout2(self.for_lstm1(inputs_embed)[0]+self.projection1(inputs_embed))
        
        '''second layer lstm'''
        forward2 =self.dropout3(self.for_lstm2(forward1)[0]+forward1)
        
        output=self.projection2(forward2)
        return output
    
model=dec_elmo(arg).cuda()

def batch_train(inputs,input_mask,use_finetune,is_training=True):
    if is_training:
        model.opt.zero_grad()
    max_length=int(torch.max(torch.sum(input_mask,dim=1)))
    output=model(inputs[:,:max_length-1],use_finetune)
    inputs=inputs[:,:max_length].reshape(-1)
    input_mask=input_mask[:,:max_length].reshape(-1)
    
    loss=nn.CrossEntropyLoss(reduction='none')(output.reshape(-1,output.shape[2]),inputs)*input_mask
    
    loss=torch.sum(loss)/(torch.sum(input_mask)+1e-12)
    
    if is_training:
        loss.backward()
        model.opt.step()
    
    return loss

import gc
import time
batch_size=64
cnt=0
lr=1e-3
lastval=111
use_finetune=False
use_perplex=False
for ep in range(24):
    a=time.time()
    train_title_embed,title_mask,title_len_index=get_training_data(ep)
    model.train()
    ite=0
    while(ite<train_title_embed.shape[0]):
        gc.collect()
        dec_in=torch.tensor(train_title_embed[title_len_index[ite:ite+batch_size]],dtype=torch.long).cuda()
        dec_mask=torch.tensor(title_mask[title_len_index[ite:ite+batch_size]],dtype=torch.float32).cuda()
        ite+=batch_size
        
        loss=batch_train(dec_in,dec_mask,use_finetune=use_finetune)

            
        
        if (ite//batch_size)%200==0:
            cnt+=1  
            print (cnt,loss,np.sum(title_mask,axis=1)[title_len_index[ite]])
    print (time.time()-a)
    
    print ("valLoss")
    loss1=0
    loss2=0
    ite=0
    model.eval()
    with torch.no_grad():
        while(ite<4000):
            gc.collect()
            dec_in=torch.tensor(val_title_embed[ite:ite+batch_size],dtype=torch.long).cuda()
            dec_mask=torch.tensor((val_title_mask[ite:ite+batch_size]),dtype=torch.float32).cuda()

            ite+=batch_size
            loss1+=batch_train(dec_in,dec_mask,use_finetune=False,is_training=False)*(dec_in.shape[0])
    loss1=loss1/4000
    print (lastval,loss1)

    if lastval<loss1:
        lr*=0.5
        if lr<=5e-05:
            break
        model.opt = optim.Adam(lr=lr)
        
    else:
        lastval=loss1
    
    if lr<=0.001:
        use_finetune=True
        print (np.sum(np.array(model.embeddings.weight.data)-embed_matrix))
    print ()
    
torch.save(model.state_dict(), '../checkpoint/dec_elmo2.pkl')