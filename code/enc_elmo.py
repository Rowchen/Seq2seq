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
def get_mask(data):
    mask=np.ones(data.shape,dtype=np.int)
    mask[data==0]=0
    return mask

def sort_len(title_mask):
    title_sum=np.sum(title_mask,axis=1)
    ite=0
    idx=[]
    while ite<title_sum.shape[0]:
        idx.extend(list(ite+np.argsort(title_sum[ite:ite+10000])))
        ite+=10000
    return np.array(idx)

def get_training_data(cnt):
    train_doc_embed=np.load('../data/train_embed%d.npy'%(cnt%6))
    
    '''remove the title len less than 3'''
    doc_sum=np.sum((train_doc_embed!=0),axis=1)
    filter_=((doc_sum>2))
    train_doc_embed=train_doc_embed[filter_]
    '''get mask'''
    doc_mask=get_mask(train_doc_embed)
    '''resort'''
    doc_len_index=sort_len(doc_mask)
    print (train_doc_embed.shape)
    
    print ('loading training train_embed%d.npy'%(cnt%6))
    return train_doc_embed,doc_mask,doc_len_index

val_doc_embed=np.load('../data/train_embed6.npy')[:4000]
val_doc_mask=get_mask(val_doc_embed)



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


class enc_elmo(nn.Module):
    '''这里是参照bert的方法的'''
    def __init__(self, arg):
        super(enc_elmo, self).__init__()
        
        self.embeddings=nn.Embedding(arg.vocab_size, arg.embedding_size).cuda()
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embeddings.weight.requires_grad = False
        
        self.lstm1= nn.LSTM(arg.embedding_size,arg.lstm_dim//2,arg.num_layers,batch_first=True,bidirectional=True)
        self.lstm2= nn.LSTM(arg.lstm_dim,arg.lstm_dim//2,arg.num_layers,batch_first=True,bidirectional=True)
        
        self.projection1 = nn.Linear(arg.embedding_size, arg.lstm_dim, bias=False)
        self.projection2 = nn.Linear(arg.lstm_dim, arg.vocab_size, bias=False)
        self.opt=optim.Adam(self.parameters(), lr=0.001)
        
        self.start=START
        
        '''init'''
        init.xavier_uniform_(self.projection1.weight.data)
        init.xavier_uniform_(self.projection2.weight.data, gain=np.sqrt(6))
        for i in range(len(self.lstm1.all_weights)):
            for j in range(len(self.lstm1.all_weights[i])):
                try:
                    init.xavier_uniform_(self.lstm1.all_weights[i][j])
                    init.xavier_uniform_(self.lstm2.all_weights[i][j])
                except:
                    pass
            
    def forward(self, inputs ,sample_id):

        start=self.start*torch.ones(inputs.size(0),1,dtype=torch.long).cuda()
        padding=torch.zeros(inputs.size(0),1,dtype=torch.long).cuda()
        inputs=torch.cat((start,inputs,padding),dim=1)
        
        '''采用像bert一样的mask方法进行屏蔽'''
        r3=np.random.uniform(size=(sample_id.shape[0]))
        sample_unk=sample_id[r3<0.8]
        sample_random=sample_id[r3>=0.9]
        random_word=np.random.randint(0,50000,sample_random.shape[0])
        
        inputs[:,sample_unk]=UNK
        inputs[:,sample_random]=torch.LongTensor(random_word).cuda()
        
        inputs_embed=self.embeddings(inputs)
        
        del inputs
        
        '''first layer forward'''
        forward1 =self.lstm1(inputs_embed)[0]+self.projection1(inputs_embed)
        
        '''second layer lstm'''
        forward2 =(self.lstm2(forward1)[0]+forward1)
        
        output=self.projection2(forward2[:,sample_id,:])
        return output
    
model=enc_elmo(arg).cuda()

def batch_train(inputs,input_mask,use_finetune,is_training=True):
    if is_training:
        model.opt.zero_grad()
        
    
    max_length=int(torch.max(torch.sum(input_mask,dim=1)))
    
    r1=np.random.uniform(size=(max_length))
    r2=np.arange(max_length)
    sample_id=r2[r1<0.1].copy()

    output=model(inputs,sample_id+1)
    
    inputs=inputs[:,sample_id].reshape(-1)
    input_mask=input_mask[:,sample_id].reshape(-1)
    
    loss=nn.CrossEntropyLoss(reduction='none')(output.reshape(-1,output.shape[2]),inputs)*input_mask
    loss=torch.sum(loss)/(torch.sum(input_mask)+1e-12)
    
    if is_training:
        loss.backward()
        model.opt.step()
    
    return loss


import gc
import time
batch_size=16
cnt=0
lr=1e-3
lastval=347
use_finetune=False
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
            print (cnt,loss)
    print (time.time()-a)
    
    
    
    print ("valLoss")
    loss1=0
    loss2=0
    ite=0
    model.eval()
    with torch.no_grad():
        while(ite<4000):
            gc.collect()
            dec_in=torch.tensor(val_doc_embed[ite:ite+batch_size],dtype=torch.long).cuda()
            dec_mask=torch.tensor((val_doc_mask[ite:ite+batch_size]),dtype=torch.float32).cuda()

            ite+=batch_size
            loss1+=batch_train(dec_in,dec_mask,use_finetune=False,is_training=False)*(dec_in.shape[0])
    loss1=loss1/4000
    print (lastval,loss1)
    
    
    if lastval<loss1:
        lr*=0.5
        if lr<=5e-05:
            break
        model.opt = optim.Adam(model.parameters(),lr=lr)
        
    else:
        lastval=loss1
    
    if lr<=0.00025:
        use_finetune=True
        print (np.sum(np.array(model.embeddings.weight.data)-embed_matrix))
    print ()
    
torch.save(model.state_dict(), '../checkpoint/enc_elmo.pkl')