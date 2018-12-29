import numpy as np
import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from utils import *

'''这里记得修改'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PAD = 0
UNK = 1
START = 2
END = 3

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


val_doc_embed=np.load('../data/train_embed6.npy')[:5000]
val_title_embed=np.load('../data/title_embed6.npy')[:5000]

val_doc_mask=get_doc_mask(val_doc_embed)
val_title_mask=get_title_mask(val_title_embed)
val_title_embed=set_title_end(val_title_embed)


class seq2seq_param(object):
    vocab_size=78864
    embedding_size=300
    dec_embedding_size=512
    enc_max_len=1000   
    dec_max_len=20
    enc_hidden_size=512
    dec_hidden_size=512
    attention_size=300
    enc_num_layers=2
    dec_num_layers=1
    beam_width=4
    learning_rate=1e-3
    file_path='../checkpoint/dec_elmo2.pkl'
    elmo_lstm_dim=512
    elmo_num_layers=1
    need_pretrain=True
    enc_elmo_lstm_dim=512
    
    
arg=seq2seq_param()


class DecoderRNN(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN, self).__init__()
        self.dec_max_len=arg.dec_max_len
        self.dec_hidden_size=arg.dec_hidden_size
        self.attn_size=arg.attention_size
        self.output_size=arg.vocab_size
        self.num_layers=arg.dec_num_layers
        self.enc_hidden_size=arg.enc_hidden_size
        
        self.rnn = nn.LSTM(arg.dec_embedding_size,
                           arg.dec_hidden_size, 
                           arg.dec_num_layers,
                           batch_first=True)
        
        self.attention=QKV_attention(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        
        self.rnn2 = nn.LSTM(arg.attention_size,
                           arg.dec_hidden_size, 
                           arg.dec_num_layers,
                           batch_first=True)
        
        self.attention2=QKV_attention(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        
        
        self.dec_pos_embed=position_encoding_init(self.dec_max_len,arg.elmo_lstm_dim).repeat(1,1,1)
        
        self.inputlayer=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size+self.attn_size+self.enc_hidden_size, 
                                  arg.dec_embedding_size)
        self.outlayer =nn.Sequential(nn.Dropout(p=0.3)
            ,nn.Linear(self.attn_size, self.output_size)
        )
        
        self.W_copy = nn.Linear(arg.enc_hidden_size, arg.attention_size) # copy mode
        self.scale=nn.Parameter(torch.tensor(1.0))
        
        init.xavier_uniform_(self.outlayer[1].weight.data, gain=np.sqrt(6))
        init.xavier_uniform_(self.inputlayer.weight.data)
        init.xavier_uniform_(self.W_copy.weight.data)
        
        for i in range(len(self.rnn.all_weights)):
            for j in range(len(self.rnn.all_weights[i])):
                try:
                    init.xavier_uniform_(self.rnn.all_weights[i][j],nonlinearity='relu')
                except:
                    pass

        self.dec_elmo_embed=dec_elmo(arg).cuda()
        self.dropout=nn.Dropout(p=0.3,inplace=True)
        
    def forward(self,enc_in,enc_out,enc_mask,max_length=None,inputs=None,use_teacher_forcing=True,random_int=0):
        
        if max_length is None:
            max_length=self.dec_max_len
        batch_size=enc_out.size(0)
        enc_in2=torch.unsqueeze(enc_in,dim=1)
        
        enc_mask=torch.ByteTensor(enc_mask).cuda()
        decoder_output=torch.empty(batch_size,max_length,self.output_size).cuda()
        sequence_symbols=torch.empty(batch_size,max_length,dtype=torch.int32).cuda()
        dec_hidden1,dec_hidden2,elmo_hidden1,elmo_hidden2=None,None,None,None
        
        dec_symbol=START*torch.ones(enc_out.size(0),1,dtype=torch.long).cuda()
        dec_att_out=torch.zeros(batch_size,1,self.attn_size).cuda()
        select_read=torch.zeros(batch_size,1,self.enc_hidden_size).cuda()
        
        for i in range(max_length):
            '''第一步，将上次的attenton输出和这次的input拼接起来'''
            in_embed,elmo_hidden1,elmo_hidden2=self.dec_elmo_embed(dec_symbol,elmo_hidden1,elmo_hidden2)

            dec_in=self.dropout(self.inputlayer(torch.cat((in_embed,dec_att_out,select_read),dim=2))\
                                +self.dec_pos_embed[:,i:i+1,:])
            '''2层attention 经过rnn后得到输出'''
            dec_out, dec_hidden1 = self.rnn(dec_in, dec_hidden1)
            dec_att_out=self.attention(dec_out,enc_out,enc_mask)
            
            dec_out, dec_hidden2 = self.rnn2(dec_att_out, dec_hidden2)
            dec_att_out=self.attention2(dec_out,enc_out,enc_mask)+dec_att_out
            
            '''copyscore'''
            score_c=torch.bmm(dec_att_out,torch.transpose(torch.tanh(self.W_copy(enc_out)),1,2))
            score_c.data.masked_fill_(enc_mask, -float('inf'))
            score_c=F.softmax(score_c,dim=-1)
            score_e=score_c*self.scale*self.scale

            '''经过vocab层映射得到下一个输出'''
            dec_to_vocab=self.outlayer(dec_att_out)
            dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=score_e)
            decoder_output[:,i:i+1,:]=dec_to_vocab
            
            if use_teacher_forcing:
                '''random choose a word'''
                '''在训练的时候,我会逐渐增大这个值.一般训练18个ep,这个值封顶36%的概率,因此randint每次增长2'''
                force_proba=np.random.randint(0,100)
                if force_proba>=random_int:    #如果随机到大于这个数的数,就用force,在一开始肯定能大于,必定force
                    dec_symbol=inputs[:,i:i+1]
                    sequence_symbols[:,i:i+1]=torch.argmax(dec_to_vocab,dim=2)
                else:
                    dec_symbol=torch.argmax(dec_to_vocab,dim=2)
                    sequence_symbols[:,i:i+1]=dec_symbol
            else:
                dec_symbol=torch.argmax(dec_to_vocab,dim=2)
                sequence_symbols[:,i:i+1]=dec_symbol

            score_f=score_c*((enc_in==dec_symbol).float().unsqueeze(dim=1))
            score_f=score_f/(torch.sum(score_f,dim=-1,keepdim=True)+1e-8)
            select_read=torch.bmm(score_f,enc_out)
            
                
        return decoder_output,sequence_symbols
    
class Beam_search_decoder(DecoderRNN):
    '''TODO beam search还可以修改。应该改成像veterbi算法那样。当前结点选beamwidth个不一样的进入下一步才对，如果当前结点是一样的
        那后面的无论如何都不如第一高，这个可行性就很低了。
    '''
    def __init__(self,arg):
        super(Beam_search_decoder, self).__init__(arg)
    def beam_search(self,enc_in,enc_out,enc_mask,beam_width=5):
        '''每次只接受一条输入，反正是解码，可以慢一点'''
        max_length=self.dec_max_len
        batch_size=enc_out.size(0)
        assert(batch_size==1)
        
        '''一共会有beam_width个输出'''
        dec_hidden1,dec_hidden2,elmo_hidden1,elmo_hidden2=None,None,None,None
        dec_symbol=START*torch.ones(beam_width,1,dtype=torch.long).cuda()
        dec_att_out=torch.zeros(beam_width,1,self.attn_size).cuda()
        select_read=torch.zeros(beam_width,1,self.enc_hidden_size).cuda()
        
        '''将enc的输入复制beam_width份'''
        enc_out=enc_out.repeat(beam_width,1,1)
        enc_mask=torch.ByteTensor(np.tile(enc_mask,[beam_width,1,1])).cuda()
        enc_in=enc_in.repeat(beam_width,1)
        enc_in2=torch.unsqueeze(enc_in,dim=1)

        beam_proba=torch.zeros(beam_width,1).cuda()
        sequence_symbols=[]
        length=0
        
        for i in range(max_length):
            in_embed,elmo_hidden1,elmo_hidden2=self.dec_elmo_embed(dec_symbol,elmo_hidden1,elmo_hidden2)

            dec_in=self.dropout(self.inputlayer(torch.cat((in_embed,dec_att_out,select_read),dim=2))\
                                +self.dec_pos_embed[:,i:i+1,:])
            '''2ceng 经过rnn后得到输出'''
            dec_out, dec_hidden1 = self.rnn(dec_in, dec_hidden1)
            dec_att_out=self.attention(dec_out,enc_out,enc_mask)
            dec_out, dec_hidden2 = self.rnn2(dec_att_out, dec_hidden2)
            dec_att_out=self.attention2(dec_out,enc_out,enc_mask)+dec_att_out
            
            '''copyscore'''
            score_c=torch.bmm(dec_att_out,torch.transpose(torch.tanh(self.W_copy(enc_out)),1,2))
            score_c.data.masked_fill_(enc_mask, -float('inf'))
            score_c=F.softmax(score_c,dim=-1)
            score_e=score_c*self.scale*self.scale

            '''经过vocab层映射得到下一个输出'''
            dec_to_vocab=self.outlayer(dec_att_out)
            dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=score_e)
           
            '''找到最大的proba'''
            proba=F.log_softmax(dec_to_vocab,dim=2).squeeze()+beam_proba
            
            if i==0:
                select=torch.topk(proba[0],beam_width)[1]
                dec_symbol=select.reshape(beam_width,1)
                beam_proba=proba[0,select].reshape(beam_width,1)
                sequence_symbols.append(dec_symbol)
                choose=select//self.output_size
            else:
                if i<=0:
                    maxproba=torch.max(proba,dim=0)
                    proba2=maxproba[0]
                    index=maxproba[1]

                    select=torch.topk(proba2,beam_width)[1]
                    choose=index[select]

                    beam_proba=proba2[select].reshape(beam_width,1)
                    dec_symbol=select.reshape(beam_width,1)
                    
                else:
                    proba=proba.reshape(-1)
                    select=torch.topk(proba,beam_width)[1]
                    choose=select//self.output_size

                    beam_proba=proba[select].reshape(beam_width,1)
                    select=select%self.output_size#第几个token
                    dec_symbol=select.reshape(beam_width,1)
                
                '''注意！！！这里symbol要重新安排！！！'''
                ls=torch.cat((sequence_symbols[-1][choose,:],dec_symbol),dim=1)
                sequence_symbols.append(ls)
                
                if dec_symbol[0,0]==END:
                    break
                  
            '''TODO   这一步需要认真思考！！！应该要修改,因为score_f跟上一个时刻的序列有很大的关系！！！！！'''
            score_f=score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
            score_f=score_f/(torch.sum(score_f,dim=-1,keepdim=True)+1e-8)
            select_read=torch.bmm(score_f,enc_out)
            
            length=i+1
            
            elmo_hidden1=(elmo_hidden1[0][:,choose,:],elmo_hidden1[1][:,choose,:])
            elmo_hidden2=(elmo_hidden2[0][:,choose,:],elmo_hidden2[1][:,choose,:])
            dec_hidden1=(dec_hidden1[0][:,choose,:],dec_hidden1[1][:,choose,:])
            dec_hidden2=(dec_hidden2[0][:,choose,:],dec_hidden2[1][:,choose,:])
            dec_att_out=dec_att_out[choose,:,:]
            
        return sequence_symbols[-1],beam_proba[0]/length
    
class seq2seq(nn.Module):
    def __init__(self,arg):
        super(seq2seq, self).__init__()
        self.encoder=EncoderRNN2(arg)
        
        self.decoder=Beam_search_decoder(arg)
        #self.decoder=DecoderRNN(arg)
        
        self.enc_opt=optim.Adam(self.encoder.parameters(), lr=arg.learning_rate)
        self.dec_opt=optim.Adam(self.decoder.parameters(), lr=arg.learning_rate)
        self.criterion=nn.CrossEntropyLoss(reduction='none')
        
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        
    def batch_train(self,enc_in,dec_in,dec_mask,enc_mask,TFIN,use_teacher_forcing=True,is_training=True,random_int=0):
        if is_training:
            self.enc_opt.zero_grad()
            self.dec_opt.zero_grad()
        
        enc_out=self.encoder(enc_in,TFIN)
        max_length=int(torch.max(torch.sum(dec_mask,dim=1)))
        dec_out,dec_symbols=self.decoder(enc_in,enc_out,enc_mask,
                                         max_length=max_length,inputs=dec_in,
                                         use_teacher_forcing=use_teacher_forcing,random_int=random_int)
        
        dec_mask=dec_mask[:,:max_length].reshape(-1)
        dec_label=dec_in[:,:max_length].reshape(-1)
        
        
        
        loss=self.criterion(dec_out.reshape(-1,dec_out.shape[2]),dec_label)*dec_mask
        loss=torch.sum(loss)/(torch.sum(dec_mask)+1e-12)
        
        
        if is_training:
            loss.backward()
            self.enc_opt.step()
            self.dec_opt.step()
            return loss
        else:
            return loss,dec_symbols
        
    def beam_batch_predict(self,enc_in,enc_mask,TFIN,beam_width=5):
        enc_out=self.encoder(enc_in,TFIN)
        dec_symbols,probs=self.decoder.beam_search(enc_in,enc_out,enc_mask,beam_width)
        return dec_symbols,probs

model=seq2seq(arg).cuda()

import gc
import time
'''batch-size根据显存调整，记得观察2分钟左右，因为预测长度是从2到20增长，所以显存也要经过2分钟的不断增长，直到平稳。'''
batch_size=40
cnt=0
lr=1e-3
randint=0
for ep in range(18):
    randint=ep*2
    a=time.time()
    model.train()
    train_doc_embed,train_title_embed,doc_mask,title_mask,title_len_index=get_training_data(ep)
    ite=0
    while(ite<train_doc_embed.shape[0]):
        gc.collect()
        
        enc_in=torch.tensor(train_doc_embed[title_len_index[ite:ite+batch_size]],dtype=torch.long).cuda()
        dec_in=torch.tensor(train_title_embed[title_len_index[ite:ite+batch_size]],dtype=torch.long).cuda()
        dec_mask=torch.tensor(title_mask[title_len_index[ite:ite+batch_size]],dtype=torch.float32).cuda()
        enc_mask=doc_mask[title_len_index[ite:ite+batch_size]]
        TFIN=torch.LongTensor(getTF(train_doc_embed[title_len_index[ite:ite+batch_size]])).cuda()
        ite+=batch_size
            
        loss=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,TFIN,use_teacher_forcing=True,random_int=randint)
        if (ite//batch_size)%200==0:
            cnt+=1  
            print (cnt,loss)
    print (time.time()-a)
    
    model.eval()
    print ("valLoss")
    loss1=0
    loss2=0
    ite=0
   
    with torch.no_grad():
        while(ite<5000):
            gc.collect()
            enc_in=torch.tensor(val_doc_embed[ite:ite+batch_size],dtype=torch.long).cuda()
            dec_in=torch.tensor(val_title_embed[ite:ite+batch_size],dtype=torch.long).cuda()
            dec_mask=torch.tensor((val_title_mask[ite:ite+batch_size]),dtype=torch.float32).cuda()
            enc_mask=val_doc_mask[ite:ite+batch_size]
            TFIN=torch.LongTensor(getTF(val_doc_embed[ite:ite+batch_size])).cuda()
            
            ite+=batch_size
            loss1+=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,TFIN,is_training=False)[0]*(enc_in.shape[0])
            loss2+=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,TFIN,is_training=False,
                                     use_teacher_forcing=False)[0]*(enc_in.shape[0])
    print (loss1/5000,loss2/5000)
    
    torch.save(model.state_dict(), '../checkpoint/attentionnorm_2.pkl')
    
    lr*=0.87
    model.enc_opt=optim.Adam(model.encoder.parameters(), lr=lr)
    model.dec_opt=optim.Adam(model.decoder.parameters(), lr=lr)
    

    
    
    
