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

'''
attention有两种方法：
1、rnn先输入input和hidden，得到output，然后output再和encode-hidden进行attention
2、input先和encode-hidden进行attention，得到input2之后，再输入rnn中。
这里我们实现的是第一种
'''    
class QKV_attention2(nn.Module):
    '''这里用banhandu  attention '''
    def __init__(self, attn_dim, dec_dim, enc_dim):
        super(QKV_attention2, self).__init__()
        
        self.W_dec=nn.Linear(dec_dim,dec_dim,bias=False)
        self.W_enc=nn.Linear(enc_dim,enc_dim,bias=False)
        self.W_his=nn.Linear(1,enc_dim,bias=False)
        self.VT=nn.Linear(enc_dim,1,bias=False)
        self.Bias=nn.Parameter(torch.zeros(enc_dim))
        
        init.xavier_uniform_(self.W_dec.weight.data)
        init.xavier_uniform_(self.W_enc.weight.data)
        init.xavier_uniform_(self.W_his.weight.data)
        init.xavier_uniform_(self.VT.weight.data)
        
    def forward(self, dec_out, enc_out, enc_mask,history):
        '''此处复现的是论文:go to the point ,使用了coverage'''
        '''history:[batch,1,seq_len]'''
        Q=self.W_dec(dec_out)
        K=self.W_enc(enc_out)
        HIS=self.W_his(history.transpose(1,2))
        ALL=torch.tanh(Q+K+HIS+self.Bias)
        attn=self.VT(ALL).transpose(1,2)

        attn.data.masked_fill_(enc_mask, -float('inf'))
        attn=F.softmax(attn,dim=2)
        return attn



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
        for i in range(len(self.rnn.all_weights)):
            for j in range(len(self.rnn.all_weights[i])):
                try:
                    init.xavier_uniform_(self.rnn.all_weights[i][j],nonlinearity='relu')
                except:
                    pass
        self.attention=QKV_attention2(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        self.dec_pos_embed=position_encoding_init(self.dec_max_len,arg.elmo_lstm_dim).repeat(1,1,1)
        self.inputlayer=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size+self.attn_size+self.enc_hidden_size, 
                                  arg.dec_embedding_size)
        self.outlayer =nn.Sequential(nn.Dropout(p=0.3),nn.Linear(self.attn_size, self.output_size))
        self.atten_layer=nn.Linear(arg.enc_hidden_size+arg.dec_hidden_size,arg.attention_size)
        init.xavier_uniform_(self.outlayer[1].weight.data, gain=np.sqrt(6))
        init.xavier_uniform_(self.inputlayer.weight.data)
        init.xavier_uniform_(self.atten_layer.weight.data)
        
        self.dec_elmo_embed=dec_elmo(arg)
        self.dropout=nn.Dropout(p=0.3,inplace=True)
        
        self.copy_attn=nn.Linear(arg.attention_size,1,bias=False)
        self.copy_input=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size,1,bias=False)
        init.xavier_uniform_(self.copy_attn.weight.data)
        init.xavier_uniform_(self.copy_input.weight.data)
        
        self.scale=nn.Parameter(torch.tensor(1.0))
        
    def forward(self,enc_in,enc_out,enc_mask,max_length=None,inputs=None,use_teacher_forcing=True):
        
        if max_length is None:
            max_length=self.dec_max_len
        batch_size=enc_out.size(0)
        enc_in2=torch.unsqueeze(enc_in,dim=1)
        
        enc_mask=torch.ByteTensor(enc_mask).cuda()
        decoder_output=torch.empty(batch_size,max_length,self.output_size).cuda()
        sequence_symbols=torch.empty(batch_size,max_length,dtype=torch.int32).cuda()
        dec_hidden,elmo_hidden1,elmo_hidden2=None,None,None
        
        seq_len=enc_out.size(1)
        
        dec_symbol=START*torch.ones(enc_out.size(0),1,dtype=torch.long).cuda()
        dec_att_out=torch.zeros(batch_size,1,self.attn_size).cuda()
        select_read=torch.zeros(batch_size,1,self.enc_hidden_size).cuda()
        
        atten_score_his=torch.ones(batch_size,1,seq_len).cuda()/seq_len#cumulate attention score
        
        covloss=0
        for i in range(max_length):
            '''第一步，将上次的attenton输出和这次的input拼接起来'''
            in_embed,elmo_hidden1,elmo_hidden2=self.dec_elmo_embed(dec_symbol,elmo_hidden1,elmo_hidden2)

            dec_in=self.dropout(self.inputlayer(torch.cat((in_embed,dec_att_out,select_read),dim=2))\
                                +self.dec_pos_embed[:,i:i+1,:])
            '''经过rnn后得到输出'''
            dec_out, dec_hidden= self.rnn(dec_in, dec_hidden)
            atten_score=self.attention(dec_out,enc_out,enc_mask,atten_score_his)

            '''attention'''
            context=torch.bmm(atten_score, enc_out)
            dec_att_out=self.atten_layer(torch.cat((context,dec_out),dim=2))
            
            '''算出copy的概率与生成的概率,根据coverage的论文,这个概率应该与dec_att_out以及in_embed有关系'''
            proba_copy=(self.copy_attn(dec_att_out)+self.copy_input(in_embed)+self.scale)
            
            score_e=atten_score*(proba_copy*proba_copy)
            dec_to_vocab=self.outlayer(dec_att_out)
            dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=score_e)
            decoder_output[:,i:i+1,:]=dec_to_vocab
            
            '''covloss'''
            atten_both=torch.cat((atten_score,atten_score_his),dim=1)
            covloss+=torch.sum(torch.min(atten_both,dim=1)[0])
            atten_score_his=(atten_score+atten_score_his)/2#history

            if use_teacher_forcing:
                dec_symbol=inputs[:,i:i+1]
                sequence_symbols[:,i:i+1]=torch.argmax(dec_to_vocab,dim=2)
            else:
                dec_symbol=torch.argmax(dec_to_vocab,dim=2)
                sequence_symbols[:,i:i+1]=dec_symbol

            score_f=atten_score*((enc_in==dec_symbol).float().unsqueeze(dim=1))
            select_read=torch.bmm(score_f,enc_out)
                
        return decoder_output,sequence_symbols,covloss
    
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
        seq_len=enc_out.size(1)
        assert(batch_size==1)
        
        '''一共会有beam_width个输出'''
        dec_hidden,elmo_hidden1,elmo_hidden2=None,None,None
        dec_symbol=START*torch.ones(beam_width,1,dtype=torch.long).cuda()
        dec_att_out=torch.zeros(beam_width,1,self.attn_size).cuda()
        select_read=torch.zeros(beam_width,1,self.enc_hidden_size).cuda()
        atten_score_his=torch.ones(batch_size,1,seq_len).cuda()/seq_len#cumulate attention score
        
        '''将enc的输入复制beam_width份'''
        enc_out=enc_out.repeat(beam_width,1,1)
        enc_mask=torch.ByteTensor(np.tile(enc_mask,[beam_width,1,1])).cuda()
        enc_in=enc_in.repeat(beam_width,1)
        enc_in2=torch.unsqueeze(enc_in,dim=1)
        
        beam_proba=torch.zeros(beam_width,1).cuda()
        sequence_symbols=[]
        length=0
        
        
        for i in range(max_length):
            '''第一步，将上次的attenton输出和这次的input拼接起来'''
            in_embed,elmo_hidden1,elmo_hidden2=self.dec_elmo_embed(dec_symbol,elmo_hidden1,elmo_hidden2)

            dec_in=self.dropout(self.inputlayer(torch.cat((in_embed,dec_att_out,select_read),dim=2))\
                                +self.dec_pos_embed[:,i:i+1,:])
            
            '''经过rnn后得到输出'''
            dec_out, dec_hidden= self.rnn(dec_in, dec_hidden)
            atten_score=self.attention(dec_out,enc_out,enc_mask,atten_score_his)

            '''attention'''
            context=torch.bmm(atten_score, enc_out)
            dec_att_out=self.atten_layer(torch.cat((context,dec_out),dim=2))
            
            '''算出copy的概率与生成的概率,根据coverage的论文,这个概率应该与dec_att_out以及in_embed有关系'''
            proba_copy=(self.copy_attn(dec_att_out)+self.copy_input(in_embed)+self.scale)
            
            score_e=atten_score*(proba_copy*proba_copy)
            dec_to_vocab=self.outlayer(dec_att_out)
            dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=score_e)
            
            atten_score_his=(atten_score+atten_score_his)/2#history
            
           
            '''找到最大的proba'''
            proba=F.log_softmax(dec_to_vocab,dim=2).squeeze()+beam_proba
            
            if i==0:
                select=torch.topk(proba[0],beam_width)[1]
                dec_symbol=select.reshape(beam_width,1)
                beam_proba=proba[0,select].reshape(beam_width,1)
                sequence_symbols.append(dec_symbol)
                choose=select//self.output_size
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
            score_f=atten_score[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
            select_read=torch.bmm(score_f,enc_out)
            
            length=i+1
            elmo_hidden1=(elmo_hidden1[0][:,choose,:],elmo_hidden1[1][:,choose,:])
            elmo_hidden2=(elmo_hidden2[0][:,choose,:],elmo_hidden2[1][:,choose,:])
            dec_hidden=(dec_hidden[0][:,choose,:],dec_hidden[1][:,choose,:])
            dec_att_out=dec_att_out[choose,:,:]
            atten_score_his=atten_score_his[choose,:,:]
        
            
        return sequence_symbols[-1],beam_proba[0]/length
    
class seq2seq(nn.Module):
    def __init__(self,arg):
        super(seq2seq, self).__init__()
        self.encoder=EncoderRNN1(arg)
        
        self.decoder=Beam_search_decoder(arg)
        #self.decoder=DecoderRNN(arg)
        
        self.enc_opt=optim.Adam(self.encoder.parameters(), lr=arg.learning_rate)
        self.dec_opt=optim.Adam(self.decoder.parameters(), lr=arg.learning_rate)
        self.criterion=nn.CrossEntropyLoss(reduction='none')
        
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        
    def batch_train(self,enc_in,dec_in,dec_mask,enc_mask,use_teacher_forcing=True,
                    is_training=True,with_coverage=False):
        if is_training:
            self.enc_opt.zero_grad()
            self.dec_opt.zero_grad()
        
        enc_out=self.encoder(enc_in)
        max_length=int(torch.max(torch.sum(dec_mask,dim=1)))
        dec_out,dec_symbols,covloss=self.decoder(enc_in,enc_out,enc_mask,
                                         max_length=max_length,inputs=dec_in,
                                         use_teacher_forcing=use_teacher_forcing)
        
        dec_mask=dec_mask[:,:max_length].reshape(-1)
        dec_label=dec_in[:,:max_length].reshape(-1)
        
        loss=self.criterion(dec_out.reshape(-1,dec_out.shape[2]),dec_label)*dec_mask
        dec_mask_sum=(torch.sum(dec_mask)+1e-12)
        loss=torch.sum(loss)/dec_mask_sum
        covloss=covloss/dec_mask_sum
        
        if with_coverage:
            all_loss=0.5*covloss+loss
        else:
            all_loss=loss

        if is_training:
            all_loss.backward()
            self.enc_opt.step()
            self.dec_opt.step()
            
        return loss,covloss

    
    def beam_batch_predict(self,enc_in,enc_mask,beam_width=5):
        enc_out=self.encoder(enc_in)
        dec_symbols,probs=self.decoder.beam_search(enc_in,enc_out,enc_mask,beam_width)
        return dec_symbols,probs
    
model=seq2seq(arg).cuda()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

import gc
import time
batch_size=40
cnt=0
lr=1e-3
for ep in range(18):
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
        ite+=batch_size
        if ep<12:
            loss=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,use_teacher_forcing=True,with_coverage=False)
        else:
            loss=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,use_teacher_forcing=True,with_coverage=True)
            
        if (ite//batch_size)%200==0:
            cnt+=1  
            print (cnt,loss[0].item(),loss[1].item())
    print (time.time()-a)
    
    model.eval()
    print ("valLoss")
    loss1,loss2,loss3=0,0,0
    ite=0
    with torch.no_grad():
        while(ite<5000):
            gc.collect()
            enc_in=torch.tensor(val_doc_embed[ite:ite+batch_size],dtype=torch.long).cuda()
            dec_in=torch.tensor(val_title_embed[ite:ite+batch_size],dtype=torch.long).cuda()
            dec_mask=torch.tensor((val_title_mask[ite:ite+batch_size]),dtype=torch.float32).cuda()
            enc_mask=val_doc_mask[ite:ite+batch_size]

            ite+=batch_size
            loss=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,is_training=False)
            loss1+=loss[0]*(enc_in.shape[0])
            loss2+=loss[1]*(enc_in.shape[0])
            loss3+=model.batch_train(enc_in,dec_in,dec_mask,enc_mask,is_training=False,
                                     use_teacher_forcing=False)[0]*(enc_in.shape[0])
            
    print (loss1/5000,loss2/5000,loss3/5000)
    
    torch.save(model.state_dict(), '../checkpoint/enc_elmo_copy_coverage.pkl')
    lr*=0.87
    model.enc_opt=optim.Adam(model.encoder.parameters(), lr=lr)
    model.dec_opt=optim.Adam(model.decoder.parameters(), lr=lr)
    
    
    
