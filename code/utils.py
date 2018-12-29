import numpy as np
import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
import allennlp
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder

PAD = 0
UNK = 1
START = 2
END = 3

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor).cuda()


'''train'''
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

def get_doc_mask(data):
    '''doc mask'''
    mask=np.zeros(data.shape,dtype=np.int)
    mask[data==0]=1
    return np.expand_dims(mask,1)

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
    print ('loading training train_embed%d.npy'%(cnt%6))
    train_doc_embed=np.load('../data/train_embed%d.npy'%(cnt%6))
    train_title_embed=np.load('../data/title_embed%d.npy'%(cnt%6))
    
    if cnt%6==5:
        train_doc_embed2=np.load('../data/train_embed6.npy')[5000:]
        train_title_embed2=np.load('../data/title_embed6.npy')[5000:]
        
        train_doc_embed=np.concatenate((train_doc_embed,train_doc_embed2),axis=0)
        train_title_embed=np.concatenate((train_title_embed,train_title_embed2),axis=0)
        
    '''remove the title len less than 3'''
    id_sum=np.sum((train_doc_embed!=0),axis=1)
    tit_sum=np.sum((train_title_embed!=0),axis=1)
    filter_=((id_sum!=0)&(tit_sum>2))
    
    train_doc_embed=train_doc_embed[filter_]
    train_title_embed=train_title_embed[filter_]
    
    title_mask=get_title_mask(train_title_embed)
    doc_mask=get_doc_mask(train_doc_embed)
    train_title_embed=set_title_end(train_title_embed)
    print (train_doc_embed.shape,train_title_embed.shape)
    
    title_len_index=sort_title_len(title_mask)
    return train_doc_embed,train_title_embed,doc_mask,title_mask,title_len_index


def getTF(data): 
    TF=np.zeros((data.shape[0],1000),dtype=np.int32)
    for row in range(data.shape[0]):
        w_cnt={}
        for i in range(1000):
            w_cnt[data[row,i]]=w_cnt.get(data[row,i],0)+1
        w_cnt[0]=0
        for i in range(1000):
            TF[row,i]=w_cnt[data[row,i]]//4
    TF[TF>=50]=49
    return TF


class enc_elmo(nn.Module):
    '''这里是参照bert的方法的'''
    def __init__(self, arg):
        super(enc_elmo, self).__init__()
        self.embeddings=nn.Embedding(arg.vocab_size, arg.embedding_size).cuda()
        
        self.lstm1= nn.LSTM(arg.embedding_size,arg.enc_elmo_lstm_dim//2,
                            1,batch_first=True,bidirectional=True)
        
        self.lstm2= nn.LSTM(arg.enc_elmo_lstm_dim,arg.enc_elmo_lstm_dim//2,
                            1,batch_first=True,bidirectional=True)
        self.projection1 = nn.Linear(arg.embedding_size, arg.enc_elmo_lstm_dim, bias=False)
        
        
        '''init'''
        if arg.need_pretrain:
            predtrained_dec_embedding=torch.load('../checkpoint/enc_elmo.pkl')
            pretrain_dict={k: v for k, v in predtrained_dec_embedding.items() if k in self.state_dict()}
            self.load_state_dict(pretrain_dict)
            self.lstm1.flatten_parameters()
            self.lstm2.flatten_parameters()
            del predtrained_dec_embedding
            torch.cuda.empty_cache()
            
        self.enc_elmo_s=nn.Parameter(torch.ones(3).cuda())
        self.enc_elmo_r=nn.Parameter(torch.ones(1).cuda())
        
        self.embeddings.weight.requires_grad = False
        self.lstm1.requires_grad=False
        self.lstm2.requires_grad=False
        self.projection1.requires_grad=False

                

    def forward(self, inputs):
        
        start=START*torch.ones(inputs.size(0),1,dtype=torch.long).cuda()
        padding=torch.zeros(inputs.size(0),1,dtype=torch.long).cuda()
        inputs=torch.cat((start,inputs,padding),dim=1)
        inputs_embed=self.embeddings(inputs)
        
        proj_embed=self.projection1(inputs_embed)
        '''first layer forward'''
        forward1 =self.lstm1(inputs_embed)[0]
        '''second layer lstm'''
        forward2 =self.lstm2(forward1+proj_embed)[0]
        '''weight'''
        s=F.softmax(self.enc_elmo_s,dim=-1)
        
        ELMO=self.enc_elmo_r*(proj_embed*s[0]+forward1*s[1]+forward2*s[2])
        
        output=torch.cat((inputs_embed,ELMO),dim=-1)[:,1:-1,:]
        
        return output

class EncoderRNN1(nn.Module):
    def __init__(self, arg):
        super(EncoderRNN1, self).__init__()
        
        self.elmo_embed=enc_elmo(arg)
        
        self.rnn = nn.LSTM(arg.enc_elmo_lstm_dim+arg.embedding_size,
                          arg.enc_hidden_size//2,
                          arg.enc_num_layers,
                          batch_first=True,
                          bidirectional=True)
        
        for i in range(len(self.rnn.all_weights)):
            for j in range(len(self.rnn.all_weights[i])):
                try:
                    init.xavier_uniform_(self.rnn.all_weights[i][j])
                except:
                    pass
        self.dropout=nn.Dropout(p=0.3,inplace=True)
        
    def forward(self, inputs):
        inputs_embed=self.elmo_embed(inputs)
        output,_ = self.rnn(self.dropout(inputs_embed))
        return output

class EncoderRNN2(nn.Module):
    def __init__(self, arg):
        super(EncoderRNN2, self).__init__()
        self.IDFembeddings=nn.Embedding(arg.vocab_size, 200).cuda()
        self.IDFembeddings.weight.data.copy_(torch.from_numpy(np.load('../data/IDFweight.npy')))
        self.IDFembeddings.weight.data.requires_grad=False

        self.TFembeddings=nn.Embedding(50, 50).cuda()
        self.TFembeddings.weight.data.copy_(torch.from_numpy(np.eye(50)))
        self.TFembeddings.weight.data.requires_grad=False
        
        self.elmo_embed=enc_elmo(arg)
        
        self.rnn = nn.LSTM(arg.enc_elmo_lstm_dim+arg.embedding_size+200+50,
                          arg.enc_hidden_size//2,
                          arg.enc_num_layers,
                          batch_first=True,
                          bidirectional=True)
        
        for i in range(len(self.rnn.all_weights)):
            for j in range(len(self.rnn.all_weights[i])):
                try:
                    init.xavier_uniform_(self.rnn.all_weights[i][j])
                except:
                    pass
        self.dropout=nn.Dropout(p=0.3,inplace=True)
        
    def forward(self, inputs,TFin):
        IDF=self.IDFembeddings(inputs)
        inputs_embed=self.elmo_embed(inputs)
        TFembed=self.TFembeddings(TFin)
        output,_ = self.rnn(self.dropout(torch.cat((inputs_embed,IDF,TFembed),dim=2)))
        return output    
    

class QKV_attention(nn.Module):
    '''根据上一时刻的输出st-1与encoder的隐状态，得到这时刻的context'''
    def __init__(self, attn_dim, dec_dim, enc_dim):
        super(QKV_attention, self).__init__()
        self.atten_layer=nn.Linear(enc_dim+dec_dim,attn_dim,bias=False)
        init.xavier_uniform_(self.atten_layer.weight.data, gain=np.sqrt(6))
        
    def forward(self, dec_out, enc_out, enc_mask):
        attn=torch.bmm(dec_out, enc_out.transpose(1, 2))
        attn.data.masked_fill_(enc_mask, -float('inf'))
        attn=F.softmax(attn,dim=2)
        context=torch.bmm(attn, enc_out)
        output=self.atten_layer(torch.cat((context,dec_out),dim=2))
        return output
    
class EncoderRNN3(nn.Module):
    def __init__(self, arg):
        super(EncoderRNN3, self).__init__()
        self.IDFembeddings=nn.Embedding(arg.vocab_size, 200).cuda()
        self.IDFembeddings.weight.data.copy_(torch.from_numpy(np.load('../data/IDFweight.npy')))
        self.IDFembeddings.weight.data.requires_grad=False
        
        self.gate_state = nn.Linear(256, 512)
        self.to_gate = to_gate = nn.Linear(1024, 512)
        self.gate_output =  nn.Linear(512, 512)
        self.cnnencoder = CnnEncoder(512,64,output_dim=512).cuda()
        
        self.TFembeddings=nn.Embedding(50, 50).cuda()
        self.TFembeddings.weight.data.copy_(torch.from_numpy(np.eye(50)))
        self.TFembeddings.weight.data.requires_grad=False
        
        self.elmo_embed=enc_elmo(arg)
        
        self.rnn = nn.LSTM(arg.enc_elmo_lstm_dim+arg.embedding_size+200+50,
                          arg.enc_hidden_size//2,
                          arg.enc_num_layers,
                          batch_first=True,
                          bidirectional=True)
        
        for i in range(len(self.rnn.all_weights)):
            for j in range(len(self.rnn.all_weights[i])):
                try:
                    init.xavier_uniform_(self.rnn.all_weights[i][j])
                except:
                    pass
        self.dropout=nn.Dropout(p=0.3,inplace=True)
        
    def forward(self, inputs,TFin):
        IDF=self.IDFembeddings(inputs)
        inputs_embed=self.elmo_embed(inputs)
        TFembed=self.TFembeddings(TFin)
        output, hidden = self.rnn(self.dropout(torch.cat((inputs_embed,IDF,TFembed),dim=2)))

        select_output = self.gate_output(output)
        select_state = self.cnnencoder(output,mask=None)
        
        select_mask = torch.sigmoid(self.to_gate(torch.cat([select_state.unsqueeze(1).expand_as(select_output), select_output],-1)))
        output = output * select_mask
        return output
    
    
class dec_elmo(nn.Module):
    def __init__(self, arg):
        super(dec_elmo, self).__init__()
        
        self.embeddings=nn.Embedding(arg.vocab_size, arg.embedding_size).cuda()
    
        self.for_lstm1= nn.LSTM(arg.embedding_size,
                            arg.elmo_lstm_dim,
                            arg.elmo_num_layers,
                            batch_first=True,
                            bidirectional=False)
       
        self.for_lstm2= nn.LSTM(arg.elmo_lstm_dim,
                            arg.elmo_lstm_dim,
                            arg.elmo_num_layers,
                            batch_first=True,
                            bidirectional=False)
        
        self.projection1 = nn.Linear(arg.embedding_size, arg.elmo_lstm_dim, bias=False)
        
        if arg.need_pretrain:
            predtrained_dec_embedding=torch.load(arg.file_path)
            pretrain_dict={k: v for k, v in predtrained_dec_embedding.items() if k in self.state_dict()}
            self.load_state_dict(pretrain_dict)
            self.for_lstm1.flatten_parameters()
            self.for_lstm2.flatten_parameters()
            del predtrained_dec_embedding
            torch.cuda.empty_cache()
            
        self.dec_elmo_s=nn.Parameter(torch.ones(3).cuda())
        self.dec_elmo_r=nn.Parameter(torch.ones(1).cuda())
        self.embeddings.weight.requires_grad = False
        self.for_lstm1.requires_grad=False
        self.for_lstm2.requires_grad=False
        self.projection1.requires_grad=False
        
    def forward(self,inputs,hidden1=None,hidden2=None):
        '''dropout embedding'''
        inputs_embed=self.embeddings(inputs)
        proj_embed=self.projection1(inputs_embed)
        
        '''first layer forward'''
        forward1,hidden1=self.for_lstm1(inputs_embed,hidden1)
        
        '''second layer lstm'''
        forward2,hidden2 =self.for_lstm2(forward1+proj_embed,hidden2)

        s=F.softmax(self.dec_elmo_s,dim=-1)
        ELMO=self.dec_elmo_r*(proj_embed*s[0]+forward1*s[1]+forward2*s[2])
        
        output=torch.cat((inputs_embed,ELMO),dim=-1)
        
        return output,hidden1,hidden2
    

    
    
    
    
    