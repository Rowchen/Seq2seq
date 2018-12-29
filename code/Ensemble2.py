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

from utils import *

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

test_doc_embed2=np.load('../data/test_embed2.npy')
test_doc_mask2=get_doc_mask(test_doc_embed2)



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
    need_pretrain=False
    enc_elmo_lstm_dim=512
    
arg=seq2seq_param()

'''
注意这里用了一个elmo embedding的方法
'''

class DecoderRNN1(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN1, self).__init__()
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
        
        self.dec_pos_embed=position_encoding_init(self.dec_max_len,arg.elmo_lstm_dim).repeat(1,1,1)
        
        self.inputlayer=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size+self.attn_size+self.enc_hidden_size, 
                                  arg.dec_embedding_size)
        self.outlayer =nn.Sequential(nn.Dropout(p=0.3)
            ,nn.Linear(self.attn_size, self.output_size)
        )
        
        self.W_copy = nn.Linear(arg.enc_hidden_size, arg.attention_size) # copy mode
        self.scale=nn.Parameter(torch.tensor(1.0))

        self.dec_elmo_embed=dec_elmo(arg).cuda()
        
'''
注意这里用了一个elmo embedding的方法
'''

class DecoderRNN2(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN2, self).__init__()
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
          
        self.dec_elmo_embed=dec_elmo(arg).cuda()
        
        
'''
注意这里用了一个elmo embedding的方法
'''
class DecoderRNN3(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN3, self).__init__()
        self.dec_max_len=arg.dec_max_len
        self.dec_hidden_size=arg.dec_hidden_size
        self.attn_size=arg.attention_size
        self.output_size=arg.vocab_size
        self.num_layers=arg.dec_num_layers
        self.enc_hidden_size=arg.enc_hidden_size
        
        self.rnn = nn.LSTM(arg.dec_embedding_size,arg.dec_hidden_size, 
                           arg.dec_num_layers,batch_first=True)
        self.attention=QKV_attention(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        
        self.rnn2 = nn.LSTM(arg.attention_size,arg.dec_hidden_size, 
                           arg.dec_num_layers,batch_first=True)
        self.attention2=QKV_attention(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        
        self.rnn3 = nn.LSTM(arg.attention_size,arg.dec_hidden_size, 
                           arg.dec_num_layers,batch_first=True)
        self.attention3=QKV_attention(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        
        self.dec_pos_embed=position_encoding_init(self.dec_max_len,arg.elmo_lstm_dim).repeat(1,1,1)
        
        self.inputlayer=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size+self.attn_size+self.enc_hidden_size, 
                                  arg.dec_embedding_size)
        
        self.outlayer =nn.Sequential(nn.Dropout(p=0.3)
            ,nn.Linear(self.attn_size, self.output_size)
        )
        
        self.W_copy = nn.Linear(arg.enc_hidden_size, arg.attention_size) # copy mode
        self.scale=nn.Parameter(torch.tensor(1.0))
        
        self.dec_elmo_embed=dec_elmo(arg).cuda()
        
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
    
class DecoderRNN4(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN4, self).__init__()
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

        self.attention=QKV_attention2(self.attn_size,self.dec_hidden_size,arg.enc_hidden_size)
        self.dec_pos_embed=position_encoding_init(self.dec_max_len,arg.elmo_lstm_dim).repeat(1,1,1)
        self.inputlayer=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size+self.attn_size+self.enc_hidden_size, 
                                  arg.dec_embedding_size)
        self.outlayer =nn.Sequential(nn.Dropout(p=0.3),nn.Linear(self.attn_size, self.output_size))
        self.atten_layer=nn.Linear(arg.enc_hidden_size+arg.dec_hidden_size,arg.attention_size)
        
        self.dec_elmo_embed=dec_elmo(arg)
        self.dropout=nn.Dropout(p=0.3,inplace=True)
        
        self.copy_attn=nn.Linear(arg.attention_size,1,bias=False)
        self.copy_input=nn.Linear(arg.elmo_lstm_dim+arg.embedding_size,1,bias=False)
        
        self.scale=nn.Parameter(torch.tensor(1.0))
        
'''
注意这里用了一个elmo embedding的方法
'''
class DecoderRNN5(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN5, self).__init__()
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
                
        self.dec_elmo_embed=dec_elmo(arg).cuda()
'''
注意这里用了一个elmo embedding的方法
'''
class DecoderRNN6(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN6, self).__init__()
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
        self.dec_elmo_embed=dec_elmo(arg).cuda()

'''
注意这里用了一个elmo embedding的方法
'''

class DecoderRNN7(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN7, self).__init__()
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

        self.dec_elmo_embed=dec_elmo(arg).cuda()
        self.dropout=nn.Dropout(p=0.3,inplace=True)

'''
注意这里用了一个elmo embedding的方法
'''

class DecoderRNN8(nn.Module):
    def __init__(self,arg):
        super(DecoderRNN8, self).__init__()
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
        self.dec_elmo_embed=dec_elmo(arg).cuda()
class seq2seq8(nn.Module):
    def __init__(self,arg):
        super(seq2seq8, self).__init__()
        self.encoder=EncoderRNN3(arg)
        self.decoder=DecoderRNN8(arg)
model8=seq2seq8(arg).cuda()
model8.load_state_dict(torch.load('../checkpoint/attent2-500dim_select_2.pkl'))

class seq2seq7(nn.Module):
    def __init__(self,arg):
        super(seq2seq7, self).__init__()
        self.encoder=EncoderRNN2(arg)
        self.decoder=DecoderRNN7(arg)
model7=seq2seq7(arg).cuda()
model7.load_state_dict(torch.load('../checkpoint/attentionnorm_2.pkl'))

class seq2seq6(nn.Module):
    def __init__(self,arg):
        super(seq2seq6, self).__init__()
        self.encoder=EncoderRNN2(arg)
        self.decoder=DecoderRNN6(arg)
model6=seq2seq6(arg).cuda()
model6.load_state_dict(torch.load('../checkpoint/2attention_tradition.pkl'))

class seq2seq5(nn.Module):
    def __init__(self,arg):
        super(seq2seq5, self).__init__()
        self.encoder=EncoderRNN2(arg)
        self.decoder=DecoderRNN5(arg)
class seq2seq4(nn.Module):
    def __init__(self,arg):
        super(seq2seq4, self).__init__()
        self.encoder=EncoderRNN1(arg)
        self.decoder=DecoderRNN4(arg)
class seq2seq1(nn.Module):
    def __init__(self,arg):
        super(seq2seq1, self).__init__()
        self.encoder=EncoderRNN1(arg)
        self.decoder=DecoderRNN1(arg)
class seq2seq2(nn.Module):
    def __init__(self,arg):
        super(seq2seq2, self).__init__()
        self.encoder=EncoderRNN1(arg)
        self.decoder=DecoderRNN2(arg)
class seq2seq3(nn.Module):
    def __init__(self,arg):
        super(seq2seq3, self).__init__()
        self.encoder=EncoderRNN1(arg)
        self.decoder=DecoderRNN3(arg)
        
model5=seq2seq5(arg).cuda()
model5.load_state_dict(torch.load('../checkpoint/enc_elmo_copy_traidition.pkl'))
model4=seq2seq4(arg).cuda()
model4.load_state_dict(torch.load('../checkpoint/enc_elmo_copy_coverage.pkl'))
model3=seq2seq3(arg).cuda()
model3.load_state_dict(torch.load('../checkpoint/enc_elmo_copy_3attention.pkl'))
model2=seq2seq2(arg).cuda()
model2.load_state_dict(torch.load('../checkpoint/enc_elmo_copy_2attention.pkl'))
model1=seq2seq1(arg).cuda()
model1.load_state_dict(torch.load('../checkpoint/enc_elmo_copy.pkl'))


def beam_search(enc_in,enc_mask,TFIN,beam_width=5):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    model8.eval()

    max_length=model1.decoder.dec_max_len
    batch_size=enc_in.size(0)
    
    assert(batch_size==1)
    
    '''model1'''
    a_enc_out=model1.encoder(enc_in)
    a_dec_hidden,a_elmo_hidden1,a_elmo_hidden2=None,None,None
    a_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    a_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    a_enc_out=a_enc_out.repeat(beam_width,1,1)
    
    '''model2'''
    b_enc_out=model2.encoder(enc_in)
    b_dec_hidden1,b_dec_hidden2,b_elmo_hidden1,b_elmo_hidden2=None,None,None,None
    b_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    b_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    b_enc_out=b_enc_out.repeat(beam_width,1,1)
    
    '''model3'''
    c_enc_out=model3.encoder(enc_in)
    c_dec_hidden1,c_dec_hidden2,c_dec_hidden3,c_elmo_hidden1,c_elmo_hidden2=None,None,None,None,None
    c_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    c_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    c_enc_out=c_enc_out.repeat(beam_width,1,1)
    
    '''model4'''
    d_enc_out=model4.encoder(enc_in)
    seq_len=d_enc_out.size(1)
    d_dec_hidden,d_elmo_hidden1,d_elmo_hidden2=None,None,None
    d_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    d_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    d_enc_out=d_enc_out.repeat(beam_width,1,1)
    d_atten_score_his=torch.ones(batch_size,1,seq_len).cuda()/seq_len#cumulate attention sco
    
    '''model5'''
    e_enc_out=model5.encoder(enc_in,TFIN)
    e_dec_hidden1,e_dec_hidden2,e_elmo_hidden1,e_elmo_hidden2=None,None,None,None
    e_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    e_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    e_enc_out=e_enc_out.repeat(beam_width,1,1)
    
    '''model6'''
    f_enc_out=model6.encoder(enc_in,TFIN)
    f_dec_hidden1,f_dec_hidden2,f_elmo_hidden1,f_elmo_hidden2=None,None,None,None
    f_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    f_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    f_enc_out=f_enc_out.repeat(beam_width,1,1)
    
    '''model7'''
    g_enc_out=model7.encoder(enc_in,TFIN)
    g_dec_hidden1,g_dec_hidden2,g_elmo_hidden1,g_elmo_hidden2=None,None,None,None
    g_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    g_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    g_enc_out=g_enc_out.repeat(beam_width,1,1)
    
    '''model8'''
    h_enc_out=model8.encoder(enc_in,TFIN)
    h_dec_hidden1,h_dec_hidden2,h_elmo_hidden1,h_elmo_hidden2=None,None,None,None
    h_dec_att_out=torch.zeros(beam_width,1,arg.attention_size).cuda()
    h_select_read=torch.zeros(beam_width,1,arg.enc_hidden_size).cuda()
    h_enc_out=h_enc_out.repeat(beam_width,1,1)
    
    '''share'''
    dec_symbol=START*torch.ones(beam_width,1,dtype=torch.long).cuda()
    enc_in=enc_in.repeat(beam_width,1)
    enc_in2=torch.unsqueeze(enc_in,dim=1)
    enc_mask=torch.ByteTensor(np.tile(enc_mask,[beam_width,1,1])).cuda()

    beam_proba=torch.zeros(beam_width,1).cuda()
    sequence_symbols=[]
    length=0
    
    
    
    for i in range(max_length):
        '''model1'--- one attention'''
        a_in_embed,a_elmo_hidden1,a_elmo_hidden2=\
                    model1.decoder.dec_elmo_embed(dec_symbol,a_elmo_hidden1,a_elmo_hidden2)
        
        a_dec_in=model1.decoder.inputlayer(torch.cat((a_in_embed,a_dec_att_out,a_select_read),dim=2))+\
                    model1.decoder.dec_pos_embed[:,i:i+1,:]
        
        a_dec_out, a_dec_hidden= model1.decoder.rnn(a_dec_in, a_dec_hidden)
        a_dec_att_out=model1.decoder.attention(a_dec_out,a_enc_out,enc_mask)

        a_score_c=torch.bmm(a_dec_att_out,torch.transpose(torch.tanh(model1.decoder.W_copy(a_enc_out)),1,2))
        a_score_c.data.masked_fill_(enc_mask, -float('inf'))
        a_score_c=F.softmax(a_score_c,dim=-1)
        a_score_e=a_score_c*model1.decoder.scale*model1.decoder.scale
        
        a_dec_to_vocab=model1.decoder.outlayer(a_dec_att_out)
        a_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=a_score_e)
        
        
        '''model2--- two attention'''
        b_in_embed,b_elmo_hidden1,b_elmo_hidden2=\
                    model2.decoder.dec_elmo_embed(dec_symbol,b_elmo_hidden1,b_elmo_hidden2)

        b_dec_in=model2.decoder.inputlayer(torch.cat((b_in_embed,b_dec_att_out,b_select_read),dim=2))+\
                    model2.decoder.dec_pos_embed[:,i:i+1,:]

        b_dec_out, b_dec_hidden1 = model2.decoder.rnn(b_dec_in, b_dec_hidden1)
        b_dec_att_out=model2.decoder.attention(b_dec_out,b_enc_out,enc_mask)
        b_dec_out, b_dec_hidden2 = model2.decoder.rnn2(b_dec_att_out, b_dec_hidden2)
        b_dec_att_out=model2.decoder.attention2(b_dec_out,b_enc_out,enc_mask)+b_dec_att_out

        b_score_c=torch.bmm(b_dec_att_out,torch.transpose(torch.tanh(model2.decoder.W_copy(b_enc_out)),1,2))
        b_score_c.data.masked_fill_(enc_mask, -float('inf'))
        b_score_c=F.softmax(b_score_c,dim=-1)
        b_score_e=b_score_c*model2.decoder.scale*model2.decoder.scale
        
        b_dec_to_vocab=model2.decoder.outlayer(b_dec_att_out)
        b_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=b_score_e)
        
        '''model3--- three attention'''
        c_in_embed,c_elmo_hidden1,c_elmo_hidden2=\
                    model3.decoder.dec_elmo_embed(dec_symbol,c_elmo_hidden1,c_elmo_hidden2)

        c_dec_in=model3.decoder.inputlayer(torch.cat((c_in_embed,c_dec_att_out,c_select_read),dim=2))+\
                    model3.decoder.dec_pos_embed[:,i:i+1,:]
        
        c_dec_out, c_dec_hidden1 = model3.decoder.rnn(c_dec_in, c_dec_hidden1)
        c_dec_att_out=model3.decoder.attention(c_dec_out,c_enc_out,enc_mask)
        c_dec_out, c_dec_hidden2 = model3.decoder.rnn2(c_dec_att_out, c_dec_hidden2)
        c_dec_att_out=0.707*(model3.decoder.attention2(c_dec_out,c_enc_out,enc_mask)+c_dec_att_out)
        c_dec_out, c_dec_hidden3 = model3.decoder.rnn3(c_dec_att_out, c_dec_hidden3)
        c_dec_att_out=0.707*(model3.decoder.attention3(c_dec_out,c_enc_out,enc_mask)+c_dec_att_out)

        c_score_c=torch.bmm(c_dec_att_out,torch.transpose(torch.tanh(model3.decoder.W_copy(c_enc_out)),1,2))
        c_score_c.data.masked_fill_(enc_mask, -float('inf'))
        c_score_c=F.softmax(c_score_c,dim=-1)
        c_score_e=c_score_c*model3.decoder.scale*model3.decoder.scale

        c_dec_to_vocab=model3.decoder.outlayer(c_dec_att_out)
        c_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=c_score_e)
        
        
        '''model4---coverage'''
        d_in_embed,d_elmo_hidden1,d_elmo_hidden2=\
                    model4.decoder.dec_elmo_embed(dec_symbol,d_elmo_hidden1,d_elmo_hidden2)

        d_dec_in=model4.decoder.inputlayer(torch.cat((d_in_embed,d_dec_att_out,d_select_read),dim=2))+\
                    model4.decoder.dec_pos_embed[:,i:i+1,:]

        d_dec_out, d_dec_hidden= model4.decoder.rnn(d_dec_in, d_dec_hidden)
        d_atten_score=model4.decoder.attention(d_dec_out,d_enc_out,enc_mask,d_atten_score_his)

        d_context=torch.bmm(d_atten_score, d_enc_out)
        d_dec_att_out=model4.decoder.atten_layer(torch.cat((d_context,d_dec_out),dim=2))

        d_proba_copy=(model4.decoder.copy_attn(d_dec_att_out)+model4.decoder.copy_input(d_in_embed)+model4.decoder.scale)

        d_score_e=d_atten_score*(d_proba_copy*d_proba_copy)
        d_dec_to_vocab=model4.decoder.outlayer(d_dec_att_out)
        d_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=d_score_e)
        
        d_atten_score_his=(d_atten_score+d_atten_score_his)/2
        
        '''model5--- new training and tradition1'''
        e_in_embed,e_elmo_hidden1,e_elmo_hidden2=\
                    model5.decoder.dec_elmo_embed(dec_symbol,e_elmo_hidden1,e_elmo_hidden2)

        e_dec_in=model5.decoder.inputlayer(torch.cat((e_in_embed,e_dec_att_out,e_select_read),dim=2))+\
                    model5.decoder.dec_pos_embed[:,i:i+1,:]

        e_dec_out, e_dec_hidden1 = model5.decoder.rnn(e_dec_in, e_dec_hidden1)
        e_dec_att_out=model5.decoder.attention(e_dec_out,e_enc_out,enc_mask)
        e_dec_out, e_dec_hidden2 = model5.decoder.rnn2(e_dec_att_out, e_dec_hidden2)
        e_dec_att_out=model5.decoder.attention2(e_dec_out,e_enc_out,enc_mask)+e_dec_att_out

        e_score_c=torch.bmm(e_dec_att_out,torch.transpose(torch.tanh(model5.decoder.W_copy(e_enc_out)),1,2))
        e_score_c.data.masked_fill_(enc_mask, -float('inf'))
        e_score_c=F.softmax(e_score_c,dim=-1)
        e_score_e=e_score_c*model5.decoder.scale*model5.decoder.scale
        
        e_dec_to_vocab=model5.decoder.outlayer(e_dec_att_out)
        e_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=e_score_e)
        
        '''model6--- new training and tradition2'''
        f_in_embed,f_elmo_hidden1,f_elmo_hidden2=\
                    model6.decoder.dec_elmo_embed(dec_symbol,f_elmo_hidden1,f_elmo_hidden2)

        f_dec_in=model6.decoder.inputlayer(torch.cat((f_in_embed,f_dec_att_out,f_select_read),dim=2))+\
                    model6.decoder.dec_pos_embed[:,i:i+1,:]

        f_dec_out, f_dec_hidden1 = model6.decoder.rnn(f_dec_in, f_dec_hidden1)
        f_dec_att_out=model6.decoder.attention(f_dec_out,f_enc_out,enc_mask)
        f_dec_out, f_dec_hidden2 = model6.decoder.rnn2(f_dec_att_out, f_dec_hidden2)
        f_dec_att_out=model6.decoder.attention2(f_dec_out,f_enc_out,enc_mask)+f_dec_att_out

        f_score_c=torch.bmm(f_dec_att_out,torch.transpose(torch.tanh(model6.decoder.W_copy(f_enc_out)),1,2))
        f_score_c.data.masked_fill_(enc_mask, -float('inf'))
        f_score_c=F.softmax(f_score_c,dim=-1)
        f_score_e=f_score_c*model6.decoder.scale*model6.decoder.scale
        
        f_dec_to_vocab=model6.decoder.outlayer(f_dec_att_out)
        f_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=f_score_e)
        
        '''model7--- new training and tradition3_nrom'''
        g_in_embed,g_elmo_hidden1,g_elmo_hidden2=\
                    model7.decoder.dec_elmo_embed(dec_symbol,g_elmo_hidden1,g_elmo_hidden2)

        g_dec_in=model7.decoder.inputlayer(torch.cat((g_in_embed,g_dec_att_out,g_select_read),dim=2))+\
                    model7.decoder.dec_pos_embed[:,i:i+1,:]

        g_dec_out, g_dec_hidden1 = model7.decoder.rnn(g_dec_in, g_dec_hidden1)
        g_dec_att_out=model7.decoder.attention(g_dec_out,g_enc_out,enc_mask)
        g_dec_out, g_dec_hidden2 = model7.decoder.rnn2(g_dec_att_out, g_dec_hidden2)
        g_dec_att_out=model7.decoder.attention2(g_dec_out,g_enc_out,enc_mask)+g_dec_att_out

        g_score_c=torch.bmm(g_dec_att_out,torch.transpose(torch.tanh(model7.decoder.W_copy(g_enc_out)),1,2))
        g_score_c.data.masked_fill_(enc_mask, -float('inf'))
        g_score_c=F.softmax(g_score_c,dim=-1)
        g_score_e=g_score_c*model7.decoder.scale*model7.decoder.scale
        
        g_dec_to_vocab=model7.decoder.outlayer(g_dec_att_out)
        g_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=g_score_e)
        
        '''model8--- new training and tradition-select'''
        h_in_embed,h_elmo_hidden1,h_elmo_hidden2=\
                    model8.decoder.dec_elmo_embed(dec_symbol,h_elmo_hidden1,h_elmo_hidden2)

        h_dec_in=model8.decoder.inputlayer(torch.cat((h_in_embed,h_dec_att_out,h_select_read),dim=2))+\
                    model8.decoder.dec_pos_embed[:,i:i+1,:]

        h_dec_out, h_dec_hidden1 = model8.decoder.rnn(h_dec_in, h_dec_hidden1)
        h_dec_att_out=model8.decoder.attention(h_dec_out,h_enc_out,enc_mask)
        h_dec_out, h_dec_hidden2 = model8.decoder.rnn2(h_dec_att_out, h_dec_hidden2)
        h_dec_att_out=model8.decoder.attention2(h_dec_out,h_enc_out,enc_mask)+h_dec_att_out

        h_score_c=torch.bmm(h_dec_att_out,torch.transpose(torch.tanh(model8.decoder.W_copy(h_enc_out)),1,2))
        h_score_c.data.masked_fill_(enc_mask, -float('inf'))
        h_score_c=F.softmax(h_score_c,dim=-1)
        h_score_e=h_score_c*model8.decoder.scale*model8.decoder.scale
        
        h_dec_to_vocab=model8.decoder.outlayer(h_dec_att_out)
        h_dec_to_vocab.scatter_add_(dim=-1,index=enc_in2,src=h_score_e)

        
        a_proba=F.log_softmax(a_dec_to_vocab,dim=2).squeeze()
        b_proba=F.log_softmax(b_dec_to_vocab,dim=2).squeeze()
        c_proba=F.log_softmax(c_dec_to_vocab,dim=2).squeeze()
        d_proba=F.log_softmax(d_dec_to_vocab,dim=2).squeeze()
        e_proba=F.log_softmax(e_dec_to_vocab,dim=2).squeeze()
        f_proba=F.log_softmax(f_dec_to_vocab,dim=2).squeeze()
        g_proba=F.log_softmax(g_dec_to_vocab,dim=2).squeeze()
        h_proba=F.log_softmax(h_dec_to_vocab,dim=2).squeeze()
        
        proba=(a_proba+b_proba+c_proba+d_proba+e_proba+f_proba+g_proba+h_proba)/8+beam_proba
        
        
        if i==0:
            select=torch.topk(proba[0],beam_width)[1]
            dec_symbol=select.reshape(beam_width,1)
            beam_proba=proba[0,select].reshape(beam_width,1)
            sequence_symbols.append(dec_symbol)
            choose=select//arg.vocab_size
            
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
                choose=select//arg.vocab_size

                beam_proba=proba[select].reshape(beam_width,1)
                select=select%arg.vocab_size#第几个token
                dec_symbol=select.reshape(beam_width,1)
            
            ls=torch.cat((sequence_symbols[-1][choose,:],dec_symbol),dim=1)
            sequence_symbols.append(ls)
            if dec_symbol[0,0]==END:
                break
                
        '''model1'''
        a_score_f=a_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        a_select_read=torch.bmm(a_score_f,a_enc_out)
        a_elmo_hidden1=(a_elmo_hidden1[0][:,choose,:],a_elmo_hidden1[1][:,choose,:])
        a_elmo_hidden2=(a_elmo_hidden2[0][:,choose,:],a_elmo_hidden2[1][:,choose,:])
        a_dec_hidden=(a_dec_hidden[0][:,choose,:],a_dec_hidden[1][:,choose,:])
        a_dec_att_out=a_dec_att_out[choose,:,:]
        
        '''model2'''
        b_score_f=b_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        b_select_read=torch.bmm(b_score_f,b_enc_out)
        b_elmo_hidden1=(b_elmo_hidden1[0][:,choose,:],b_elmo_hidden1[1][:,choose,:])
        b_elmo_hidden2=(b_elmo_hidden2[0][:,choose,:],b_elmo_hidden2[1][:,choose,:])
        b_dec_hidden1=(b_dec_hidden1[0][:,choose,:],b_dec_hidden1[1][:,choose,:])
        b_dec_hidden2=(b_dec_hidden2[0][:,choose,:],b_dec_hidden2[1][:,choose,:])
        b_dec_att_out=b_dec_att_out[choose,:,:]
        
        '''model3'''
        c_score_f=c_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        c_select_read=torch.bmm(c_score_f,c_enc_out)
        c_elmo_hidden1=(c_elmo_hidden1[0][:,choose,:],c_elmo_hidden1[1][:,choose,:])
        c_elmo_hidden2=(c_elmo_hidden2[0][:,choose,:],c_elmo_hidden2[1][:,choose,:])
        c_dec_hidden1=(c_dec_hidden1[0][:,choose,:],c_dec_hidden1[1][:,choose,:])
        c_dec_hidden2=(c_dec_hidden2[0][:,choose,:],c_dec_hidden2[1][:,choose,:])
        c_dec_hidden3=(c_dec_hidden3[0][:,choose,:],c_dec_hidden3[1][:,choose,:])
        c_dec_att_out=c_dec_att_out[choose,:,:]
        
        '''model4'''
        d_score_f=d_atten_score[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        d_select_read=torch.bmm(d_score_f,d_enc_out)
        d_elmo_hidden1=(d_elmo_hidden1[0][:,choose,:],d_elmo_hidden1[1][:,choose,:])
        d_elmo_hidden2=(d_elmo_hidden2[0][:,choose,:],d_elmo_hidden2[1][:,choose,:])
        d_dec_hidden=(d_dec_hidden[0][:,choose,:],d_dec_hidden[1][:,choose,:])
        d_dec_att_out=d_dec_att_out[choose,:,:]
        d_atten_score_his=d_atten_score_his[choose,:,:]
        
        '''model5'''
        e_score_f=e_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        e_select_read=torch.bmm(e_score_f,e_enc_out)
        e_elmo_hidden1=(e_elmo_hidden1[0][:,choose,:],e_elmo_hidden1[1][:,choose,:])
        e_elmo_hidden2=(e_elmo_hidden2[0][:,choose,:],e_elmo_hidden2[1][:,choose,:])
        e_dec_hidden1=(e_dec_hidden1[0][:,choose,:],e_dec_hidden1[1][:,choose,:])
        e_dec_hidden2=(e_dec_hidden2[0][:,choose,:],e_dec_hidden2[1][:,choose,:])
        e_dec_att_out=e_dec_att_out[choose,:,:]
        
        '''model6'''
        f_score_f=f_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        f_select_read=torch.bmm(f_score_f,f_enc_out)
        f_elmo_hidden1=(f_elmo_hidden1[0][:,choose,:],f_elmo_hidden1[1][:,choose,:])
        f_elmo_hidden2=(f_elmo_hidden2[0][:,choose,:],f_elmo_hidden2[1][:,choose,:])
        f_dec_hidden1=(f_dec_hidden1[0][:,choose,:],f_dec_hidden1[1][:,choose,:])
        f_dec_hidden2=(f_dec_hidden2[0][:,choose,:],f_dec_hidden2[1][:,choose,:])
        f_dec_att_out=f_dec_att_out[choose,:,:]
        
        '''model7'''
        g_score_f=g_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        g_score_f=g_score_f/(torch.sum(g_score_f,dim=-1,keepdim=True)+1e-8)
        g_select_read=torch.bmm(g_score_f,g_enc_out)
        g_elmo_hidden1=(g_elmo_hidden1[0][:,choose,:],g_elmo_hidden1[1][:,choose,:])
        g_elmo_hidden2=(g_elmo_hidden2[0][:,choose,:],g_elmo_hidden2[1][:,choose,:])
        g_dec_hidden1=(g_dec_hidden1[0][:,choose,:],g_dec_hidden1[1][:,choose,:])
        g_dec_hidden2=(g_dec_hidden2[0][:,choose,:],g_dec_hidden2[1][:,choose,:])
        g_dec_att_out=g_dec_att_out[choose,:,:]
        
        '''model8'''
        h_score_f=h_score_c[choose,:]*((enc_in==dec_symbol).float().unsqueeze(dim=1))
        h_score_f=h_score_f/(torch.sum(h_score_f,dim=-1,keepdim=True)+1e-8)
        h_select_read=torch.bmm(h_score_f,h_enc_out)
        
        h_elmo_hidden1=(h_elmo_hidden1[0][:,choose,:],h_elmo_hidden1[1][:,choose,:])
        h_elmo_hidden2=(h_elmo_hidden2[0][:,choose,:],h_elmo_hidden2[1][:,choose,:])
        h_dec_hidden1=(h_dec_hidden1[0][:,choose,:],h_dec_hidden1[1][:,choose,:])
        h_dec_hidden2=(h_dec_hidden2[0][:,choose,:],h_dec_hidden2[1][:,choose,:])
        h_dec_att_out=h_dec_att_out[choose,:,:]
        
    return sequence_symbols[-1],beam_proba[0]
    
import re
def remove_repeat_phrase(text):
    words=text.split()
    fixwords=[]
    i=0
    leng=len(words)
    while i <leng:
        '''接下来6个字符与前面6个一样'''
        if i<leng-5:
            ls=words[i]+' '+words[i+1]+' '+words[i+2]+' '+words[i+3]+' '+words[i+4]+' '+words[i+5]
            if i>=6: 
                ls2=words[i-6]+' '+words[i-5]+' '+words[i-4]+' '+words[i-3]+' '+words[i-2]+' '+words[i-1]
                if(ls2==ls):
                    i+=6
                    continue
            '''隔了一个一样'''   
            if i>=7:
                ls2=words[i-7]+' '+words[i-6]+' '+words[i-5]+' '+words[i-4]+' '+words[i-3]+' '+words[i-2]
                if(ls2==ls):
                    i+=6
                    continue
        
        '''接下来5个字符与前面5个一样'''
        if i<leng-4:
            ls=words[i]+' '+words[i+1]+' '+words[i+2]+' '+words[i+3]+' '+words[i+4]
            if i>=5: 
                ls2=words[i-5]+' '+words[i-4]+' '+words[i-3]+' '+words[i-2]+' '+words[i-1]
                if(ls2==ls):
                    i+=5
                    continue
            '''隔了一个一样'''   
            if i>=6:
                ls2=words[i-6]+' '+words[i-5]+' '+words[i-4]+' '+words[i-3]+' '+words[i-2]
                if(ls2==ls):
                    i+=5
                    continue
                    
        '''接下来4个字符与前面4个一样'''
        if i<leng-3:
            ls=words[i]+' '+words[i+1]+' '+words[i+2]+' '+words[i+3]
            if i>=4:
                ls2=words[i-4]+' '+words[i-3]+' '+words[i-2]+' '+words[i-1]
                if(ls2==ls):
                    i+=4#跳过这四个字符
                    continue
            '''隔了一个一样'''           
            if i>=5:
                ls2=words[i-5]+' '+words[i-4]+' '+words[i-3]+' '+words[i-2]
                if(ls2==ls):
                    i+=4
                    continue
                
        if i<leng-2:
            ls=words[i]+' '+words[i+1]+' '+words[i+2]
            if i>=3 :
                ls2=words[i-3]+' '+words[i-2]+' '+words[i-1]
                if(ls2==ls):
                    i+=3
                    continue
            if i>=4:
                ls3=words[i-4]+' '+words[i-3]+' '+words[i-2]
                if(ls3==ls):
                    i+=3
                    continue

        if i<leng-1:
            ls=words[i]+' '+words[i+1]
            if i>=2:
                ls2=words[i-2]+' '+words[i-1]
                if(ls2==ls):
                    i+=2
                    continue
            if i>=3:
                ls2=words[i-3]+' '+words[i-2]
                if(ls2==ls):
                    i+=2
                    continue
                
        '''连续三个一样的字符'''
        if i<leng-2:
            ls=words[i+1]
            ls2=words[i+2]
            if ls==words[i] and ls2==words[i]:
                i+=2
                continue
                
        if i<leng-1:
            ls=words[i+1]
            if ls==words[i]:
                i+=1
                continue
        
        fixwords.append(words[i])
        i+=1
        
    sentence=" ".join(fixwords)
    sentence=re.sub("’","'",sentence)
    sentence=re.sub("‘","'",sentence)
    return sentence

'''test'''
import gc
import time
testunk2=np.load('../data/test_doc_hash_map2.npy')
batch_size=1
a=time.time()
model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()
model7.eval()
ite=0
test_pred=[]
with torch.no_grad():
    #for ite in tqdm(range(10,20)):
    for ite in tqdm(range(test_doc_embed2.shape[0])):
        if(ite==366):
            test_pred.append(" ")
            continue
        gc.collect()
        enc_in=torch.tensor(test_doc_embed2[ite:ite+batch_size],dtype=torch.long).cuda()
        enc_mask=test_doc_mask2[ite:ite+batch_size]
        TFIN=torch.LongTensor(getTF(test_doc_embed2[ite:ite+batch_size])).cuda()
        pred_sym,_=beam_search(enc_in,enc_mask,TFIN,4)
        ls=[]
        for w in pred_sym[0]:
            w=vocab[w]
            if w=='END':
                break
            if 'UNK' in w:
                num=int(w.strip('UNK'))
                for j in testunk2[ite]:
                    if testunk2[ite][j]==num:
                        ls.append(j)
                        break
            else:
                ls.append(w)
        test_pred.append(" ".join(ls))
    
print (time.time()-a)


doc_hash_map=np.load('../data/test_doc_hash_map2.npy')
digit_hash_map=np.load('../data/test_digit_hash_map2.npy')
digit_hash_map_before=np.load('../data/test_digit_hash_map_before2.npy')

test_pred2=[]
cnt=0
for row in range(len(test_pred)):
    words=(test_pred[row]).split()
    digitmap=digit_hash_map[row]
    digitmap_b=digit_hash_map_before[row]
    words2=[]
    for i in range(len(words)):
        try:
            if words[i]=='DIGIT':
                ls=words[i+1]+" "+words[i+2]
                if ls in digitmap:
                    words[i]=digitmap[ls][0]
        except:
            pass
        try:
            if words[i]=='DIGIT':
                ls=words[i+1]
                if ls in digitmap:
                    words[i]=digitmap[ls][0]
        except:
            pass
        try:
            if words[i]=='DIGIT':
                ls=words[i-1]
                if ls in digitmap_b:
                    words[i]=digitmap_b[ls][0]
        except:
            pass
        
        if words[i]=='DIGIT':
            cnt+=1
        else:
            words2.append(words[i])
            
    test_pred2.append(" ".join(words2))
print(cnt) 
print (len(test_pred2))

test_pred3=[]
for row in range(len(test_pred2)):
    ls=remove_repeat_phrase(test_pred2[row])
    test_pred3.append(remove_repeat_phrase(ls))


    
import json
doc=[]
idx=[]
'''test_file'''
f = open("../data/bytecup.corpus.test_set.txt", encoding='utf-8')
line=f.readline()
while (line):
    ls=json.loads(line)
    doc.append(ls['content'])
    idx.append(ls['id'])
    line=f.readline()
f.close()
gc.collect()

for i in tqdm(range(test_doc_embed2.shape[0])):
    fi=int(idx[i])
    f= open('../team_result/myresult2/%d.txt'%(fi),'w',encoding='utf-8')
    f.write(test_pred3[i])
    f.close()
        

