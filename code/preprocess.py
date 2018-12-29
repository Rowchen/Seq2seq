import json
import pandas as pd
from tqdm import tqdm
import gc
import numpy as np
import random


from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/home/row/stanfordnlp/stanford-corenlp-full-2018-01-31')
wordcnt={}
for i in range(1,9):
    doc=[]
    title=[]
    idx=[]
    
    f = open("../data/bytecup.corpus.train.%d.txt"%i, encoding='utf-8')
    line=f.readline()
    while (line):
        ls=json.loads(line)
        doc.append(ls['content'])
        title.append(ls['title'])
        line=f.readline()
    f.close()
    gc.collect()
    
    for i in tqdm(range(len(doc))):
        for w in nlp.word_tokenize(str(doc[i][:20000]+" "+title[i])):
            w=w.lower()
            wordcnt[w]=wordcnt.get(w,0)+1
            
import operator
cd=dict(sorted(wordcnt.items(),key=operator.itemgetter(1),reverse=True)[:80000])
vocab=list(cd.keys())
f = open('../data/vocab2.txt','w')
for w in vocab:
    f.write(w+'\n')
f.close()


doc=[]
title=[]
idx=[]
for i in tqdm(range(1)):
    f = open("../data/bytecup.corpus.train.%d.txt"%i, encoding='utf-8')
    line=f.readline()
    while (line):
        ls=json.loads(line)
        doc.append(ls['content'])
        title.append(ls['title'])
        line=f.readline()
    f.close()
    gc.collect()
    


doc=[]
title=[]
idx=[]
for i in tqdm(range(9)):
    f = open("../data/bytecup.corpus.train.%d.txt"%i, encoding='utf-8')
    line=f.readline()
    while (line):
        ls=json.loads(line)
        doc.append(ls['content'])
        title.append(ls['title'])
        line=f.readline()
    f.close()
    gc.collect()
    
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

word2vec={}
for w in vocab:
    word2vec[w]=(np.random.rand(300)-0.5)/2
word2vec['PADDING']=np.zeros(300)

word2idx={}
cnt=0
for w in vocab:
    word2idx[w]=cnt
    cnt+=1
    
    
'''get embedding matrix'''
import io
cnt=0
with io.open('../glove.840B.300d.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')as f:
    for line in tqdm(f):
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocab2:
            word2vec[tokens[0]]=np.array(tokens[1:]).astype(float)
            cnt+=1
            

embedding=[]
for w in vocab:
    embedding.append(word2vec[w])
embedding=np.array(embedding)

print (embedding.shape)
np.save('../data/embedding.npy',embedding)


'''get the doc represent id'''
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/home/row/stanfordnlp/stanford-corenlp-full-2018-01-31')
doc_max_len=1000
train_embed=np.zeros((200000,doc_max_len),dtype=np.int32)
doc_hash_map=[]
for i,sent in tqdm(enumerate(doc)):
    hashmap={}
    for j,w in enumerate(nlp.word_tokenize(str(sent)[:10000])[:doc_max_len]): 
        w=w.lower()
        if w.isdigit():
            train_embed[i%200000,j]=word2idx['DIGIT']
    
        elif w in vocab2:  
            train_embed[i%200000,j]=word2idx[w]
            
        else:
            if w in hashmap:
                train_embed[i%200000,j]=word2idx['UNK%d'%(hashmap[w])] 
            else:
                if len(hashmap)==50:
                    train_embed[i%200000,j]=word2idx['UNK']
                else:
                    hashmap[w]=len(hashmap)
                    train_embed[i%200000,j]=word2idx['UNK%d'%(hashmap[w])]
                    
    doc_hash_map.append(hashmap)        
    
    if i%50000==0:#每隔一段时间重启一下nlp，不然会停住，并且有卡内存的风险
        nlp.close()
        nlp = StanfordCoreNLP('/home/row/stanfordnlp/stanford-corenlp-full-2018-01-31')
        gc.collect()
        
    if (i+1)%200000==0:#每隔20万个文件存档，否则也容易爆内存
        np.save('../data/train_embed%d.npy'%(i//200000),train_embed)
        train_embed=np.zeros((200000,doc_max_len),dtype=np.int32)
        np.save('../data/doc_hash_map%d.npy'%(i//200000),doc_hash_map)
        doc_hash_map=[]
        gc.collect()
        
    if i==len(doc)-1:#如果到文件末尾了
        train_embed=train_embed[:(i%200000)+1,:]
        print (train_embed.shape)
        np.save('../data/train_embed%d.npy'%(i//200000),train_embed)
        np.save('../data/doc_hash_map%d.npy'%(i//200000),doc_hash_map)
        
'''get title embed'''
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/home/row/stanfordnlp/stanford-corenlp-full-2018-01-31')
unkcnt=0
title_max_len=20
title_embed=np.zeros((200000,title_max_len),dtype=np.int32)
doc_hash_map=np.load('../data/doc_hash_map0.npy')
for i,sent in tqdm(enumerate(title)):
    hash_map=doc_hash_map[i%200000]
    for j,w in enumerate(nlp.word_tokenize(str(sent))[:title_max_len]):   
        w=w.lower()
        if w.isdigit():
            title_embed[i%200000,j]=word2idx['DIGIT']
        elif w in vocab2:  
            title_embed[i%200000,j]=word2idx[w]
        else:
            if w in hash_map:
                title_embed[i%200000,j]=word2idx['UNK%d'%hash_map[w]]
            else:
                try:
                    title_embed[i%200000,j]=word2idx['UNK%d'%np.random.randint(0,len(hash_map))]
                except:
                    title_embed[i%200000,j]=word2idx['UNK']
        
    if (i+1)%200000==0:#每隔20万个文件存档，否则也容易爆内存
        np.save('../data/title_embed%d.npy'%(i//200000),title_embed)
        doc_hash_map=np.load('../data/doc_hash_map%d.npy'%((i//200000)+1))
        title_embed=np.zeros((200000,title_max_len),dtype=np.int32)
        gc.collect()
        
    if i==len(title)-1:#如果到文件末尾了
        title_embed=title_embed[:(i%200000)+1,:]
        print (title_embed.shape)
        np.save('../data/title_embed%d.npy'%(i//200000),title_embed)
        
title=np.load('../data/title_embed0.npy')
doc=np.load('../data/train_embed0.npy')
for i in range(1,7):
    title=np.concatenate((title,np.load('../data/title_embed%d.npy'%i)),axis=0)
    doc=np.concatenate((doc,np.load('../data/train_embed%d.npy'%i)),axis=0)
    gc.collect()
print (title.shape,doc.shape)


del doc
del title
del idx
gc.collect()


'''random shffuer'''
r2=list(range(title.shape[0]))
random.shuffle(r2)

ite=0
for i in range(7):
    np.save('../data2/title_embed%d.npy'%i,title[r2[ite:ite+200000]])
    np.save('../data2/train_embed%d.npy'%i,doc[r2[ite:ite+200000]])
    ite+=200000
    

'''get IDF'''
docIDF=np.zeros(78864)
for i in range(7):
    t=np.load('../data/train_embed%d.npy'%i)
    for row in tqdm(range(t.shape[0])):
        docIDF[t[row]]+=1
titleIDF=np.zeros(78864)
for i in range(7):
    t=np.load('../data/title_embed%d.npy'%i)
    for row in tqdm(range(t.shape[0])):
        titleIDF[t[row]]+=1
docIDF=np.log(docIDF+1)
titleIDF=np.log(titleIDF+1)
docIDF[-50:]=0
docIDF[:4]=0
titleIDF[-50:0]=5
titleIDF[:4]=5
titleIDF[4]=14
doc_bin=(np.argsort(docIDF))//(790)
title_bin=(np.argsort(titleIDF))//(790)
weight=np.concatenate((np.eye(100)[doc_bin],np.eye(100)[title_bin]),axis=1)
np.save('../data/IDFweight.npy',weight)

    


number_map={
    'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15','sixteen':'16','seventeen':'17',
    'eighteen':'18','nineteen':'19','twenty':'20'
}

doc=[]
idx=[]
'''validation_file'''
f = open("../data/bytecup.corpus.validation_set.txt", encoding='utf-8')
line=f.readline()
while (line):
    ls=json.loads(line)
    doc.append(ls['content'])
    idx.append(ls['id'])
    line=f.readline()
f.close()
gc.collect()
print (len(doc))


'''get test doc represent id'''
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/home/row/stanfordnlp/stanford-corenlp-full-2018-01-31')

doc_max_len=1000
test_embed=np.zeros((len(doc),doc_max_len),dtype=np.int32)
doc_hash_map=[]
digit_hash_map=[]
digit_hash_map_before=[]

for i,sent in tqdm(enumerate(doc)):
    hashmap={}
    digitmap={}
    digitmap_b={}
    word_list=nlp.word_tokenize(str(sent)[:20000])[:doc_max_len]
    for j,w in enumerate(word_list): 
        w=w.lower()
        if w.isdigit():
            try:
                ls1=word_list[j+1].lower()
                if ls1 not in digitmap:
                    digitmap[ls1]=[]
                digitmap[ls1].append(w)
                
                ls2=word_list[j+1].lower()+' '+word_list[j+2].lower()
                if ls2 not in digitmap:
                    digitmap[ls2]=[]
                digitmap[ls2].append(w)  
            except:
                pass
            
            try:
                ls1=word_list[j-1].lower()
                if ls1 not in digitmap_b:
                    digitmap_b[ls1]=[]
                digitmap_b[ls1].append(w)
            except:
                pass
            
            test_embed[i,j]=word2idx['DIGIT']
            continue

        w2=w
        if w in number_map:
            w=number_map[w]
            try:
                ls1=word_list[j+1].lower()
                if ls1 not in digitmap:
                    digitmap[ls1]=[]
                digitmap[ls1].append(w)
                
                ls2=word_list[j+1].lower()+' '+word_list[j+2].lower()
                if ls2 not in digitmap:
                    digitmap[ls2]=[]
                digitmap[ls2].append(w)  
            except:
                pass
            
            try:
                ls1=word_list[j-1].lower()
                if ls1 not in digitmap_b:
                    digitmap_b[ls1]=[]
                digitmap_b[ls1].append(w)
            except:
                pass
            
        w=w2
        
        if w in vocab2:  
            test_embed[i,j]=word2idx[w]
        else:
            if w in hashmap:
                test_embed[i,j]=word2idx['UNK%d'%(hashmap[w])] 
            else:
                if len(hashmap)==50:
                    test_embed[i,j]=word2idx['UNK']
                else:
                    hashmap[w]=len(hashmap)
                    test_embed[i,j]=word2idx['UNK%d'%(hashmap[w])]
                    
    digit_hash_map.append(digitmap)  
    doc_hash_map.append(hashmap)
    digit_hash_map_before.append(digitmap_b)  


print (test_embed.shape)  
np.save('../data/test_embed.npy',test_embed)
np.save('../data/test_doc_hash_map.npy',doc_hash_map)
np.save('../data/test_digit_hash_map.npy',digit_hash_map)
np.save('../data/test_digit_hash_map_before.npy',digit_hash_map_before)


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
print (len(doc))


'''get test doc represent id'''

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('/home/row/stanfordnlp/stanford-corenlp-full-2018-01-31')

doc_max_len=1000
test_embed=np.zeros((len(doc),doc_max_len),dtype=np.int32)
doc_hash_map=[]
digit_hash_map=[]
digit_hash_map_before=[]

for i,sent in tqdm(enumerate(doc)):
    hashmap={}
    digitmap={}
    digitmap_b={}
    word_list=nlp.word_tokenize(str(sent)[:20000])[:doc_max_len]
    for j,w in enumerate(word_list): 
        w=w.lower()
        if w.isdigit():
            try:
                ls1=word_list[j+1].lower()
                if ls1 not in digitmap:
                    digitmap[ls1]=[]
                digitmap[ls1].append(w)
                
                ls2=word_list[j+1].lower()+' '+word_list[j+2].lower()
                if ls2 not in digitmap:
                    digitmap[ls2]=[]
                digitmap[ls2].append(w)  
            except:
                pass
            
            try:
                ls1=word_list[j-1].lower()
                if ls1 not in digitmap_b:
                    digitmap_b[ls1]=[]
                digitmap_b[ls1].append(w)
            except:
                pass
            
            test_embed[i,j]=word2idx['DIGIT']
            continue

        w2=w
        if w in number_map:
            w=number_map[w]
            try:
                ls1=word_list[j+1].lower()
                if ls1 not in digitmap:
                    digitmap[ls1]=[]
                digitmap[ls1].append(w)
                
                ls2=word_list[j+1].lower()+' '+word_list[j+2].lower()
                if ls2 not in digitmap:
                    digitmap[ls2]=[]
                digitmap[ls2].append(w)  
            except:
                pass
            
            try:
                ls1=word_list[j-1].lower()
                if ls1 not in digitmap_b:
                    digitmap_b[ls1]=[]
                digitmap_b[ls1].append(w)
            except:
                pass
            
        w=w2
        
        if w in vocab2:  
            test_embed[i,j]=word2idx[w]
        else:
            if w in hashmap:
                test_embed[i,j]=word2idx['UNK%d'%(hashmap[w])] 
            else:
                if len(hashmap)==50:
                    test_embed[i,j]=word2idx['UNK']
                else:
                    hashmap[w]=len(hashmap)
                    test_embed[i,j]=word2idx['UNK%d'%(hashmap[w])]
                    
    digit_hash_map.append(digitmap)  
    doc_hash_map.append(hashmap)
    digit_hash_map_before.append(digitmap_b)  


print (test_embed.shape)  
np.save('../data/test_embed2.npy',test_embed)
np.save('../data/test_doc_hash_map2.npy',doc_hash_map)
np.save('../data/test_digit_hash_map2.npy',digit_hash_map)
np.save('../data/test_digit_hash_map_before2.npy',digit_hash_map_before)



 
