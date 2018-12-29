# Seq2seq
This code is for the [Bytecup 2018 competition](https://biendata.com/competition/bytecup2018/), for text Neural Headline Generation (NHG).The rank in the final Leaderboard is 4/1066 </br>

## 运行环境：
ubuntu 16.04</br>
python >=3.6</br>
pytorch >=0.4.0</br>
allennlp </br>
tqdm</br>


## 运行说明：

代码都在code文件夹里</br>
先运行preprocess.py 预处理数据</br>
然后分别运行dec_elmo.py 与 enc_elmo.py 预训练文档与标题的语言模型</br>
之后运行除了ensemble.py之外的其他8个文件，分开运行，一共训练8个模型</br>
最后运行ensemble.py进行模型融合</br>

初始的数据与中间数据在data文件夹下</br>
存储模型在checkpoint文件夹下</br>
最终生成结果在result文件夹里</br>


## elmo 
仿照elmo的嵌入方式和bert的mask方式，用双向LSTM预训练了decoder和encoder的elmo语言模型，作为词向量。

## copy mechanism 
使用了copynet论文中的copy 机制，有效提高对UNK的处理

## new training method 
在训练的过程中，逐步提高预测词作为下次输入的概率，以减小训练时和预测时分布不同所带来的影响。

## multi attention 
与facebook的CovS2S一样，采用了两层attention，效果比单层要好

The code will be released when the competition is over
