# Seq2seq
This code is for the [competition](https://biendata.com/competition/bytecup2018/), for text Neural Headline Generation (NHG).The rank in the final Leaderboard is 4/1066 </br>

## elmo 
仿照elmo的嵌入方式和bert的mask方式，用双向LSTM预训练了decoder和encoder的elmo语言模型，作为词向量。

## copy mechanism 
使用了copynet论文中的copy 机制，有效提高对UNK的处理

## new training method 
在训练的过程中，逐步提高预测词作为下次输入的概率，以减小训练时和预测时分布不同所带来的影响。

## multi attention 
与facebook的CovS2S一样，采用了两层attention，效果比单层要好

The code will be released when the competition is over
