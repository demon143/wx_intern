# wx_intern
微信视频号推荐比赛相关

复现环境：

对于word2vec_item.py：
Python 3.8.8,
numpy==1.19.2,
gensim==4.1.2

对于run_mmoe.py:
Python==3.6.9,
gensim==4.1.2,
numpy==1.18.5,
deepctr==0.9.0,
tensorflow==1.15.5

先运行word2vec_item.py生成item embedding向量之后，在mmoe中将feedid对应的embedding向量拼接上去。最后运行run_mmoe.py文件，训练mmoe。
感觉我在面试时关于冷启动的embedding初始化部分没有讲清楚，这里再进一步说明一下:
在word2vec模型训练完毕之后，如果测试集又来了一个新的user或者item，要想得到新的Embedding，就必须把这个新的user/item加到网络中去，这就意味着你要更改输入向量的维度，这进一步意味着你要重新训练整个神经网络。但是，由于Embedding层的训练往往是整个网络中参数最大，速度最慢的。所以一般对于测试集中新出现的视频或者用户embedding向量不会重新训练。
但是如果测试集新出现的视频的embedding初始化为-1肯定效果不好。此时embedding初始化的方法为：利用视频的多模态feed embedding信息，找到测试集新出现的feedid在多模态embedding中最相似的top 10个feedid，利用这10个feed的word2vec embedding向量取平均作为新出现的视频的embedding。
