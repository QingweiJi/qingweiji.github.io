---
title: Transformer Model Learning
summary: Transformer模型入门
date: 2024-10-10
math: true
author:
  - Qingwei Ji
tags:
  - LLM Inference 
  - Transformer
---

Transformer (原文[Attention is all your need](https://docalysis.s3.us-west-1.amazonaws.com/prod/sgk7dcw7kftek1a6rehs4cnslht2rj993jv8ywp8rbp7fx6vfpz4mbykgjwn7vzn.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAUUXFV6NCXUJP7F5H%2F20241014%2Fus-west-1%2Fs3%2Faws4_request&X-Amz-Date=20241014T091710Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=2deac3c9002a4b352f9402fae70c4fa1946bbea89eeb83ce87940fcb65d007ba)) 小白学习路线：

1. [台大李宏毅自注意力机制和Transformer详解](https://www.bilibili.com/video/BV1v3411r78R/?spm_id_from=333.337.search-card.all.click&vd_source=e092b13f32ccf57ffa840f905bb9dd08)
（强推！）
1. [李沐Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE/?vd_source=e092b13f32ccf57ffa840f905bb9dd08)
2. [【官方双语】GPT是什么？直观解释Transformer | 深度学习第5章](https://www.bilibili.com/video/BV13z421U7cs/?spm_id_from=333.999.0.0&vd_source=e092b13f32ccf57ffa840f905bb9dd08)


## self-attention

Self-Attention是Transformer最核心的内容。在每个自注意力模块中，对于输入的prompt，每个单词有3个不同的矩阵，分别是查询（Query $Q$）、键（Key $K$）和值（Value $V$），长度均是64。它们是通过3个不同的权值矩阵由输入矩阵$X$乘以三个不同的权值矩阵$W^Q, W^K, W^V$得到，相当于经历了一次线性变换。其中$W^Q, W^K, W^V$是三个可训练的参数矩阵，三个矩阵的尺寸都是$512 \times 64$。
得到的$Q, K, V$可以理解为：

$Q$：要查询的信息，查询

$K$：被查询的向量，索引

$V$：查询得到的值，内容

Transformer原文《Attention is all you need》中将这个Attention公式描述为：Scaled Dot-Product Attention。

$Q = XW_Q$

$K = XW_K$

$V = XW_V$

$Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_k}})$V

Attention不直接使用$X$，而是使用经过矩阵乘法生成的这三个矩阵，因为使用三个可训练的参数矩阵，可增强模型的拟合能力。

那么$Q, K, V$三个矩阵在Attention的计算中扮演什么角色呢？我们对Attention的计算方法分步骤解读。

（1）将输入单词转化为嵌入向量$X$

（2）根据嵌入向量得到$Q, K, V$三个向量

（3）计算$Q$和$K$的相似度，也就是使用点积法计算相关性得分，即用$Q$中每一个向量与$K$中每一个向量计算点积，具体到矩阵的形式：$score = Q \cdot K^T$，$score$是一个$(2,2)$的矩阵

（4）transformer对$score$进行归一化，即$score = \frac{score}{\sqrt{d_k}}$。$d_k$就是$Q, K$矩阵的列数，即向量维度。这个除法被称为Scale，当$d_k$很大时，$QK^T$的乘法结果方差变大，进行Scale可以使方差变小，训练时梯度更新更稳定。

（5）通过softmax激活函数，将对$score$转换成$[0, 1]$之间的概率分布，同时更加凸显单词之间的关系。经过softmax的归一化后，每个值是一个大于0小于1的权重系数$\alpha$，且总和为0，这个结果可以被理解成一个权重矩阵
<!-- 经过softmax后，$score$转换成一个值分布在$[0, 1]$之间的$(2, 2)\alpha$概率分布矩阵。 -->

（6）根据每个单词之间的概率分布，然后乘上对应的Values值，$\alpha$与$V$进行点积，即$Z=softmax(score) \cdot V$，得到加权的每个输入向量的评分
<!-- 其中，$v$的维度是$(2, 64)$，$(2, 2) \times (2, 64)$最后得到的$Z$是$(2,64)$维的矩阵 -->

![](./1-self-attention-output.png)


## Multi-Head Attention

为了增强拟合功能，Transformer对Attention继续扩展，提出了Multi-Head Attention。多头注意力的原理很简单，就是self-attention的基础上，对于输入的embedding矩阵，self-attention只使用了一组$W^Q, W^K, W^V$来变换得到Query， Keys， Values。而Multi-Head Attention使用多组$W^Q, W^K, W^V$得到多组Query， Keys， Values，每组分别计算生成不同的$Q, K, V$。
然后每一组分别计算得到一个加权之后的特征矩阵$Z_i$，将得到多个$Z_i$矩阵拼接（Concat）成一个大的特征矩阵$Z$，最后再用$Z$乘以$W^O$便得到了整个多头注意力层的输出。

具体的，其计算公式为：
$$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O$$
$$where\ head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)$$
其中，$W_i^Q \in \mathcal{R}^{d_{model}\times{d_k}}, W_i^K \in \mathcal{R}^{d_{model}\times{d_k}}, W_i^V \in \mathcal{R}^{d_{model}\times{d_v}}, W^O \in \mathcal{R}^{hd_v\times{d_{model}}}$。

![](./2-attention.png)

>在计算每个head的注意力后，将所有头连接在一起，并通过一个线性层($W^O$矩阵)传递它。反过来，每个头都是用query、key和value(分别为$W^Q, W^K, W^V$矩阵)的三个单独矩阵的scaled dot-product attention。这三个矩阵对于每个头都是不同的，这就是下标$i$存在的原因。


在原文中，作者使用了$h=8$个并行的自注意力模块（8个头）来构建一个注意力层，并且对于每个自注意力模块都限定了$d_k = d_v = d_{model}/h = 64$。也就是说，多头注意力机制其实就是将一个大的高维单头拆分成了$h$个多头
整个过程如下图所示：
![](./3-MHA.png)

## Add & Norm

得到特征向量$Z$后，并没有直接传入全连接神经网络FNN，而是经过了一步：Add ＆ Normalize。
![](./4-transformer_resideual_layer_norm_2.png)

Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：
$$LayerNorm(X + MultiHeadAttention(X))$$

$$LayerNorm(X + FeedForward(X))$$

其中，$X$表示 MultiHeadAttention 或 FeedForward 的输入，MultiHeadAttention($X$) 和 FeedForward($X$) 表示输出（输出和输入$X$维度是一样的，所以可以相加）

Add指$X$+MultiHeadAttention($X$)，这是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分。

Norm指Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛。


## FFN

transformer模型中的全连接层是一个两层的神经网络，第一层的激活函数是ReLU，第二层是一个线性激活函数，可以表示为：

$$FFN(Z) = \max(0, ZW_1+b1)W_2+b2$$

先线性变换，然后ReLU非线性，再线性变换。输入的$Z$就是多头注意力的输出。经过FFN后最终得到的输出矩阵的维度与输入$Z$是一致的。因为这两层网络是为了将输入的$Z$映射到更高维的空间中，然后通过非线性函数ReLU进行筛选，筛选完后再变回原来的维度。


## At the End


***Maybe: Attention is all your need!***


***

BGM：

<ul>
<li>亲爱的，那不是爱情 -- 张韶涵</li>
</ul>



# Reference

[1] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

[2] [注意力机制到底在做什么，Q/K/V怎么来的？一文读懂Attention注意力机制](https://www.zhihu.com/tardis/zm/art/414084879?source_id=1005)

[3] [This post is all you need（上卷）——层层剥开Transformer](https://zhuanlan.zhihu.com/p/420820453)

[4] [详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)

[5] [史上最小白之Transformer详解](https://zhuanlan.zhihu.com/p/697473529)

[6] [Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)

[7] [关于Transformer的那些个为什么](https://zhuanlan.zhihu.com/p/360144789)

[8] [How to Estimate the Number of Parameters in Transformer models](https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0)

[9] [手写 Self-Attention 的四重境界，从 self-attention 到 multi-head self-attention](https://bruceyuan.com/hands-on-code/from-self-attention-to-multi-head-self-attention.html)