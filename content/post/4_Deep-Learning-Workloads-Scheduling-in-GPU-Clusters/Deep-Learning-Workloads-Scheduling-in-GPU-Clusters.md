---
title: Deep Learning Workloads Scheduling in GPU Clusters
date: 2023-04-08 21:24:04
tags:
- 分布式训练
- 任务调度优化
- 弹性训练
categories: 
- 论文总结
---

本文是笔者近期在组会中分享的工作汇总PPT。该PPT是对大规模高性能GPU集群环境中深度学习训练(Deep Learning Training，DLT)任务的研究意义和难点的总结与分析。

<!--more-->

该PPT主要包括研究背景、现状总结和未来发展三个方面的内容。

## Introduction

<ul>
<li>DLT任务是一种计算密集型任务，非常依赖昂贵的硬件设施；DL训练是一个不断试错的过程，所以集群的高效调度对于DLT任务来说非常重要</li>
<li>讨论现有的分布式机器学习系统中是如何管理DLT任务并调度资源的</li>
<li>现有集群环境中调度DLT任务存在的一些问题，或者说因此产生的一些研究动机</li>
<li>对上述问题的原因分析：主要原因就是DLT任务不同于以往大数据任务的一些独特特征</li>
<li>对调度DLT任务存在的挑战进行分析与总结</li>
<li>以上：我们要为DLT任务量身定制一个具有效益的且高效的集群调度器</li>
</ul>


## Existing Works

从调度目标出发，对现有的调度工作进行分类。主要可以分为：降低任务完成时间(&提升资源利用率)、减少能耗、保障公平性、保证deadline。

## Future Work

笔者在最后给出了一些思考，包括研究目标、需要考虑的因素和技术手段等三个方面。


>该PPT的部分内容参考了[[Gandiva-OSDI'18]](https://www.usenix.org/sites/default/files/conference/protected-files/osdi18_slides_sivathanu.pdf)，[[Tiresias-NSDI'19]](https://www.usenix.org/sites/default/files/conference/protected-files/nsdi19_slides_gu.pdf)，[[Pollux-OSDI'21]](https://www.usenix.org/system/files/osdi21_slides_qiao.pdf)等工作的相关公开汇报PPT。


## At the End 

笔者认为，一篇好的论文主要包括两个方面：一是如何讲述好一个故事；二是如何从实验里体现故事的价值。

因此，笔者在做这个PPT的时候，主要目的是给组会的其他同学讲清楚这项研究的意义和难点。从笔者角度来说，认为这份PPT已经完成了这个要求。


***Maybe: Start Your Own Work By Imitating The Good Ones!***

***
PPT的具体内容如下：

<embed src="./Deep-Learning-Workloads-Scheduling-in-GPU-Clusters.pdf" width="100%" height="750" type="application/pdf">


