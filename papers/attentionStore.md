# AttentionStore - Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving

![alt text](png/attentionStore/image.png)

Huawei Cloud, Chengdu

## Motivation

Multi-turn Conversation

$$
q_{1}a_{1}q_{2}a_{2}\dots q_{n-1}a_{n-1}q_{n}
$$

In ShareGPT dataset, 73% of the conversations are multi-turn. 30% of conversations have more than 4K tokens 

![alt text](png/attentionStore/image-1.png)

![alt text](png/attentionStore/image-2.png)

> we observe that if the KV caches can be reused across multiple turns of conversations, up to **98%** of prefilling cost can be reduced

## AttentionStore Architecture

![alt text](png/attentionStore/image-3.png)

![alt text](png/attentionStore/image-4.png)

## Evaluation

```
4 NVIDIA A100 GPUs, each with 80GB GPU memory
128 GB DRAM and 10 TB SSDs.
```

PyTorch

Compared with recomputation(RE)

![alt text](png/attentionStore/image-5.png)

![alt text](png/attentionStore/image-6.png)