<div align="center">

# Small Chat Language Model 0.2B  

[中文](./README.md) | English

</div>
 
# 1. Introduction 
SCLM (Small Chat Language Model) is a Chinese conversation model trained from scratch. The model parameters are only 0.2B (about 210M in shared weights), and can be pre-trained on machines with a minimum of 4GB of video memory ('batch_size=1', 'fp16' or 'bf16'), and 'float16' only needs at least 512MB of video memory for loading and inference.

The project implements a complete training process of generative language models, including: data cleaning, tokenizer training, model pre-training, SFT instruction fine-tuning, RLHF optimization, etc. 

**RECENT UPDATE**

<details close> 
<summary> <b>2024-01-31</b> </summary>
- The project is open source, and the model weights are open for download. <br/>
</details>

# 2. SCLM-0.2B model training process 

## 2.1 Dataset
All datasets are from the single-round conversation dataset published on the Internet, and are cleaned and formatted and saved as parquet files.

### 2.1.1 Pretrained dataset

Datasets used include: 

1. Community Q&A JSON version webtext2019zh - large-scale high-quality dataset, see: [nlp_Chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus).
2. baike_qa2019 encyclopedia questions and answers, see: [baike_qa2019] (https://aistudio.baidu.com/datasetdetail/107726).
3. For the Q&A dataset in the field of Chinese medicine, see: [Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data).
4. For Zhihu Q&A data, see: [Zhihu-KOL] (https://huggingface.co/datasets/wangrui6/Zhihu-KOL).
5. Belle open source instruction training data, introduction: [BELLE] (https://github.com/LianjiaTech/BELLE), download: [BelleGroup] (https://huggingface.co/BelleGroup), select "Belle_open_source_1M", "train_2M_CN" part of the answers are shorter, Data that does not contain complex table structures or translation tasks.
6. Wikipedia entry data, piecing together the entries as prompts, the first 'N' words of the encyclopedia are answers, using the encyclopedia data of '202310'. Wiki download: [zhwiki](https://dumps.wikimedia.org/zhwiki/), convert the downloaded bz2 file to wiki.txt reference: [WikiExtractor](https://github.com/apertium/WikiExtractor). 

Data examples:
```json
{
    "prompt": "对于花园街，你有什么了解或看法吗？",
    "response": "花园街（是香港油尖旺区的一条富有特色的街道，位于九龙旺角东部，北至界限街，南至登打士街，与通菜街及洗衣街等街道平行。现时这条街道是香港著名的购物区之一。位于亚皆老街以南的一段花园街，也就是\"波鞋街\"整条街约150米长，有50多间售卖运动鞋和运动用品的店舖。旺角道至太子道西一段则为排档区，售卖成衣、蔬菜和水果等。花园街一共分成三段。明清时代，花园街是芒角村栽种花卉的地方。此外，根据历史专家郑宝鸿的考证：花园街曾是1910年代东方殷琴拿烟厂的花园。纵火案。自2005年起，花园街一带最少发生5宗纵火案，当中4宗涉及排档起火。2010年。2010年12月6日，花园街222号一个卖鞋的排档于凌晨5时许首先起火，浓烟涌往旁边住宅大厦，消防接报4"
}
```
### 2.1.2 SFT fine-tuning datasets

Datasets used include: 
1. Belle's open-source instruction training data, introduction: [BELLE] (https://github.com/LianjiaTech/BELLE), download: [BelleGroup] (https://huggingface.co/BelleGroup), select [generated_chat_0.4M]( https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)、[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)。

Data examples:
```json
{
    "prompt": "解释什么是欧洲启示录",
    "response": "欧洲启示录（The Book of Revelation）是新约圣经的最后一卷书，也被称为《启示录》、《默示录》或《约翰默示录》。这本书从宗教的角度描述了世界末日的来临，以及上帝对世界的审判和拯救。 书中的主题包括来临的基督的荣耀，上帝对人性的惩罚和拯救，以及魔鬼和邪恶力量的存在。欧洲启示录是一个充满象征和暗示的文本，对于解读和理解有许多不同的方法和观点。"
}
```

### 2.1.3 DPO Optimization Dataset
1. Dataset: [alpaca-gpt4-data-en](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh), the fine-tuned 'chosen' text comes from the dataset, and the reject text 'rejected' comes from the model output after SFT fine-tuning 1 epoch
2. Dataset: [huozi_rlhf_data_json] (https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json)
3. Dataset: [rlhf-reward-single-round-trans_Chinese] (https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)

Data examples:
```json
    {
        "prompt": "为给定的产品创建一个创意标语。，输入：可重复使用的水瓶。",
        "chosen": "\"保护地球，从拥有可重复使用的水瓶开始！\"",
        "rejected": "\"让你的水瓶成为你的生活伴侣，使用可重复使用的水瓶，让你的水瓶成为你的伙伴\""
    }
```

## 2.2 Model

T5 model (Text-to-Text Transfer Transformer), see the paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).

The source code of the model is from huggingface, see: [T5ForConditionalGeneration] (https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1557).

See [model_config.json] (https://huggingface.co/charent/ChatLM-mini-Chinese/blob/main/config.json) for model configuration, the official 'T5-base': 'encoder layer' and 'decoder layer' are both 12 layers, and these two parameters in this project are modified to 10 layers. 

Model parameters: 0.2B.

## 2.3 Training Process
Hardware:
```bash
CPU: 28 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
RAM: 128 GB
Graphics card: NVIDIA GeForce RTX 4090 Ti 24GB*1
```
1. Generate Training Data: Place the data file according to the directory structure in the training script, and then execute: scripts/make_data_train.py to generate pre-training data, tools/make_data_sft.py to generate SFT fine-tuning data, and tools/make_data_rlhf.py to generate DPO optimization data after fine-tuning model training.

2. **tokenizer training**: Execute: 'tools/make_token.py' to generate 'tonknizer', the training inventory is in the OOM problem, load 10 million pieces of data, about 100GB memory is required, and the appropriate amount of data can be selected for training according to the hardware situation.

3. **Text-to-Text Pre-training**: Execute: 'sclm/trainer_pre.py' to pre-train the model.

4. Prompt Supervised Fine-Tuning (SFT): Execute: 'sclm/trainer_sft.py' to perform SFT fine-tuning. 

5. DPO Direct Preference Optimization: Execute: 'SCLM/trainer_dpo.py' for model preference optimization. 

## 2.4 Effect Display

By default, the 'TextIteratorStreamer' of 'huggingface transformers' is used to implement streaming conversations, only 'greedy search' is supported, if you need other generation methods such as 'beam sample', please modify the 'stream_chat' parameter of 'cli_demo.py' to 'False'.

1. Console Operation
```bash
python cli_demo.py
```

2. API calls
```bash
python api_demo.py
```

Example of API call:
```bash
curl --location '127.0.0.1:8192/api/chat' \
--header 'Content-Type: application/json' \
--data '{
    "input_txt": "感冒了要怎么办"
}'
```

There is a problem: there are only more than 10 million pre-trained datasets, and the model parameters are only 0.2B.

# 3. References

If you find this project helpful, please feel free to refer to it.

```conf
@misc{paineliu2024,
    author={liu tingchao},
    title={A small chat language model with 0.2B parameters base on T5},
    year={2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/paineliu/sclm}},
}
```

# 4. Other matters
The project does not assume the risks and responsibilities for data security and public opinion caused by open source models and codes, or the risks and responsibilities arising from the misleading, abuse, dissemination and improper use of any models.

# 5. Thanks

This project refers to the ChatLM-mini-Chinese project and is modified based on this project, and I would like to express my deep gratitude.

```conf
@misc{Charent2023,
    author={Charent Chen},
    title={A small chinese chat language model with 0.2B parameters base on T5},
    year={2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/charent/ChatLM-mini-Chinese}},
}
```

