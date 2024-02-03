<div align="center">

# 中文对话模型 0.2B

中文  | [English](./README.en.md)  

</div>
 
# 一、介绍 
SCLM(Small Chat Language Model)是一个从头开始训练的中文对话模型。模型参数只有0.2B（算共享权重约210M），可以在最低4GB显存的机器进行预训练（`batch_size=1`，`fp16`或者` bf16`），`float16`加载、推理最少只需要512MB显存。

项目实现了生成式语言模型的完整训练流程，包括：数据清洗、tokenizer训练、模型预训练、SFT指令微调、RLHF优化等。 

**最近更新**

<details close> 
<summary> <b>2024-01-31</b> </summary>
- 项目开源， 开放模型权重供下载。 <br/>
</details>


# 二、SCLM-0.2B模型训练过程 

## 2.1 数据集
所有数据集均来自互联网公开的**单轮对话**数据集，经过数据清洗、格式化后保存为parquet文件。

### 2.1.1 预训练数据集

使用的数据集包括： 

1. 社区问答json版webtext2019zh-大规模高质量数据集，见：[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)。
2. baike_qa2019百科类问答，见：[baike_qa2019](https://aistudio.baidu.com/datasetdetail/107726)。
3. 中国医药领域问答数据集，见：[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)。
4. 知乎问答数据，见：[Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)。
5. belle开源的指令训练数据，介绍：[BELLE](https://github.com/LianjiaTech/BELLE)，下载：[BelleGroup](https://huggingface.co/BelleGroup)，选取`Belle_open_source_1M`、`train_2M_CN`中部分回答较短、不含复杂表格结构、翻译任务的数据。
6. 维基百科（Wikipedia）词条数据，将词条拼凑为提示语，百科的前`N`个词为回答，使用`202310`的百科数据。Wiki下载：[zhwiki](https://dumps.wikimedia.org/zhwiki/)，将下载的bz2文件转换为wiki.txt参考：[WikiExtractor](https://github.com/apertium/WikiExtractor)。 

数据示例：
```json
{
    "prompt": "对于花园街，你有什么了解或看法吗？",
    "response": "花园街（是香港油尖旺区的一条富有特色的街道，位于九龙旺角东部，北至界限街，南至登打士街，与通菜街及洗衣街等街道平行。现时这条街道是香港著名的购物区之一。位于亚皆老街以南的一段花园街，也就是\"波鞋街\"整条街约150米长，有50多间售卖运动鞋和运动用品的店舖。旺角道至太子道西一段则为排档区，售卖成衣、蔬菜和水果等。花园街一共分成三段。明清时代，花园街是芒角村栽种花卉的地方。此外，根据历史专家郑宝鸿的考证：花园街曾是1910年代东方殷琴拿烟厂的花园。纵火案。自2005年起，花园街一带最少发生5宗纵火案，当中4宗涉及排档起火。2010年。2010年12月6日，花园街222号一个卖鞋的排档于凌晨5时许首先起火，浓烟涌往旁边住宅大厦，消防接报4"
}
```
### 2.1.2 SFT微调数据集

使用的数据集包括： 
1. belle开源的指令训练数据，介绍：[BELLE](https://github.com/LianjiaTech/BELLE)，下载：[BelleGroup](https://huggingface.co/BelleGroup)，选取[generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)、[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)。

数据示例：
```json
{
    "prompt": "解释什么是欧洲启示录",
    "response": "欧洲启示录（The Book of Revelation）是新约圣经的最后一卷书，也被称为《启示录》、《默示录》或《约翰默示录》。这本书从宗教的角度描述了世界末日的来临，以及上帝对世界的审判和拯救。 书中的主题包括来临的基督的荣耀，上帝对人性的惩罚和拯救，以及魔鬼和邪恶力量的存在。欧洲启示录是一个充满象征和暗示的文本，对于解读和理解有许多不同的方法和观点。"
}
```

### 2.1.3 DPO优化数据集
1. 数据集：[alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh)，微调的`chosen`文本来自数据集，拒绝文本`rejected`来自SFT微调1个epoch后的模型输出
2. 数据集：[huozi_rlhf_data_json](https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json)
3. 数据集：[rlhf-reward-single-round-trans_chinese](https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)

数据示例：
```json
    {
        "prompt": "为给定的产品创建一个创意标语。，输入：可重复使用的水瓶。",
        "chosen": "\"保护地球，从拥有可重复使用的水瓶开始！\"",
        "rejected": "\"让你的水瓶成为你的生活伴侣，使用可重复使用的水瓶，让你的水瓶成为你的伙伴\""
    }
```

## 2.2 模型

T5模型（Text-to-Text Transfer Transformer），详情见论文: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)。

模型源码来自huggingface，见：[T5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1557)。

模型配置见[model_config.json](https://huggingface.co/charent/ChatLM-mini-Chinese/blob/main/config.json)，官方的`T5-base`：`encoder layer`和`decoder layer `均为为12层，本项目这两个参数修改为10层。 

模型参数：0.2B。

## 2.3 训练过程
硬件：
```bash
CPU: 28 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
内存：128 GB
显卡：NVIDIA GeForce RTX 4090 Ti 24GB * 1
```
1. **生成训练数据**： 将数据文件按照训练脚本中的目录结构放置，然后执行：`scripts/make_data_train.py`生成预训练数据；执行`tools/make_data_sft.py`生成SFT微调数据，在微调模型训练后，执行`tools/make_data_dpo.py`生成DPO优化数据。

2. **tokenizer 训练**： 执行：`tools/make_token.py`生成`tonknizer`，训练库存在OOM问题，加载1000万条数据，大约需要100GB内存，可以根据硬件情况，选取合适数量的数据进行训练。

3. **Text-to-Text 预训练**：执行：`sclm/trainer_pre.py`进行模型预训练。

4. **prompt监督微调（SFT）**：执行：`sclm/trainer_sft.py`进行SFT微调。 

5. **dpo直接偏好优化**：执行：`sclm/trainer_dpo.py`进行模型偏好优化。 

## 2.4 效果展示

默认使用`huggingface transformers`的 `TextIteratorStreamer`实现流式对话，只支持`greedy search`，如果需要`beam sample`等其他生成方式，请将`cli_demo.py`的`stream_chat`参数修改为`False`。

1. 控制台运行：
```bash
python cli_demo.py
```

2. API调用
```bash
python api_demo.py
```

API调用示例：
```bash
curl --location '127.0.0.1:8192/api/chat' \
--header 'Content-Type: application/json' \
--data '{
    "input_txt": "感冒了要怎么办"
}'
```

存在问题：预训练数据集只有1000多万条，模型参数也仅0.2B，会有答非所问、废话生成器的情况。

# 三、引用

如果你觉得本项目对你有所帮助，欢迎引用。
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

# 四、其他事项
本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。

# 五、感谢

本项目参考了[ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)项目，并基于这个项目修改，在此表示深深的谢意。
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

