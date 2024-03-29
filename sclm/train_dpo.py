import os
import sys
sys.path.append(os.path.dirname(__file__))

from typing import Dict, Optional
import time
import os 

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, TaskType, PeftModel

from config import DpoConfig, T5ModelConfig
from model import TextToTextModel
from transformers import T5Config

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_T5_config(config: T5ModelConfig, vocab_size: int, decoder_start_token_id: int=0, eos_token_id: int=1) -> T5Config:
    '''
    用户配置转换为T5Config
    '''
    t5_config = T5Config()
    # t5_config.model_type = 'TextToTextModel'
    # 初始化
    t5_config.d_ff = config.d_ff
    t5_config.d_kv = config.d_kv
    t5_config.d_model = config.d_model
    t5_config.num_decoder_layers = config.num_decoder_layers
    t5_config.num_heads = config.num_heads
    t5_config.num_layers = config.num_layers
    t5_config.vocab_size = vocab_size
    t5_config.decoder_start_token_id = decoder_start_token_id
    t5_config.eos_token_id = eos_token_id

    return t5_config

def get_dataset(split: str, file: str, cache_dir: str = '.cache') -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    dataset = load_dataset('json', data_files=file,  split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(sample: dict) -> Dict[str, str]:
        return {
            # add an eos token for signal that end of sentence, using in generate.
            "prompt": f"{sample['prompt']}[EOS]",
            "chosen": f"{sample['chosen']}[EOS]",
            "rejected": f"{sample['rejected']}[EOS]",
        }

    return dataset.map(split_prompt_and_responses).shuffle(2333)


def train_dpo(config: DpoConfig, peft_config: LoraConfig=None) -> None:

    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    
    # step 2. 加载预训练模型
    model_train, model_ref = None, None 
    if os.path.isdir(config.sft_model_file):
        # 传入文件夹则 from_pretrained
        model_train = TextToTextModel.from_pretrained(config.sft_model_file)
        model_ref = TextToTextModel.from_pretrained(config.sft_model_file)
    else:
        # load_state_dict
        t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

        model_train = TextToTextModel(t5_config)
        model_train.load_state_dict(torch.load(config.sft_model_file, map_location='cpu')) # set cpu for no exception

        model_ref = TextToTextModel(t5_config)
        model_ref.load_state_dict(torch.load(config.sft_model_file, map_location='cpu'))
    
    # 4. 加载训练数据集
    train_dataset = get_dataset("train", file=config.dpo_train_file)

    # 5. 加载评估数据集
    # eval_dataset = get_dataset("train", file=config.dpo_eval_file)
    eval_dataset = None

    # 6. 初始化训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_first_step=True,
        logging_steps=config.logging_steps, 
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        optim="adafactor",
        report_to="tensorboard",
        log_level='info',
        warmup_steps=config.warmup_steps,
        bf16=False,
        fp16=config.fp16,
        seed=config.seed,
        logging_dir=config.log_dir,
    )

    # 7. 初始化 DPO trainer
    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,
        args=training_args,
        beta=config.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        max_target_length=config.max_seq_len,
        max_prompt_length=config.max_seq_len,
        generate_during_eval=True,
        is_encoder_decoder=True,
    )

    # 8. 训练
    dpo_trainer.train(
        # resume_from_checkpoint=True
    )

    # 9. save log
    loss_log = pd.DataFrame(dpo_trainer.state.log_history)
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    loss_log.to_csv(f"{log_dir}/dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    
    # 10. 保存模型/lora
    suffixe = '/lora/' if peft_config is not None else '/dpo'
    model_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1]) + suffixe

    dpo_trainer.save_model(model_save_dir)
    print('save model or lora adapter to: {}'.format(model_save_dir))

def merge_lora_weight_into_model(config: DpoConfig, peft_config: LoraConfig) -> None:

    # step 1. 加载tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    
    # step 2. 加载预训练模型
    sft_model = None
    if os.path.isdir(config.sft_model_file):
        # 传入文件夹则 from_pretrained
        sft_model = TextToTextModel.from_pretrained(config.sft_model_file)
    else:
        # load_state_dict
        t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        sft_model = TextToTextModel(t5_config)
        sft_model.load_state_dict(torch.load(config.sft_model_file, map_location='cpu')) # set cpu for no exception
        
    # 注意这个路径要和上面的model_save_dir一致
    # train_dpo函数代码
        # 9. 保存模型/lora
        # suffixe = '/lora/' if peft_config is not None else '/dpo'
        # model_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1]) + suffixe

    adapter_save_dir = '/'.join(config.sft_model_file.split('/')[0: -1]) + '/lora'
    
    # peft_model = PeftModel.from_pretrained(
    #     model=sft_model,
    #     # model_id=adapter_save_dir,
    #     # model_id="sclm/modellora",
    #     config=peft_config,
    #     adapter_name='adapter',
    # )
    
    peft_model = PeftModel(
        model=sft_model,
        peft_config=peft_config,
        adapter_name='adapter',
    )

    # 3. load adapter
    
    print('load adapter from dir: {}'.format(adapter_save_dir))

    peft_model.load_adapter(model_id=adapter_save_dir, adapter_name='adapter',)

    # 4. merge
    peft_model = peft_model.merge_and_unload()
    
    # 5. save
    save_merge_file = config.sft_model_file + '.dpo_lora_merged'
    sft_model.save_pretrained(save_merge_file)
    print('save merge model file to: {}'.format(save_merge_file))

   
if __name__ == "__main__":

    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model 
         inference_mode=False, 
         r=16, 
         lora_alpha=16, 
         lora_dropout=0.1, 
         bias="all",
    )

    dpo_config = DpoConfig()

    # 1. train
    # train_dpo(dpo_config, peft_config=None)

    # 2. merge lora adapter into model
    merge_lora_weight_into_model(dpo_config, peft_config)
    