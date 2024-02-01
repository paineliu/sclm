import sys
sys.path.extend(['.','..'])

import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace
from tokenizers.normalizers import NFKC 
from multiprocessing import RLock, Pool
from multiprocessing.managers import BaseManager
import os 
import time

from chatbot import Logger

def train_hf_bpe_tokenizer(corpus_filename, token_filename, pretrained_token_pathname, recreate=False, max_train_line: int=None) -> None:
    '''
    训练tokenizer with huggingface 1000万条数据大约需要100G内存。
    '''
    os.makedirs(os.path.dirname(token_filename), exist_ok=True)
    log_items = []
    save_log_filename = token_filename + ".log"    

    log_items.append('{} -> {}'.format(corpus_filename, token_filename))
    log.info(log_items[-1], save_to_file=True)
    if recreate:
        if os.path.isfile(save_log_filename):
            os.remove(save_log_filename)

    if os.path.isfile(save_log_filename):
        log_items.append('{} skip'.format(corpus_filename))
        log.info(log_items[-1], save_to_file=True)
        return False

    def get_training_corpus(buffer_size: int=1000, chunk_len: int=2048) -> list: # type: ignore
        '''
        一个文本块大小2048
        '''
        line_cnt = 0
        buffer = []
        with open(corpus_filename, 'r', encoding='utf-8') as f_read:
            cur_chunk_txt, txt_len = [], 0
            for line in f_read:

                cur_chunk_txt.append(line)
                txt_len += len(line)
                line_cnt += 1

                if txt_len >= chunk_len:
                    buffer.append(
                        ''.join(cur_chunk_txt)
                    )
                    cur_chunk_txt, txt_len = [], 0
                
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

                if isinstance(max_train_line, int) and line_cnt > max_train_line:
                    break
                
            # yield last
            if len(buffer) > 0: yield buffer        

    start = time.time()
    model = BPE(unk_token="[UNK]")

    tokenizer = Tokenizer(model)
    
    special_tokens = ["[PAD]","[EOS]","[SEP]","[BOS]", "[CLS]", "[MASK]", "[UNK]"]

    # 用兼容等价分解合并对utf编码进行等价组合，比如全角A转换为半角A
    tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])

    # 标点符号，数字，及Metaspace预分割（否则decode出来没有空格）
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [Punctuation(), Digits(individual_digits=True), Metaspace()]
    )

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.decoder = decoders.Metaspace()

    trainer = BpeTrainer(vocab_size=40960, min_frequency=100, show_progress=True, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # add \t \n 
    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\t'])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\n'])

    tokenizer.save(token_filename)
    
    trained_tokenizer_to_pretrained_tokenizer_fast(token_filename, pretrained_token_pathname)

    end = time.time()
    duration = end - start
    log_items.append('time cost = {:.2f}s'.format(duration))
    log.info(log_items[-1], save_to_file=True)        

    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(corpus_filename), save_to_file=True)

def trained_tokenizer_to_pretrained_tokenizer_fast(trained_token_filename, pretrained_token_pathname):
    '''
    将Tokenizer转换为 PreTrainedTokenizerFast
    '''
    from transformers import PreTrainedTokenizerFast

    tokenizer_obj = Tokenizer.from_file(trained_token_filename)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer_obj.token_to_id('[PAD]')
    tokenizer.unk_token = '[UNK]'
    tokenizer.unk_token_id = tokenizer_obj.token_to_id('[UNK]')
    tokenizer.eos_token = '[EOS]'
    tokenizer.eos_token_id = tokenizer_obj.token_to_id('[EOS]')

    tokenizer.save_pretrained(pretrained_token_pathname)

def make_token():
    global log
    log = Logger('make_token', save2file=True, file_name= './logs/make_token.log')
    train_hf_bpe_tokenizer('./data/result/cbot_dataset.txt', './data/result/hf_bpe_tokenizer.json', './output/tokenizer', max_train_line=10000000)

if __name__ == '__main__':

    make_token()