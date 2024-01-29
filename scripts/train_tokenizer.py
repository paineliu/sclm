from fastparquet import ParquetFile
import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace
from tokenizers.normalizers import NFKC 
from multiprocessing import RLock, Pool
from multiprocessing.managers import BaseManager
import os 
from collections import defaultdict

def train_hf_bpe_tokenizer(corpus_filename, token_filename, max_train_line: int=None) -> None:
    '''
    训练tokenizer with huggingface，至少需要32G内存，运行大概需要半个小时。
    '''
    os.makedirs(os.path.dirname(token_filename), exist_ok=True)

    def get_training_corpus(buffer_size: int=1000, chunk_len: int=2048) -> list:
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

                if isinstance(max_train_line, int) and line_cnt > max_train_line: break
                
            # yield last
            if len(buffer) > 0: yield buffer        

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

if __name__ == '__main__':

    train_hf_bpe_tokenizer('./data/raw/zhwiki/zhwiki_cn.txt', './output/hf_bpe_tokenizer.json')
