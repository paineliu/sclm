import ujson
import re
import os
import time
from collections import defaultdict

from matplotlib import pyplot as plt
import codecs, csv
import pandas as pd 
import numpy as np
from rich import progress
from rich.table import Table
from rich.console import Console
from fastparquet import ParquetFile, write
import pyarrow.parquet as pq
from opencc import OpenCC
import sys
sys.path.extend(['.','..'])

from lib.logger import Logger

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence

def get_sentences_dice_similarity(st_a: str, st_b: str) -> float:
    '''
    获取两个句子的Dice相似度（Dice similarity）
    s(a, b) =  2 * len( set(a) & set(b) ) / (len(set(a)) + len(set(b)))
    '''
    set_a, set_b = set(st_a), set(st_b)
    total_len  = len(set_a) + len(set_b)
    
    if total_len == 0: return 0.0

    inter_set =  set_a & set_b
    
    return ( 2 * len(inter_set)) / total_len

def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    write(file_name, data_frame, compression='GZIP', append=exists(file_name))

def read_and_write_template(read_file: str, write_to_file: str, call_back: object, group_cnt: int=10000) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    group_cnt: parquet file分割行数
    '''

    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    
    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []

        for line in f_read:
            try:
                raw_line_cnt += 1

                write_dict = call_back(line)

                if write_dict is None: continue

                keep_line_cnt += 1
                cur_rows.append(write_dict)

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows.clear()

            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e
        
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
    
    end = time.time()
    return (raw_line_cnt, keep_line_cnt, end -start)

def process_data_files(data_pathname, save_filename, recreate, data_filenames, process_function):
    log_items = []
    save_log_filename = save_filename + ".log"    

    log_items.append('{} -> {}'.format(data_pathname, save_filename))
    log.info(log_items[-1], save_to_file=True)
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    if recreate:
        if os.path.isfile(save_log_filename):
            os.remove(save_log_filename)

    if os.path.isfile(save_log_filename):
        log_items.append('{} skip'.format(data_pathname))
        log.info(log_items[-1], save_to_file=True)
        return False

    if os.path.isfile(save_filename):
        os.remove(save_filename)

    for data_filename in data_filenames:

        log_items.append('{}'.format(data_filename))
        log.info(log_items[-1], save_to_file=True)

        raw_line_cnt, keep_line_cnt, duration = read_and_write_template(data_filename, save_filename, process_function)
        log_items.append('total line = {}, remain line = {}, time cost = {:.2f}s'.format(raw_line_cnt, keep_line_cnt, duration))
        log.info(log_items[-1], save_to_file=True)

    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(data_pathname), save_to_file=True)

def process_web_text_zh(data_pathname, save_filename, recreate=False, keep_start: int=5, response_less_word: int=10) -> None:
    '''
    处理425万社区问答webtext2019zh知识类数据集
    keep_start: 只保留点赞数大于keep_start的问答
    response_less_word: 答案至少要有response_less_word个字
    '''
    data_filenames = [
        os.path.join(data_pathname, 'web_text_zh_test.json'),
        # os.path.join(data_pathname, 'web_text_zh_train.json'),
        os.path.join(data_pathname, 'web_text_zh_valid.json'),
    ]

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if item['star'] < keep_start or len(item['content']) < response_less_word: 
            return None

        # 数据清洗
        # 去除重复的标点符号
        prompt = remove_duplicate_punctuation(item['title'])
        response = remove_duplicate_punctuation(item['content'])
        write_dict = {
            "prompt": prompt,
            "response": response,
        }
        return write_dict
    
    process_data_files(data_pathname, save_filename, recreate, data_filenames, process_function)
    
def process_baike_qa(data_pathname, save_filename, recreate=False, response_less_word: int=15) -> None:
    '''
    处理147万百度知道知识类数据集
    '''
    data_filenames = [
        os.path.join(data_pathname, 'baike_qa_train.json'),
        os.path.join(data_pathname, 'baike_qa_valid.json'),
    ]

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if len(item['answer']) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item['title'], item['desc']) >= 0.90:
            # title 和desc 相似度过高，只用title作为问题
            prompt = item['title']
        else:
            # title 和desc拼接形成问题
            prompt = "{}{}".format(item['title'], item['desc'])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = item['answer'].replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < 3 or len(response) < response_less_word:
            return None
        
        write_dict = {
                "prompt": prompt,
                "response": response,
            }

        return write_dict

    process_data_files(data_pathname, save_filename, recreate, data_filenames, process_function)

def process_belle_knowledge(data_pathname, save_filename, recreate=False, response_less_words: int=15, group_cnt: int=10000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    data_filenames = [
        os.path.join(data_pathname, 'Belle_open_source_1M.json'),
        os.path.join(data_pathname, 'train_2M_CN.json'),
    ]

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if '翻译' in prompt or 'translate' in prompt.lower():
            return None
        
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(response) < response_less_words:
            return None
        
        prompt = remove_duplicate_punctuation(prompt)
        response = remove_duplicate_punctuation(response)

        if len(response) < response_less_words:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }

        return write_dict
    
    process_data_files(data_pathname, save_filename, recreate, data_filenames, process_function)

def process_belle_knowledge_finetune(data_pathname, save_filename, recreate=False, max_len: int=320, group_cnt: int=50000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    data_filenames = [
        os.path.join(data_pathname, 'Belle_open_source_0.5M.json'),
        os.path.join(data_pathname, 'generated_chat_0.4M.json'),
    ]

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if 'translate' in prompt.lower(): return None
        for word in ('翻译', '英译', '译英', '中译',  '译中', '汉译', '译汉'):
            if word in prompt:
                return None
        
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(prompt) > max_len or len(response) > max_len:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }

        return write_dict

    process_data_files(data_pathname, save_filename, recreate, data_filenames, process_function)

def csv_gbk_utf8_file(gbk_filename: str, utf8_filename: str) -> None:
    '''
    修复csv文件，将文件中换行符替换为\n，字段中的英文字符替换为中文字符
    '''
    with codecs.open(gbk_filename, 'r', encoding='gbk', errors='ignore') as f_gbk:
        reader = csv.reader(f_gbk)
        new_lines = []

        for line in reader:
            for i in range(len(line)):
                line[i] = line[i].replace('\n', '\\n') # 处理异常的换行符
                line[i] = line[i].replace(',', '，') # 英文逗号换为中文逗号
            new_lines.append(line)

        with open(utf8_filename, 'w', encoding='utf-8', newline="") as f_utf8:
            writer = csv.writer(f_utf8)
            writer.writerows(new_lines)

def process_chinese_medical(gbk_data_pathname, utf8_data_pathname, save_filename, recreate=False, response_less_word: int=15) -> None:
    '''
    处理中国医药领域问答数据集
    '''
    def process_function(line: str) -> dict:
        # department,title,ask,answer
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[3]) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item[1], item[2]) >= 0.90:
            # title 和ask 相似度过高，只用ask作为问题
            prompt = item[2]
        else:
            # title 和 ask 拼接形成问题
            prompt = "{}{}".format(item[1], item[2])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = ''.join(item[3: ]).replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < 3 or len(response) < response_less_word:
            return None
        
        write_dict = {
                "prompt": prompt,
                "response": response,
            }

        return write_dict
    
    data_filenames = []
    for root,dirs,files in os.walk(gbk_data_pathname):
        for file in files:
            if file.endswith("csv"):
                gbk_filename = os.path.join(root, file)

                utf8_filename = os.path.join(utf8_data_pathname, gbk_filename[len(gbk_data_pathname) + 1:])
                data_filenames.append(utf8_filename)
                os.makedirs(os.path.dirname(utf8_filename), exist_ok=True)
                log_message = '{} -> {}'.format(gbk_filename, utf8_filename)
                log.info(log_message, save_to_file=True)
                if recreate or not os.path.isfile(utf8_filename):
                    start = time.time()
                    csv_gbk_utf8_file(gbk_filename, utf8_filename)
                    end = time.time()
                    duration = end - start
                    log_message = 'time cost = {:.2f}s'.format(duration)
                    log.info(log_message, save_to_file=True)
                else:
                    log.info('skip', save_to_file=True)

    process_data_files(gbk_data_pathname, save_filename, recreate, data_filenames, process_function)

def process_zhihu_kol(data_pathname, save_filename, recreate=False, prompt_less_word: int=4, response_less_word: int=10, group_cnt: int=10000) -> None:
    '''
    处理知乎数据集
    '''
    file_names = [
        os.path.join(data_pathname, 'train-00000-of-00005-a1278ede4e8c5cdb.parquet'),
        os.path.join(data_pathname, 'train-00001-of-00005-1fc2da944397e9c2.parquet'),
        os.path.join(data_pathname, 'train-00002-of-00005-68ced004a1458143.parquet'),
        os.path.join(data_pathname, 'train-00003-of-00005-1dae36b67c12169f.parquet'),
        os.path.join(data_pathname, 'train-00004-of-00005-c374cc9fbda9fda7.parquet')
    ]   
    
    def process_function(sentence: str) -> str:
        '''
        针对一个句子的数据清洗
        '''
        # 删除\r
        sentence = sentence.replace('\r','') 

        # 删除重复的标点符号
        sentence = remove_duplicate_punctuation(sentence)

        return sentence

    log_items = []
    save_log_filename = save_filename + ".log"    

    log_items.append('{} -> {}'.format(data_pathname, save_filename))
    log.info(log_items[-1], save_to_file=True)
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    if recreate:
        if os.path.isfile(save_log_filename):
            os.remove(save_log_filename)

    if os.path.isfile(save_log_filename):
        log_items.append('{} skip'.format(data_pathname))
        log.info(log_items[-1], save_to_file=True)
        return False

    if os.path.isfile(save_filename):
        os.remove(save_filename)

    cur_rows = []
    for data_filename in file_names:
        
        raw_line_cnt, keep_line_cnt = 0, 0
        start = time.time()

        pf = pq.read_table(data_filename)
        log_items.append('{}'.format(data_filename))
        log.info(log_items[-1], save_to_file=True)

        for prompt, response in progress.track(zip(pf['INSTRUCTION'], pf['RESPONSE']), total=pf.num_rows):
            raw_line_cnt += 1
            prompt, response = prompt.as_py(), response.as_py()
            
            prompt = process_function(prompt)
            response = process_function(response)

            if len(prompt) < prompt_less_word or len(response) < response_less_word:
                continue
            
            keep_line_cnt += 1
            write_dict = {
                'prompt': prompt,
                'response': response,
            }
            cur_rows.append(write_dict)

            if len(cur_rows) >= group_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_filename, df)
                cur_rows.clear()
                
        end = time.time()
        duration = end - start
        log_items.append('total line = {}, remain line = {}, time cost = {:.2f}s'.format(raw_line_cnt, keep_line_cnt, duration))
        log.info(log_items[-1], save_to_file=True)        

    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_filename, df)
        cur_rows = []

    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(data_pathname), save_to_file=True)

def process_zh_wiki(data_filename, save_filename, txt_filename, recreate=False, groups_cnt: int=10000, max_len: int=512, seed: int=23333) -> None:
    '''
    将Wiki中文数转换为问答数据集
    wiki 下载地址：https://dumps.wikimedia.org/zhwiki/
    将下载的bz2文件转换为wiki.txt参考：https://github.com/apertium/WikiExtractor
    '''

    # 将繁体转换为简体
    cc = OpenCC('t2s')
    
    # 构造问题的前缀
    prompt_prefix = [
        '什么是{}？',
        '介绍一下{}',
        '介绍一下什么是{}',
        '写一篇关于{}的介绍',
        '{}是什么？',
        '你知道{}吗？',
        '生成关于{}的介绍',
        '我想知道关于{}的详细信息',
        '你了解{}吗？',
        '请解释一下{}',
        '对于{}，你有什么了解或看法吗？',
        '请告诉我关于{}的信息',
        '请简要描述一下{}',
        '请提供有关{}的一些详细信息',
        '能否解释一下{}是什么?',
        '请分享一些关于{}的背景知识',
        '请简要概括一下{}',
        '能给我一些关于{}的背景资料吗?',
        '有关{}的信息可以分享一下吗？',
        '你能告诉我{}是什么吗？',
    ]

    def process_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
        
        line = convert_en_punctuation_to_zh_punct(line) # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line
        
    np.random.seed(seed)
    log_items = []
    save_log_filename = save_filename + ".log"
    if not recreate and os.path.isfile(save_log_filename):
        log.info('{} skip'.format(data_filename), save_to_file=True)
        return
    
    log_items.append('{} -> {}'.format(data_filename, save_filename))
    log.info(log_items[-1], save_to_file=True)

    f_txt = open(txt_filename, 'w', encoding='utf-8')

    raw_line_cnt, keep_line_cnt = 0, 0
    start = time.time()

    log_items.append('{}'.format(data_filename))
    log.info(log_items[-1], save_to_file=True)

    with progress.open(data_filename, 'r', encoding='utf-8') as read_file:
        prompt = '' 
        response = '' 
        pre_line_len = 0
        cur_rows = []
        txt_rows = []
        for line_raw in read_file:
            raw_line_cnt += 1

            line = process_line(line_raw)
            if len(line.strip()) == 0:
                txt_rows.append(line.strip())
            else:
                txt_rows.append(line)
            line = line.strip()

            if len(txt_rows) >= groups_cnt:
                f_txt.write('\n'.join(txt_rows))
                txt_rows.clear()

            # 确定问题行，上一行是空行，则当前行是新的百科词条，设置为prompt
            if line.endswith('：') and pre_line_len == 0:
                if len(prompt) > 0 and len(response) > 0:
                    cur_rows.append({'prompt': prompt, 'response': ''.join(response)})
                    prompt = ''
                    response = ''
                prompt = np.random.choice(prompt_prefix).format(line[0: -1])
                response = ''
            else:
                if len(prompt) > 0 and len(line) > 0 and len(response) + len(line) + 1 < max_len:
                        response += line + '\n'
                        keep_line_cnt += 1

            pre_line_len = len(line_raw.strip())
            
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_filename, df)
                cur_rows.clear()

        if len(prompt) > 0 and len(response) > 0:
            cur_rows.append({'prompt': prompt, 'response': ''.join(response)})

        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_filename, df)

        if len(txt_rows) > 0:
            f_txt.write('\n'.join(txt_rows))
            txt_rows.clear()

    end = time.time()
    duration = end - start
    log_items.append('total line = {}, remain line = {}, time cost = {:.2f}s'.format(raw_line_cnt, keep_line_cnt, duration))
    log.info(log_items[-1], save_to_file=True)
    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(data_filename), save_to_file=True)

def merge_dataset(data_pathname, save_filename, recreate=False, groups_cnt: int=50000, max_len: int=512, min_len: int=3, cut_max_len: bool=False) -> None:
    '''
    将多个数据集合并为一个数据集
    '''
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)

    log_items = []
    save_log_filename = save_filename + ".log"    
    log_items.append('{} -> {}'.format(data_pathname, save_filename))
    log.info(log_items[-1], save_to_file=True)

    if not recreate and os.path.isfile(save_log_filename):
        log.info('{} skip'.format(data_pathname), save_to_file=True)
        return
    
    if os.path.isfile(save_filename):
        os.remove(save_filename)

    data_filenames = []
    for root,dirs,files in os.walk(data_pathname):
        for file in files:
            if file.endswith("parquet"):
                filename = os.path.join(root, file)
                data_filenames.append(filename)

    cur_rows = []

    raw_line_cnt, keep_line_cnt = 0, 0

    start = time.time()
    for file in data_filenames:
        log_items.append('{}'.format(file))
        log.info(log_items[-1], save_to_file=True)

        parquet_table = pq.read_table(file)     
        for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):
            prompt, response = prompt.as_py(), response.as_py()
            raw_line_cnt += 1
            if len(prompt) < min_len or len(response) < min_len:
                continue
            if cut_max_len and (len(prompt) > max_len or len(response) > max_len):
                prompt = prompt[0: max_len]
                response = response[0: max_len]

            keep_line_cnt += 1
            cur_rows.append({'prompt': prompt , 'response': response})

            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_filename, df)
                cur_rows.clear()
       
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_filename, df)

    end = time.time()
    duration = end - start
    log_items.append('total line = {}, remain line = {}, time cost = {:.2f}s'.format(raw_line_cnt, keep_line_cnt, duration))
    log.info(log_items[-1], save_to_file=True)
    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(data_pathname), save_to_file=True)

def shuffle_dataset(data_filename: str, save_filename: str, recreate=False, seed: int=23333, groups_cnt: int=65536) -> None:
    '''
    打乱一个parquet文件数据集
    '''
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)

    log_items = []
    save_log_filename = save_filename + ".log"    
    log_items.append('{} -> {}'.format(data_filename, save_filename))
    log.info(log_items[-1], save_to_file=True)

    if not recreate and os.path.isfile(save_log_filename):
        log.info('{} skip'.format(data_filename), save_to_file=True)
        return
    
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    
    start = time.time()
    
    pf =  pq.read_table(data_filename)
    df = pf.to_pandas()
    df = df.sample(frac=1.0, replace=False, random_state=seed, axis=0)
        
    # 分块写入parquet，否则小内存读取直接OOM
    n = len(df)
    for i in range(0, n, groups_cnt):
        cur_group_df = df[i: i + groups_cnt]
        write_single_parquet_file(save_filename, cur_group_df)
    
    end = time.time()
    duration = end - start
    log_items.append('time cost = {:.2f}s'.format(duration))
    log.info(log_items[-1], save_to_file=True)
    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(data_filename), save_to_file=True)

def split_datasets(source_parquet_file: str, recreate=False, max_len: int=320, seed: int=23333, train_ratio: float=0.91, test_ratio: float=0.0875, valid_ratio: float=0.0025, groups_cnt: int=50000) -> None:
    '''
    将原始数据拆分为训练集、测试集和验证集
    '''
    assert train_ratio + test_ratio + valid_ratio == 1.0

    log_items = []
    save_log_filename = source_parquet_file + "_split.log"    
    log_items.append('{}'.format(source_parquet_file))
    log.info(log_items[-1], save_to_file=True)

    if not recreate and os.path.isfile(save_log_filename):
        log.info('{} skip'.format(source_parquet_file), save_to_file=True)
        return
        
    start = time.time()
    
 
    train_parquet_file = source_parquet_file[:-8] + '_train.parquet'
    test_parquet_file = source_parquet_file[:-8] + '_test.parquet'
    valid_parquet_file =  source_parquet_file[:-8] + '_valid.parquet'
    if os.path.isfile(train_parquet_file):
        os.remove(train_parquet_file)
    if os.path.isfile(test_parquet_file):
        os.remove(test_parquet_file)
    if os.path.isfile(valid_parquet_file):
        os.remove(valid_parquet_file)

    np.random.seed(seed)

    start = time.time()

    train, test, valid = [], [], []

    parquet_table =  pq.read_table(source_parquet_file)

    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):
        
        prompt, response = prompt.as_py(), response.as_py()
        rand = np.random.random()
        cur_data = {'prompt': ''.join(prompt[0: max_len]) , 'response': ''.join(response[0: max_len])}

        if 0 <= rand < train_ratio:
            train.append(cur_data)
        elif train_ratio <= rand  < train_ratio + test_ratio:
            test.append(cur_data)
        else:
            valid.append(cur_data)
        
        if len(train) >= groups_cnt:
            write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
            train = []
        
        if len(test) >= groups_cnt:
            write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
            test = []
        
        if len(valid) >= groups_cnt:
            write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
            valid = []
                

    if len(train) > 0:
        write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
        train = []
    
    if len(test) > 0:
        write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
        test = []
    
    if len(valid) > 0:
        write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
        valid = []

    end = time.time()
    duration = end - start
    log_items.append('time cost = {:.2f}s'.format(duration))
    log.info(log_items[-1], save_to_file=True)
    f = open(save_log_filename, 'w', encoding='utf-8')
    f.writelines('\n'.join(log_items))
    f.close()

    log.info('{} success'.format(source_parquet_file), save_to_file=True)

def parquet_to_text(pq_filename, txt_filename, sep='[SEP]', buffer_size: int=50000) -> None:
    '''
    将parquet文件转换为txt预料，句子之间用sep隔开
    txt文件用于训练tokenizer，使用huggingface的BPE训练会导致OOM
    '''
    os.makedirs(os.path.dirname(txt_filename), exist_ok=True)

    source_pf = ParquetFile(pq_filename)
    cur_rows = []
    append = cur_rows.append
    with open(txt_filename, 'w', encoding='utf-8') as f_write:
        for pf_chunk in progress.track(source_pf):
            for rows in pf_chunk.iter_row_groups():
                for prompt, response in zip(rows['prompt'], rows['response']):
                    append(prompt + sep + response + sep + '\n')

                    if len(cur_rows) >= buffer_size:
                        f_write.writelines(cur_rows)
                        cur_rows = []
                        append = cur_rows.append
                       
        # end for
        if len(cur_rows) > 0:
            f_write.writelines(cur_rows)
            cur_rows = []

def parquet_to_json(pq_filename, json_filename) -> None:
    '''
    将parquet文件转换为json
    '''
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    source_pf = ParquetFile(pq_filename)
    cur_rows = []
   
    for pf_chunk in progress.track(source_pf):
        for rows in pf_chunk.iter_row_groups():
            for prompt, response in zip(rows['prompt'], rows['response']):
                if len(response) == 0 or len(prompt) == 0: continue
                cur_rows.append({
                    'prompt': str(prompt),
                    'response': str(response),
                })

    with open(json_filename, 'w', encoding='utf-8') as f:
        ujson.dump(cur_rows, f, indent=4, ensure_ascii=False)

def stat_parquet_data_length(pq_filename, img_filename) -> None:

    os.makedirs(os.path.dirname(img_filename), exist_ok=True)

    parquet_table = pq.read_table(pq_filename)

    que_len_dict, ans_len_dict = defaultdict(int), defaultdict(int)
    
    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']), total=parquet_table.num_rows):

        prompt, response = prompt.as_py(), response.as_py()

        que_len_dict[len(prompt)] += 1
        ans_len_dict[len(response)] += 1

    que_len, ans_len = [], []
    for k, v in que_len_dict.items():
        que_len.append([k, v])
    for k, v in ans_len_dict.items():
        ans_len.append([k, v])

    def gather_gt_x(array: list[tuple], x: int=512) -> list:
        '''
        长度大于x的合并在一起
        '''
        new_array = []
        gt_x_cnt = 0
        for item in array:
            if item[0] < x:
                new_array.append([item[0], item[1]])
            else:
                gt_x_cnt += item[1]
        new_array.append([x, gt_x_cnt])

        return new_array
    
    max_len = 512
    ans_list = gather_gt_x(ans_len, max_len)
    ans_list.sort(key=lambda x: x[0])
    que_list = gather_gt_x(que_len, max_len)
    que_list.sort(key=lambda x: x[0])
    
    ans_pd = pd.DataFrame(ans_list, columns=['length', 'count'])
    que_pd = pd.DataFrame(que_list, columns=['length', 'count'])

    def plot_sub_bar(plt, x, y, title: str, color: str='g') ->None:
        plt.bar(x, y, color=color, label='sample count')
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
        plt.legend()
        plt.xlabel('length')
        plt.ylabel('count')
        plt.title(title)

    plt.figure(figsize=(10, 10),dpi=200)
    plt.subplot(2, 2, 1)
    plot_sub_bar(plt, que_pd['length'], que_pd['count'], title='prompt length', color='c')

    plt.subplot(2, 2, 2)
    plot_sub_bar(plt, ans_pd['length'], ans_pd['count'], title='response length', color='g')

    le512_pd = ans_pd[ans_pd['length'] < 512]
    plt.subplot(2, 2, 3)
    plot_sub_bar(plt, le512_pd['length'], le512_pd['count'], title='response length < 512', color='limegreen')

    le320_pd = ans_pd[ans_pd['length'] < 320]
    plt.subplot(2, 2, 4)
    plot_sub_bar(plt, le320_pd['length'], le320_pd['count'], title='response length < 320', color='limegreen')

    plt.savefig(img_filename)
    plt.show()

def stat_parquet_data_lines(parquet_file: str=None) -> None:
    '''
    统计parquet数据集数据量
    '''
    data_filenames = []

    if os.path.isdir(parquet_file):
        for root,dirs,files in os.walk(parquet_file):
            for file in files:
                if file.endswith(".parquet"):
                    data_filename = os.path.join(root, file)
                    data_filenames.append(data_filename)
    else:
        data_filenames = [parquet_file]

    result = [['file_name', 'count']]
    raw_line_cnt = 0
    for file in data_filenames:
        file_name = file.split('/')[-1]
        cur_cnt = 0
        pf = ParquetFile(file)

        for pf_chunk in pf:
            cur_cnt += pf_chunk.info['rows']
        
        raw_line_cnt += cur_cnt
        result.append([file_name, cur_cnt])
    
    result.append(['total', raw_line_cnt])

    log.info(str(result), save_to_file=True)

    console = Console()
    table = Table(show_header=True, show_lines=True,)

    for col in result[0]:
        table.add_column(col)
    for i in range(1, len(result)): # 跳过表头
        table.add_row(str(result[i][0]), str(result[i][1]))

    console.print(table)    


if __name__ == '__main__':

    log = Logger('data_process', save2file=True, file_name='./logs/raw_data_process.log')
    recreate = True

    # =================================================================
    # data process
    process_chinese_medical('./data/raw/chinese_medical', './data/raw/chinese_medical_utf8', './data/parquet/chinese_medical_uft8.parquet', recreate=recreate, response_less_word=15)
    process_web_text_zh('./data/raw/web_text_zh', './data/parquet/web_text_zh.parquet', recreate=recreate, keep_start=5, response_less_word=15)
    process_baike_qa('./data/raw/baike_qa', './data/parquet/baike_qa.parquet', recreate=recreate, response_less_word=15)
    process_zhihu_kol('./data/raw/zhihu_kol','./data/parquet/zhihu_kol.parquet', recreate, prompt_less_word=4, response_less_word=10)
    process_belle_knowledge('./data/raw/belle_knowledge', './data/parquet/belle_knowledge.parquet', recreate=recreate, response_less_words=5)
    process_zh_wiki('./data/raw/zhwiki/wiki.txt', './data/parquet/zhwiki_cn.parquet', './data/raw/zhwiki/zhwiki_cn.txt', recreate=recreate, groups_cnt=10000, max_len=512)
    process_belle_knowledge_finetune('./data/raw/belle_knowledge', './data/result/belle_knowledge_finetune.parquet', recreate=recreate, max_len=320, group_cnt=50000)

    # =================================================================
    # dataset
    merge_dataset('./data/parquet', './data/result/dataset_all.parquet', recreate=recreate, groups_cnt=50000, min_len=3, max_len=512, cut_max_len=True)
    shuffle_dataset('./data/result/dataset_all.parquet', './data/result/dataset_shuffle.parquet', recreate=recreate, seed=23333)
    split_datasets('./data/result/dataset_shuffle.parquet', recreate=recreate, max_len=320, groups_cnt=50000)

    # =================================================================
    # convert
    parquet_to_text('./data/result/dataset_shuffle.parquet', './data/text/dataset_shuffle.txt')
    parquet_to_json('./data/result/dataset_shuffle.parquet', './data/text/dataset_shuffle.json')
    # stat
    stat_parquet_data_lines('./data/parquet')
    # stat_parquet_data_length('./data/result/dataset_shuffle.parquet', './data/img/dataset_sentence_length.png')

