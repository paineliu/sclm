import sys
sys.path.extend(['.','..'])

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
from tools.make_data_pre import process_belle_knowledge_finetune_sft, parquet_to_json, stat_data_line_total
import os
import shutil

from sclm import Logger

def copy_train_data_file(json_filename, train_filename):
    os.makedirs(os.path.dirname(train_filename), exist_ok=True)
    shutil.copy(json_filename, train_filename)
    
def make_data_sft():

    log = Logger('make_data_sft', save2file=True, file_name='./logs/make_data_sft' + '-' + str(time.strftime('%Y%m%d-%H%M', time.localtime())) +'.log')
    recreate = False
    process_belle_knowledge_finetune_sft('./data/raw/belle_knowledge', './data/tmp/dataset/data_sft/data_sft.parquet', log, recreate=recreate, max_len=320, group_cnt=50000)
    parquet_to_json('./data/tmp/dataset/data_sft/data_sft.parquet', './data/result/sc_data_sft.json', log, recreate=recreate)
    copy_train_data_file('./data/result/sc_data_sft.json', './data/result/sc_data_sft/train.json')
    stat_data_line_total('./data/tmp/dataset/data_sft/data_sft.parquet', log)


if __name__ == '__main__':

    make_data_sft()
