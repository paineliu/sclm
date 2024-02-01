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
from scripts.make_data_train import process_belle_knowledge_finetune_sft, parquet_to_json

from chatbot import Logger

def make_data_sft():
    global log
    log = Logger('make_data_sft', save2file=True, file_name='./logs/make_data_sft.log')
    recreate = False
    process_belle_knowledge_finetune_sft('./data/raw/belle_knowledge', './data/result/cbot_dataset_sft.parquet', recreate=recreate, max_len=320, group_cnt=50000)
    parquet_to_json('./data/result/cbot_dataset_sft.parquet', './data/result/cbot_dataset_sft.json')


if __name__ == '__main__':

    make_data_sft()
