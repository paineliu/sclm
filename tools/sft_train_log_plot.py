import sys
sys.path.extend(['.', '..'])

from datetime import datetime
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def str_to_timestamp(string: str) -> float:
    '''
    '''
    date_fmt = '%Y-%m-%d %H:%M:%S.%f'
    string = string.replace('[', '').replace(']', '')

    # 转化为时间戳
    return datetime.strptime(string, date_fmt).timestamp()

def plot_train_loss(log_file: str, pic_save_to_file: str=None) -> None:
    '''
    将log日志中记录的画图，按需保存到文件，由于log日志打印内容较多，需要指定要打印loss的开始时间和结束时间
    '''
    
    # ,loss,learning_rate,epoch,step,train_runtime,train_samples_per_second,train_steps_per_second,total_flos,train_loss
    loss_list = []
    with open(log_file, 'r', encoding='utf-8') as f:

        for line in f:
            if line[0] != ',':
                line = line.split(',')

                if len(line) != 10: continue
 
                epoch = line[5][6: -1]  # 'epoch:0,'
                step = line[6][5: -1]   # 'step:0,'
                loss = float(line[7][5: -1])   # 'loss:0.11086619377136231\n'
                device = line[8][7: -1]
                loss_list.append([epoch, step, loss, device])
    
    df = pd.DataFrame(loss_list, columns=['epoch', 'step', 'loss', 'device'])
    
    # 多项式拟合
    x = list(range(0, len(df['loss'])))
    x_range = np.arange(0, len(df['loss']), step=0.005)
    fit3 = np.polyfit(x, df['loss'], 3)
    p1d = np.poly1d(fit3)
    y_fit = p1d(x_range)

    plt.figure(figsize=(8, 6),dpi=100)
    plt.plot(df['loss'],'g',label = 'loss')
    plt.plot(x_range, y_fit, 'r', label='fit loss')     
    plt.ylabel('loss')
    plt.xlabel('sampling step')
    plt.legend()        #个性化图例（颜色、形状等）
    
    if pic_save_to_file is not None:
        plt.savefig(pic_save_to_file) 
    
    plt.show()


if __name__ == '__main__':
    
    plot_train_loss('./logs/sft_train_log_20240226-1621.csv', pic_save_to_file='./img/train_sft_loss.png')
    # plot_train_loss('./logs/chat_trainer-20231018.log', '[2023-10-18 02:06:28.137]', '[2023-10-18 18:03:35.230]', pic_save_to_file='./img/finetune_loss.png')