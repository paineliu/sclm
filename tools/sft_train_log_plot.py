import sys
sys.path.extend(['.', '..'])

from datetime import datetime
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def plot_csv_train_loss(log_file: str, cols, pic_save_to_file: str=None) -> None:
    
    # ,loss,learning_rate,epoch,step,train_runtime,train_samples_per_second,train_steps_per_second,total_flos,train_loss
    # ,loss,learning_rate,rewards/chosen,rewards/rejected,rewards/accuracies,rewards/margins,logps/rejected,logps/chosen,logits/rejected,logits/chosen,epoch,step,train_runtime,train_samples_per_second,train_steps_per_second,total_flos,train_loss

    loss_list = []
    with open(log_file, 'r', encoding='utf-8') as f:

        for line in f:
            if line[0] != ',':
                line = line.split(',')
                try:
                    epoch = int(float(line[cols[0]]))  # 'epoch:0,'
                    step = int(line[cols[1]])   # 'step:0,'
                    loss = float(line[cols[2]])   # 'loss:0.11086619377136231\n'
                    loss_list.append([epoch, step, loss])
                except:
                    pass
    df = pd.DataFrame(loss_list, columns=['epoch', 'step', 'loss'])
    
    # 多项式拟合
    x = list(range(0, len(df['loss'])))
    x_range = np.arange(0, len(df['loss']), step=0.005)
    fit3 = np.polyfit(x, df['loss'], 3)
    p1d = np.poly1d(fit3)
    y_fit = p1d(x_range)

    plt.figure(figsize=(8, 6),dpi=100)
    plt.plot(df['loss'],'g',label = 'loss')
    # plt.plot(x_range, y_fit, 'r', label='fit loss')     
    plt.ylabel('loss')
    plt.xlabel('sampling step')
    plt.legend()        #个性化图例（颜色、形状等）
    
    if pic_save_to_file is not None:
        plt.savefig(pic_save_to_file) 
    
    plt.show()

if __name__ == '__main__':
    
    plot_csv_train_loss('./logs/sft_train_log_20240226-1621.csv', cols=[3,4,1], pic_save_to_file='./img/train_sft_loss.png')
    plot_csv_train_loss('./logs/dpo_train_log_20240227-1234.csv', cols=[11,12,1], pic_save_to_file='./img/train_dpo_loss.png')
 