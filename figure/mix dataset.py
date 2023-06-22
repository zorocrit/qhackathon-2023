import pandas as pd
import os
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)


folder1 = os.listdir('C:/Users/Administrator/git_zorocrit/qhackathon-2023/figure/data_for_figure')

csv_fin1 = pd.DataFrame()


for files in folder1:
    csv1 = pd.read_csv('C:/Users/Administrator/git_zorocrit/qhackathon-2023/figure/data_for_figure/'+files)
    csv_fin1 = pd.concat([csv_fin1, csv1], axis=0)

# print(csv_fin1)
csv_fin1.to_csv('C:/Users/Administrator/git_zorocrit/qhackathon-2023/figure/fig_data_' + printdate + '.csv', index=False)
