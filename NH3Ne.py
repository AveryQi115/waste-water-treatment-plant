#!/usr/bin/env python
# coding: utf-8

# In[2]:


# - coding:utf-8 -
'''

@author:BlazerLean
@time:2020/4/3022:35
NH3Ne预测
'''

pre_column = 'NH3Ne'
import pandas as pd
import numpy as np


# 设置打印最大行
pd.set_option('display.max_columns', 20)


# 导入数据
dataset_path = "./2019data.xlsx"
column_names = ['data', 'volume', 'CODi', 'BODi', 'SSi', 'NH3-Ni', 
                'TPi', 'TNi', 'CODe', 'BODe', 'SSe', 'NH3Ne', 'TPe',
                'TNe', 'T', 'rain']
rawdata = pd.read_excel(dataset_path, names=column_names)


# 将降雨中的空值转化为0
rawdata = rawdata.replace(np.NaN, 0)
# data = data.fillna(0)


# 删除相关性较差列数据及时间列
del rawdata['data']
del rawdata['BODi']
del rawdata['BODe']
del rawdata['rain']


# 将总数据提出来
result = rawdata.pop(pre_column)


# 输入指标归一化
rawdata_stats = rawdata.describe()
print(rawdata_stats)
train_stats = rawdata_stats.transpose()


def norm(x):
    return (x - train_stats['mean'])/train_stats['std']


rawdata_norm = norm(rawdata)


# 整理成RNN输入数据形式
datanum = 300  # 使用数据组数////////////////////////////////////////////////////////////////////////////////////////
lookback = 5  # 设置输入变量涵盖天数///////////////////////////////////////////////////////////////////////////////
paranum = 11  # 设置输入变量的指标个数//////////////////////////////////////////////////////////////////////////////
data = pd.DataFrame(columns=['input', pre_column])  # 建立新的数据矩阵
parastr = ['volume', 'CODi', 'SSi', 'NH3-Ni', 'TPi', 'TNi', 'CODe', 'SSe', 'NH3Ne', 'TPe','TNe','T']
parastr.remove(pre_column)
print(parastr)


for i in range(datanum):
    Inputlist = []
    for j in range(lookback):
        inputlist = []
        for k in range(paranum):
            inputlist.append(rawdata_norm.loc[i+j][parastr[k]])
        Inputlist.append(inputlist)
    data.loc[i] = [Inputlist, result[i+lookback]]



# 分割训练数据和测试数据
fraction = 0.8  # 定义训练集、测试集切割比例//////////////////////////////////////////////////////////////////////////
train_data = data.sample(frac=fraction)  # frac=0.8 means train data possess 80% of all
print(len(train_data))
test_data = data.drop(train_data.index)
print(len(test_data))
# 生成训练数据和测试数据，训练数据是240*5*7的数组
trainnum = int(datanum * fraction)
testnum = datanum - trainnum
traindata = np.zeros((trainnum, lookback, paranum))
testdata = np.zeros((testnum, lookback, paranum))

for i in range(trainnum):
    for j in range(lookback):
        for k in range(paranum):
            traindata[i][j][k] = train_data.iloc[i, 0][j][k]
for i in range(testnum):
    for j in range(lookback):
        for k in range(paranum):
            testdata[i][j][k] = test_data.iloc[i, 0][j][k]


# 搭建神经网络
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
from keras import regularizers

'''无dropout双层GRU'''
model.add(layers.GRU(32,
                     activation='sigmoid',
                     return_sequences=True,
                     input_shape=(lookback, paranum)))
model.add(layers.GRU(32, activation='sigmoid'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(0.01), loss='mae')
print(model.summary())
history = model.fit(traindata, train_data[pre_column],
                    epochs=400,
                    batch_size=128,
                    validation_split=0.2)
print(model.summary())

# 可视化训练过程
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylim(0, 2)
plt.legend()
display(plt.show())


#  预测出水////////////////////////////////////////////////////////////////////////////////////////////
predictdata = testdata
predictnum = testnum
predict = model.predict(predictdata).flatten()
print(predict)


# 结果可视化/////////////////////////////////////////////////////////////////////////////////////////////
plt.figure(figsize=(6, 6))
true_value = test_data[pre_column]
plt.scatter(predict, true_value)
plt.ylabel('True Values')
plt.xlabel('Predictions')
plt.legend()
display(plt.show())

# 时序折线图
plt.figure(figsize=(10, 6))
x = np.arange(0, predictnum, 1)
plt.plot(x,
         predict,
         linestyle='-',
         linewidth=2,
         color='#ff9999',
         marker=None,
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='#ff9999',
         label='predict')
plt.plot(x,
         true_value,
         linestyle='-',
         linewidth=2,
         color='steelblue',
         marker=None,
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='steelblue',
         label='actual')

# 添加标题和坐标轴标签
plt.title('2019predict')
plt.xlabel('day')
plt.ylabel(pre_column+'effluent')

# 显示图例
plt.legend()

# 剔除图框上边界和右边界的刻度
plt.tick_params(top='off', right='off')
display(plt.show())


# 计算mre/////////////////////////////////////////////////////////////////////////////////////////////////////
mre = 0
for i in range(len(true_value)):
    mre = mre + abs(true_value.iloc[i] - predict[i]) / true_value.iloc[i]
    # 这里因为true_value的列表和real_predict的列表格式不同
    #  所以计算时，使用了不同的索引级别
Mre = mre / len(true_value)
print('当前网络结构预测的'+pre_column+'出水MRE:' + str(Mre))
print('训练集的真值平均值为：' + str(np.mean(true_value)))


# In[ ]:




