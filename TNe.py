#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# 将总氮数据提出来
TNe = rawdata.pop('TNe')


# 输入指标归一化
rawdata_stats = rawdata.describe()
train_stats = rawdata_stats.transpose()
train_stats
'''
    train_stats里面没有总氮数据，五个输入后缀为i，四个输出后缀为e，365天数据
''';


# In[2]:


def norm(x):
    return (x - train_stats['mean'])/train_stats['std']


# 归一化 可以用np函数，待修
rawdata_norm = norm(rawdata)


# 整理成RNN输入数据形式
datanum = 300  # 使用数据组数////////////////////////////////////////////////////////////////////////////////////////
lookback = 5  # 设置输入变量涵盖天数///////////////////////////////////////////////////////////////////////////////
paranum = 11  # 设置输入变量的指标个数//////////////////////////////////////////////////////////////////////////////
data = pd.DataFrame(columns=['input', 'TNe'])  # 建立新的数据矩阵
parastr = ['volume', 'CODi', 'SSi', 'NH3-Ni', 'TPi', 'TNi', 'CODe', 'SSe', 'NH3Ne', 'TPe', 'T']


# 可以简化，列表生成器
for i in range(datanum):
    Inputlist = []
    for j in range(lookback):
        inputlist = []
        for k in range(paranum):
            # inputlist为一天数据，11 para的顺序
            inputlist.append(rawdata_norm.loc[i+j][parastr[k]])
        # Inputlist为五天数据
        Inputlist.append(inputlist)
    # 每一行为五天数据向量，输出为五天后总氮输出
    data.loc[i] = [Inputlist, TNe[i+lookback]]

# 300 rows x 2 cols
data


# In[3]:


# 分割训练数据和测试数据
fraction = 0.8  # 定义训练集、测试集切割比例//////////////////////////////////////////////////////////////////////////
train_data = data.sample(frac=fraction)  # frac=0.8 means train data possess 80% of all
# print(len(train_data))
test_data = data.drop(train_data.index)
# print(len(test_data))
# 生成训练数据和测试数据，训练数据是240*5*7的数组  x7???
trainnum = int(datanum * fraction)
testnum = datanum - trainnum
# traindata(240,5,11)
traindata = np.zeros((trainnum, lookback, paranum))
# testdata(60,5,11)
testdata = np.zeros((testnum, lookback, paranum))

display(train_data)
display(test_data)
display(test_data.shape)


# In[4]:


for i in range(trainnum):
    for j in range(lookback):
        for k in range(paranum):
            traindata[i][j][k] = train_data.iloc[i, 0][j][k]
for i in range(testnum):
    for j in range(lookback):
        for k in range(paranum):
            testdata[i][j][k] = test_data.iloc[i, 0][j][k]

# (240,5,11)
display(traindata.shape)
# (60,5,11)
display(testdata.shape)


# In[6]:


# 搭建神经网络
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
from keras import regularizers
'''Simple RNN'''
# model.add(layers.SimpleRNN(32, input_shape=(lookback, paranum), activation='sigmoid'))
# model.add(layers.SimpleRNN(32, input_shape=(lookback, paranum), activation='sigmoid', dropout=0.2))
'''LSTM'''
# model.add(layers.LSTM(32, input_shape=(lookback, paranum), activation='sigmoid', dropout=0.2))
# model.add(layers.LSTM(32, input_shape=(lookback, paranum), activation='sigmoid'))
'''GRU'''
# model.add(layers.GRU(16, input_shape=(lookback, paranum), activation='sigmoid',
#                      kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.GRU(32, input_shape=(lookback, paranum), activation='sigmoid', dropout=0.2))
# model.add(layers.GRU(32, input_shape=(lookback, paranum), activation='sigmoid'))
'''带dropout用来削减过拟合的GRU'''
# model.add(layers.GRU(16,
#                      dropout=0.2,
#                      recurrent_dropout=0.2,
#                      input_shape=(lookback, paranum),
#                      activation='sigmoid'))
'''双层LSTM'''
# model.add(layers.LSTM(32, input_shape=(lookback, paranum), activation='sigmoid', return_sequences=True))
# model.add(layers.LSTM(32, activation='sigmoid'))
'''带dropout用来削减过拟合的双层GRU'''
# model.add(layers.GRU(32,
#                      dropout=0.1,
#                      activation='sigmoid',
#                      return_sequences=True,
#                      input_shape=(lookback, paranum)))
# model.add(layers.GRU(32, activation='sigmoid',
#                      dropout=0.1))
'''无dropout双层GRU'''
model.add(layers.GRU(32,
                     activation='sigmoid',
                     return_sequences=True,
                     input_shape=(lookback, paranum)))
model.add(layers.GRU(32, activation='sigmoid'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(0.01), loss='mae')
print(model.summary())
history = model.fit(traindata, train_data['TNe'],
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
true_value = test_data['TNe']
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
plt.ylabel('TNeffluent')

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
print('当前网络结构预测的TN出水MRE:' + str(Mre))
print('训练集的真值平均值为：' + str(np.mean(true_value)))

'''
    当前网络结构预测的TN出水MRE:0.04312339966536906
    训练集的真值平均值为：8.36679166666667
'''


# In[ ]:




