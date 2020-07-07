#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# 输入指标归一化
rawdata_stats = rawdata.describe()
train_stats = rawdata_stats.transpose()

def norm(x):
    return (x - train_stats['mean'])/train_stats['std']


# 归一化 可以用np函数，待修
data = norm(rawdata)


# 数据集分类

fraction = 0.8  # 定义训练集、测试集切割比例//////////////////////////////////////////////////////////////////////////
train_data = data.sample(frac=fraction, random_state=0)  # frac=0.8 means train data possess 80% of all
# print(len(train_data))
test_data = data.drop(train_data.index)

fig = plt.figure(figsize=(12,40))
ax1 = fig.add_subplot(511)
plt.scatter(x = data['CODi'],y = data['CODe'])
ax2 = fig.add_subplot(512)
plt.scatter(x = data['SSi'],y = data['SSe'])
ax3 = fig.add_subplot(513)
plt.scatter(x = data['NH3-Ni'],y = data['NH3Ne'])
ax4 = fig.add_subplot(514)
plt.scatter(x = data['TPi'],y = data['TPe'])
ax5 = fig.add_subplot(515)
plt.scatter(x = data['TNi'],y = data['TNe']);



# In[58]:


from sklearn.svm import SVR


def result(data):
    train_data = data.sample(frac=0.8)
    test_data = data[~data.index.isin(train_data.index)]
    # display(data)
    # 5输出index iloc[6,7,8,9,10]
    mre_list = []
    for i in range(6,11):
        # 5输入 5输出拟合
        in_df = train_data.loc[:,['CODi', 'SSi', 'NH3-Ni', 'TPi', 'TNi']].values
        out_df = train_data.iloc[:,i].values
        cate = str(data.columns.values[i])
        # display(cate)

        # 1输出 真实值
        test_x = test_data.loc[:,['CODi', 'SSi', 'NH3-Ni', 'TPi', 'TNi']].values
        test_y = test_data.iloc[:,[i]].values
        svc=SVR().fit(in_df,out_df)

        pred_y = svc.predict(test_x)
        mre = np.mean(abs(test_y-pred_y)/test_y)
        # print(cate+"mre:")
        # print(mre)
    
        # plt.scatter(test_y,pred_y)
        mre_list.append(mre)
    return mre_list


# In[59]:


data.columns.values[6]


# In[63]:


mre_list = []
for i in range(20):
    mre_list.append(result(data))
mre_list=np.array(mre_list)


plt.figure()
plt.subplot(511)
plt.plot(np.arange(20),mre_list[:,0])

plt.figure()
plt.subplot(512)
plt.plot(np.arange(20),mre_list[:,1])

plt.figure()
plt.subplot(513)
plt.plot(np.arange(20),mre_list[:,2])

plt.figure()
plt.subplot(514)
plt.plot(np.arange(20),mre_list[:,3])

plt.figure()
plt.subplot(515)
plt.plot(np.arange(20),mre_list[:,4])


# In[ ]:




