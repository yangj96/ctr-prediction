import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections as cs

path='/data/johnao/ctr'
train_data=pd.read_csv(path+'/train_data_1119.csv')
test_data=pd.read_csv(path+'/test_data_1119.csv')

# 填充缺失值
test_data.uuid.fillna(value=test_data.uuid[0],inplace=True)
test_data.request_cate_id.fillna(value=test_data.request_cate_id[0],inplace=True)
test_data.gender.fillna(value=0.,inplace=True)
test_data.age.fillna(value=test_data.age.mean(),inplace=True)
test_data.job.fillna(value=0.,inplace=True)
test_data.avg_price.fillna(value=test_data.avg_price.mean(),inplace=True)
test_data.poi_star.fillna(value=test_data.poi_star.mean(),inplace=True)
test_data.request_time_second.fillna(value=test_data.request_time_second.mean(),inplace=True)
test_data.poi_avg_discount.fillna(value=test_data.poi_avg_discount.mean(),inplace=True)
test_data.poi_cnt_deal.fillna(value=int(test_data.poi_cnt_deal.mean()),inplace=True)

train_data.gender.fillna(value=0.,inplace=True)
train_data.age.fillna(value=train_data.age.mean(),inplace=True)
train_data.avg_price.fillna(value=train_data.avg_price.mean(),inplace=True)
train_data.poi_star.fillna(value=train_data.poi_star.mean(),inplace=True)
train_data.poi_avg_discount.fillna(value=train_data.poi_avg_discount.mean(),inplace=True)
train_data.poi_cnt_deal.fillna(value=int(train_data.poi_cnt_deal.mean()),inplace=True)

# 离散类型数据
disc=['poi_id','request_cate_id','device_type','gender','job',
      'cate_level1','cate_level2','cate_level3','area_id']

# 连续类型数据
cont=['latitude_req','longitude_req','age','avg_price','poi_star',
      'longitude_poi','latitude_poi','distance','poi_cnt_deal','poi_avg_discount',
      'request_time_second']

# 选取需要使用的数据
train_p=train_data.loc[:,disc+cont+['action']]
test_p=test_data.loc[:,disc+cont]
for i in disc:
    train_p.loc[:,i]=train_p.loc[:,i].apply(np.int)
    test_p.loc[:,i]=test_p.loc[:,i].apply(np.int)

# 离散值按频率重新标号
cate_dict=[]
for i in range(len(disc)):
    d=cs.defaultdict(int)
    for x in train_p.iloc[:,i]:
        d[x]+=1
    d=list(d.items())
    d.sort(key=lambda x:-x[1])
    cate_dict.append(cs.defaultdict(int,{x:i for i,x in enumerate(x for x,_ in d)}))

for i in range(len(disc)):
    train_p.iloc[:,i]=train_p.iloc[:,i].apply(cate_dict[i].get)

# 连续值归一化
mv=[]
for i in cont:
    col=train_p.loc[:,i]
    mv.append([col.mean(),np.sqrt(col.var())])
    train_p.loc[:,i]=(col-mv[-1][0])/mv[-1][1]

# 对测试集做相同的操作
for i in range(len(disc)):
    test_p.iloc[:,i]=test_p.iloc[:,i].apply(lambda x:cate_dict[i][x])
for i,j in zip(cont,mv):
    test_p.loc[:,i]=(test_p.loc[:,i]-j[0])/j[1]

# 保存处理好的数据
train_p.to_csv(path+'/train_data_1120.csv')
test_p.to_csv(path+'/test_data_1120.csv')