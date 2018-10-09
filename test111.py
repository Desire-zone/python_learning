import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

train = pd.read_table('./data/round1_iflyad_train.txt')
test = pd.read_table('./data/round1_iflyad_test_feature.txt')

data = pd.concat([train,test],axis=0,ignore_index=True)
#data = data.sort_values('time')
#print(data.head())
#print(data.nunique())
data.drop('creative_is_voicead',axis=1,inplace = True)
data.drop('app_paid',axis=1,inplace=True)
data.drop('creative_is_js',axis=1,inplace=True)

jump_count = data.groupby('creative_is_jump')['click'].agg(['count']).reset_index()
download_count = data.groupby('creative_is_download')['click'].agg(['count']).reset_index()
has_deeplink_count = data.groupby('creative_has_deeplink')['click'].agg(['count']).reset_index()


data = pd.merge(data, jump_count, how='left', on='creative_is_jump')
data = pd.merge(data, download_count, how='left', on='creative_is_download')
data = pd.merge(data, has_deeplink_count, how='left', on='creative_has_deeplink')

print(data.head())