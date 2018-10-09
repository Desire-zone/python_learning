'''
2018-09-21
试跑知乎大佬的代码
'''


import numpy as np
import pandas as pd
import time
import datetime
import gc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb



# 加载数据
train = pd.read_table('./data/round1_iflyad_train.txt')
test = pd.read_table('./data/round1_iflyad_test_feature.txt')

# 合并训练集，验证集
data = pd.concat([train,test],axis=0,ignore_index=True)

# 缺失值填充
data['make'] = data['make'].fillna(str(-1))
data['model'] = data['model'].fillna(str(-1))
data['osv'] = data['osv'].fillna(str(-1))
data['app_cate_id'] = data['app_cate_id'].fillna(-1)
data['app_id'] = data['app_id'].fillna(-1)
data['click'] = data['click'].fillna(-1)
data['user_tags'] = data['user_tags'].fillna(str(-1))
data['f_channel'] = data['f_channel'].fillna(str(-1))

# replace
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink', 'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])

# labelencoder 转化
encoder = ['city', 'province', 'make', 'model', 'osv', 'os_name', 'adid', 'advert_id', 'orderid',
           'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id',
           'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf']

#简化编码
col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])

#one-hot
encoder_onehot = ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'creative_has_deeplink','creative_is_jump',
               'creative_is_download','creative_is_js','creative_is_voicead' ]

onehot_encoder = OneHotEncoder(sparse=False)
for feat in encoder_onehot:
    onehot_encoder.fit(data[feat].values.reshape((-1,1)))
    enc = onehot_encoder.transform(data[feat].values.reshape((-1,1)))
    tmp=pd.DataFrame(enc,columns=[feat]*len(enc[0]))
    data.drop(feat, axis=1, inplace=True)
    data.join(tmp,how='outer')

#将秒数转换为{年，月，日，时，分，秒，周，年中第几天，是否夏令时}
#然后取出日和时
data['day'] = data['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))

# 历史点击率
# 时间转换，将5.27到6.3号转化为序列{27,28,29,30,31,32,33,34}，增加为period列
data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31



# 删除没用的特征
drop = ['click', 'time', 'instance_id', 'user_tags',
        'app_paid']

train = data[:train.shape[0]]
test = data[train.shape[0]:]

y_train = train.loc[:,'click']
res = test.loc[:, ['instance_id']]
#print(y_train.shape)  (1001650,)

train.drop(drop, axis=1, inplace=True)
#print('train:',train.shape)   (1001650, 40)
test.drop(drop, axis=1, inplace=True)
#print('test:',test.shape)     (40024, 40)

#转化为数组
X_loc_train = train.values
y_loc_train = y_train.values
X_loc_test = test.values



# 模型部分
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)

# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    #训练
    print("Fold", i)
    lgb_model = model.fit(X_loc_train[train_index], y_loc_train[train_index],
                          eval_names =['train','valid'],
                          eval_metric='logloss',
                          eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
                                    (X_loc_train[test_index], y_loc_train[test_index])],early_stopping_rounds=100)
    #训练损失
    baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
    loss += lgb_model.best_score_['valid']['binary_logloss']

    #测试准确率
    test_pred= lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss/5)

# 加权平均
res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]
res['predicted_score'] = res['predicted_score']/5

# 提交结果
mean = res['predicted_score'].mean()
print('mean:',mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
res[['instance_id', 'predicted_score']].to_csv("./result_sub/lgb_baseline_%s.csv" % now, index=False)



