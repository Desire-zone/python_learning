import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'prize', 'class label']

col_encoder = LabelEncoder()
col_encoder.fit(df['color'])
df['color'] = col_encoder.transform(df['color'])

print(df)


onehot_encoder = OneHotEncoder(sparse=False)

#print(onehot_encoder.fit(df['color'].values))

onehot_encoder.fit(df['color'].values.reshape((-1,1)))

enc=onehot_encoder.transform(df['color'].values.reshape((-1,1)))

tempdata = pd.DataFrame(enc,columns=['house']*len(enc[0]))

print(tempdata)

print(df.join(tempdata,how='outer'))








