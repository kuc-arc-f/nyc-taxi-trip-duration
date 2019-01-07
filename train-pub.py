# encoding: utf-8
# titanic 問題。 通常データ train/ test.csv の使用して検証する。
# save csv
# 評価
# train : % 
# test  : %

# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
# 機械学習モジュール
import sklearn
from sklearn import linear_model
import pickle
import time 

#
def correct_data(taxi_data):
    taxi_data.store_and_fwd_flag = taxi_data.store_and_fwd_flag.replace(['N', 'Y'], [0, 1])
    return taxi_data

# 学習データ
global_start_time = time.time()
train = pd.read_csv("../input/train.csv") # Reading data
test  = pd.read_csv("../input/test.csv")  # Reading data

# 
#train= train[: 10000]
print(train.shape ,test.shape )
#quit()

#test  = pd.read_csv("../input/test.csv")  # Reading data
train = train[ train['trip_duration'] < 1000000]
# convert
train= correct_data(train)
test = correct_data(test )
#for df in (train_data,test_data):
#quit()

for df in (train, test ):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df['pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
    df['lat_diff'] = df['pickup_latitude'] - df['dropoff_latitude']
    df['lon_diff'] = df['pickup_longitude'] - df['dropoff_longitude']

col_name = ["vendor_id","passenger_count", "pickup_longitude", "pickup_latitude"
          , "dropoff_longitude", "dropoff_latitude","lat_diff","lon_diff"
           ,"pickup_dt" ]
train_sub  =train[col_name]
test_sub   =test[col_name]
x_test =test_sub
x_train =train_sub
y_train =train['trip_duration']
print( x_train.shape, y_train.shape )
print( x_test.shape )
#print("min=", train['pickup_datetime'].min() )
#print("max=", train['pickup_datetime'].max() )
#quit()

#print( train['lat_diff'][: 10])
#print( train['lon_diff'][: 10])
#train
# モデルのインスタンス
model = linear_model.LinearRegression()
# fit
clf = model.fit( x_train ,y_train)
print("train:",clf.__class__.__name__ ,clf.score(x_train,y_train))

#pred
pred = model.predict(x_train )
pred_int = np.array( pred , np.int32)
#pred = pred.astype(np.float16)
print( pred_int[: 10] )
#plt
a1=np.arange(len(x_train) )
plt.plot(a1 , y_train  , label = "y_train")
#plt.plot(a1 , y_test  , label = "y_test")
plt.plot(a1 , pred , label = "predict")
plt.legend()
plt.grid(True)
plt.title("taxi, trip duration")
plt.xlabel("x")
plt.ylabel("trip duration")
plt.show()
#quit()

# モデルを保存する
filename = 'model.pkl'
pickle.dump( model , open(filename, 'wb'))
print("model save, complete !!")
print ('time : ', time.time() - global_start_time)

quit()
print(train.info() )
print(train .shape )
pd.set_option('display.float_format', lambda x: '%.2f' % x)
train.describe()
#train.plot.scatter(x='passenger_count', y='trip_duration')

#
# passenger_count 
print(train["passenger_count"].max() )

