# encoding: utf-8
# 評価
# 予測問題。 通常データ train/ test.csv の使用して検証する。
# MAE : 平均絶対誤差 (MAE)
# MSE : 平均二乗誤差 (MSE)
# RMSE: 二乗平均平方根誤差 (RMSE)
#
# train : % 
# test  : %

# 途中で使用するため、あらかじめ読み込んでおいてください。
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


# 学習データ
global_start_time = time.time()


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
#quit()

# モデルのインスタンス
model = linear_model.LinearRegression()
# 保存したモデルをロードする
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

#pred
#quit()
#pred = model.predict(x_train )
pred = model.predict(x_test  )
pred_int = np.array( pred , np.int32)
#pred = pred.astype(np.float16)
print( pred_int[: 10] )

print("min=", pred_int.min() )
print("max=", pred_int.max() )
print("mean=", pred_int.mean() )
#quit()

#
#id,trip_duration
# 予測をしてCSVへ書き出す
df = pd.DataFrame(pred_int, train["id"], columns=["trip_duration"])
#
df.to_csv("out.csv", index_label=["id"])
quit()

#df= DataFrame(pred_int)
#print(df.describe() )
#quit()
# 回帰、評価
# MAE
from sklearn.metrics import mean_absolute_error
a= mean_absolute_error(y_train, pred )
print("MAE =" + str(a)  )

#MSE
from sklearn.metrics import mean_squared_error
a=mean_squared_error(y_train, pred )
print("MSE =" + str(a)  )

#RMSE
from sklearn.metrics import mean_squared_error
a= np.sqrt(mean_squared_error(y_train, pred ) )
#print("RMSE=" + str(a)  )
print("RMSE=", a  )
quit()

#plt
#a1=np.arange(len(x_test) )
a1=np.arange(len(x_train ) )
plt.plot(a1 , y_train  , label = "y_train")
#plt.plot(a1 , y_test  , label = "y_test")
plt.plot(a1 , pred , label = "predict")
plt.legend()
plt.grid(True)
plt.title("price pred")
plt.xlabel("x")
plt.ylabel("price")
plt.show()

quit()

