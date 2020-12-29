#!flask/bin/python
# Web server
from flask import Flask
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams

from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, save_model
# Get request parameters
from flask import request
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# API server
app = Flask(__name__)

graph = tf.get_default_graph()

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

df = pd.read_csv("../Data/prices-split-adjusted.csv")
df.info()


plot_x = df['date'].copy()
df.set_index("date", inplace = True)
df.index = pd.to_datetime(df.index)
df.head()


google_stock = df[df['symbol'] == 'GOOG']
google_stock.head()

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
goog_df = google_stock.copy()
goog_df.drop(['symbol'], axis=1, inplace=True)
x = goog_df[['open', 'low', 'high', 'volume']].copy()

y = goog_df['close'].copy()

x[['open', 'low', 'high', 'volume']] = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y.values.reshape(-1, 1))


model = load_model("c:\TNB\Learning\Python\SQL Server\model_01.h5")

@app.route('/getstock/<title>', methods=['GET'])
def getSearch(title):
    return title

@app.route('/getstock01/<a>/<b>', methods=['GET'])
def getSearch01(a, b):
    return a+b


def StockPricePredictApi(open, low, high, volumn):
    print('StockPricePredict aaaa')
    print('open')
    print(open)

    print('low')
    print(low)

    print('high')
    print(high)

    print('volumn')
    print(volumn)
    # print(x_scaler.transform(volumn))

    a = pd.DataFrame(list(zip(open, low, high, volumn, open)), columns=['open', 'low', 'high', 'volume', 'close'])

    Y = a[['close']]
    X = a[['open', 'low', 'high', 'volume']]

    X[['open', 'low', 'high', 'volume']] = x_scaler.transform(X)
    Y = y_scaler.transform(Y)

    X['close'] = Y

    X = X.values

    # print('X.values')

    # print(X)

    # print(type(X))

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    print(X)

    print(type(X))

    with graph.as_default():
        res = model.predict(X)

    return res


#@app.route('/getstock02/<open>/<low>/<high>/<volumn>/<close>', methods=['GET'])
def StockPricePredictTest(open, low, high, volumn,close):
    vClose = StockPricePredictApi([312.304948, 312.304948], [310.955001, 310.955001], [313.580158, 313.580158],
                                  [3927000.0, 3927000.0])
    print('vClose Api')
    print(vClose)
    return vClose


@app.route('/getstock02/<open>/<low>/<high>/<volumn>/<close>', methods=['GET'])
def StockPricePredict(open, low, high, volumn,close):
    #print('StockPricePredict aaaa')
    #print('open')
    #print(open)

    #print('low')
    #print(low)

    #print('high')
    #print(high)

    #print('volumn')
    #print(volumn)

    #sum = float(open)+float(low)+float(high)+float(volumn)+float(close)

    a = pd.DataFrame(list(zip([float(open)], [float(low)], [float(high)], [float(volumn)], [float(close)])), columns=['open', 'low', 'high', 'volume', 'close'])

    #print(a)

    X = a.values

    #print('X.values')

    #print(X)

    #print(type(X))

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    #print(X)

    #print(type(X))

    #model = load_model("c:\TNB\Learning\Python\SQL Server\model_01.h5")
    #model.summary()

    with graph.as_default():
        res = model.predict(X)

    #print(res)

    #print(res.item(0, 0))


    #print(type(res.item(0, 0)))

    #predict = 0

    #for z in np.nditer(res):
        #predict = z
        #print(z, end=' ')
    #return

    #print(predict)
    #print(type(predict))

    return str(res.item(0, 0))



# Main app
if __name__ == '__main__':
    app.run(port=1234, host='localhost', debug=True)
