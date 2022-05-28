from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import math
import base64
# import io
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import numpy as np
from array import array 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import yfinance as yf


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def form():
    stockName = request.form['Name']
    epoch = request.form['Epochs']
    batchSize = request.form['BatchSize']
    dateStart = request.form['DateStart']
    dateEnd = request.form['DateEnd']

    bs = int (batchSize)
    ep = int(epoch)
    stock = yf.Ticker(stockName)
    ds = str(dateStart)
    de = str(dateEnd)

    hist = stock.history(start=ds, end=de)
    df=hist

    # Create a new dataframe with only the Close column
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .7)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    # Create scaled training data set
    train_data = scaled_data[0:training_data_len , :]
    # Split the data into x_train and y_train dataset
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
      x_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i,0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM
    model = Sequential()
    model.add(LSTM(120, return_sequences=True,  activation="tanh", input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(30, return_sequences=False, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(30))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=bs, epochs=ep, verbose=1, validation_split=0.2)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data set x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
      x_test.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    mape = np.mean((np.abs((predictions - y_test)/predictions))*100)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)

    df=df.reset_index(drop=True)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.xticks(rotation=45)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    new_df = hist.filter(['Close'])
    # Get the last 60 day closing price values dan convert the dataframe to an array
    last_60_days = new_df[-60:].values
    # Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # Create an empty list
    x_test = []
    # Append the past 60 days
    x_test.append(last_60_days_scaled)
    # Convert the x_test data set to a numpy array
    x_test = np.array(x_test)
    # Reshape the data 
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    # Get the predicted scaled price
    pred_price = model.predict(x_test)
    # Undo the scaling
    pred_price = scaler.inverse_transform(pred_price)

    hasilMape=str(mape)
    hasilR2=str(r2)
    # hasilPred=''.join(map(str, pred_price))
    hasilPred = ' '.join(''.join(map(str,i)) for i in pred_price)
    STOCK = BytesIO()
    plt.savefig(STOCK, format="png")

    MAPE = StringIO()
    MAPE.write(hasilMape)
    R2 = StringIO()
    R2.write(hasilR2)
    PRED_PRICE = StringIO()
    PRED_PRICE.write(hasilPred)

    """Send the plot to plot.html"""

    STOCK.seek(0)
    MAPE.seek(0)
    R2.seek(0)
    PRED_PRICE.seek(0)
    plot_url = base64.b64encode(STOCK.getvalue()).decode('utf8')
    return render_template("plot.html", plot_url=plot_url, map=MAPE.read(), R2=R2.read(), PRED_PRICE=PRED_PRICE.read())



if __name__ == '__main__':
    app.run(debug=True)
