import numpy as np
import quandl as qd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# silence the tensorflow compilation warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# columns to keep and rename from the original data
use_columns = {
    'Adj. Open': 'ao',
    'Adj. High': 'ah',
    'Adj. Low': 'al',
    'Adj. Close': 'ac',
    'Adj. Volume': 'av'
}

# column to predict, after renaming
predict_column = 'ac'

# the length of x data to be passed into the LSTM
# each frame will overlap the previous by frame_length - 1 elements
frame_length = 21

# fibonacci numbers used for technical indicators
fibs = [13, 21, 34, 55, 89, 144]

def user_tickers():
    '''
    Request any number of tickers, returning them as a list.
    '''
    print()
    tickers = []
    confirm = 'y'
    while confirm == 'y':
        ticker = input('Enter a stock ticker: ')
        while not ticker:
            ticker = input('The entry must not be blank. Please retry: ')
        tickers.append(ticker.lower())
        confirm = input('Would you like to enter another ticker (y/[n])? ')
    return tickers

def user_start_date():
    '''
    Request and return the training start date.
    '''
    now = datetime.now()
    start_date = now.replace(year=now.year - 8)
    print()
    print('The default training start date is %s.' % start_date.strftime('%Y-%m-%d'))
    sd_confirm = (input('Would you like to enter a new training start date (y/[n])? ') == 'y')
    while sd_confirm:
        try:
            date = datetime.strptime(input('Enter a new start date as yyyy-mm-dd: '), '%Y-%m-%d')
        except ValueError:
            print('Try again. Make sure to enter as yyyy-mm-dd.')
            continue
        else:
            break
    return start_date

def user_end_date():
    '''
    Request and return the training end date.
    '''
    end_date = None
    print()
    print('The default training end date is %s.' % end_date)
    ed_confirm = (input('Would you like to enter a new training end date (y/[n])? ') == 'y')
    while ed_confirm:
        try:
            date = datetime.strptime(input('Enter a new end date as yyyy-mm-dd: '), '%Y-%m-%d')
        except:
            print('Try again. Make sure to enter as yyyy-mm-dd.')
        else:
            break
    return end_date

def user_advance():
    '''
    Request and return the advance length.
    '''
    print()
    while True:
        try:
            advance = int(input('How many days forward would you like to predict? '))
            if advance < 1:
                print('Your entry must be a positive integer.')
                continue
        except:
            print('Check your input and retry.')
            continue
        else:
            break
    return advance

def user_charts():
    '''
    Request and return if the user would like to see charts.
    '''
    print()
    return input('Would you like to see charts of the results? (y/[n])') == 'y'

def print_accuracy(diff, name='Diff.'):
    '''
    Print accuracies in a standard format.
    '''
    print("%s Accuracy" % name)
    print("==============")
    print("Mean: %.4f%%" % (100 * np.mean(diff)))
    print("Abs. Mean: %.4f%%" % (100 * np.mean(np.absolute(diff))))
    print("Std. Dev.: %.4f%%" % (100 * np.std(diff)))
    print()

def get_data(ticker, start_date, end_date, use_columns, predict_column, frame_length, advance):
    '''
    Retrieve data from quandl, clean data, and process.
    '''

    # get data from quandl
    #data = qd.get('wiki/' + ticker, start_date=start_date, end_date=end_date, api_key='PUT KEY HERE')
    # quandl anonymous query
    data = qd.get('wiki/' + ticker, start_date=start_date, end_date=end_date)

    # remove unadjusted columns
    data = data[sorted(list(use_columns.keys()))].rename(columns=use_columns)

    # add technical indicators
    for fib in fibs:
        data['ma' + str(fib)] = data[predict_column].ewm(span=fib).mean()
        data['diff' + str(fib)] = data[predict_column].diff(fib)
        data['pct' + str(fib)] = data[predict_column].pct_change(fib)

    # create y column and drop entries with na
    x = data.dropna()
    y = data[predict_column].rename('y')
    src = x.join(y, how='left')

    # split data into x, y, and dates
    x_src = src.drop('y', axis=1).values
    y_src = src['y'].values
    index = src.index.values

    # create scaler for independent data
    scaler = StandardScaler()

    x, y, dates = [], [], []
    for i in range(len(x_src) - frame_length - advance + 1):
        # append frame of data to x
        x.append(scaler.fit_transform(x_src[i:i+frame_length]))

        # append target value and scale value to y
        y.append((y_src[i+frame_length-1+advance] / y_src[i+frame_length-1], y_src[i+frame_length-1]))

        # append date
        dates.append(index[i+frame_length-1])

    return np.array(x), np.array(y), dates

def fit_model(x_train, x_valid, y_train, y_valid):
    '''
    Creates and fits model to supplied data.
    '''
    model = Sequential()
    model.add(Conv1D(256, 3, activation='relu', padding='same', input_shape=(x.shape[1], x.shape[2])))
    model.add(MaxPooling1D(padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(padding='same'))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(4))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')

    # add callbacks
    # save best fit
    save_str = 'weights.best.price_predictor_py.%s.hdf5' % ticker
    checkpointer = ModelCheckpoint(filepath=save_str, verbose=0, save_best_only=True)

    # stop early if no improvement after <patience> epochs
    early_stopper = EarlyStopping(patience=16, min_delta=0.0000, verbose=0)

    # fit model with callbacks
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=256,
        batch_size=2,
        callbacks=[checkpointer, early_stopper],
        verbose=1
    )

    # reload best model fit
    model.load_weights(save_str)

    return model, history

if __name__ == '__main__':
    # get user data
    tickers = user_tickers()
    start_date = user_start_date()
    end_date = user_end_date()
    advance = user_advance()
    charts = user_charts()
    if charts:
        print('Charts will be shown.')
        print()

    # do the analysis for each ticker
    for ticker in tickers:
        if end_date is None:
            end_date = datetime.now()
        intro_string = 'Analyzing %s' % ticker.upper()
        print(intro_string)
        print('=' * len(intro_string))
        print('Training dates: %s - %s' % (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        print('Predicting %s days in advance...' % advance)
        print()

        x, y, dates = get_data(ticker, start_date, end_date, use_columns, predict_column, frame_length, advance)

        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.6)
        x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5)

        model, history = fit_model(x_train, x_valid, y_train[:, 0], y_valid[:, 0])

        y_hat = np.append(np.zeros(advance)*np.nan, model.predict(x).reshape(-1) * y[:, 1])
        y_valid_hat = model.predict(x_valid).reshape(-1) * y_valid[:, 1]
        y_test_hat = model.predict(x_test).reshape(-1) * y_test[:, 1]

        y_diff = y_hat[advance:-advance] / y[advance:, 1] - 1
        y_valid_diff = (y_valid_hat / (y_valid[:, 0] * y_valid[:, 1])) - 1
        y_test_diff = (y_test_hat / (y_test[:, 0] * y_test[:, 1])) - 1

        # provide printouts
        print()
        print_accuracy(y_diff, name='Model')
        print_accuracy(y_valid_diff, name='Validation')
        print_accuracy(y_test_diff, name='Test')
        print()
        result_string = '%s RESULTS' % ticker.upper()
        print(result_string)
        print('=' * len(result_string))
        print('Training dates: %s - %s' % (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        print('Current price: %s' % y[-1, 1])
        print('Expected price %s trading days from now: %.2f' % (advance, y_hat[-1]))
        print('Expected percent change %s trading days from now: %.2f%%' % (advance, y_hat[-1] / y[-1, 1]))
        print()

        # produce charts if requested
        if charts:
            plt.figure(figsize=(16, 8))
            plt.title('Ticker: %s' % ticker.upper())
            plt.plot(dates, y[:, 1], label='Actual')
            plt.plot(dates, y_hat[:-advance], label='Predicted')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.gca().set_yscale('log')
            plt.grid(True)
            plt.legend()
            plt.show()

            plt.figure(figsize=(16, 8))
            plt.title('Ticker: %s' % ticker.upper())
            plt.plot(range(-251, 1), y[-252:, 1], label='Actual')
            plt.plot(range(-251, advance+1), y_hat[-252-advance:], label='Predicted')
            plt.xlabel('Days, Relative to End Date')
            plt.ylabel('Price')
            plt.gca().set_yscale('log')
            plt.grid(True)
            plt.legend()
            plt.show()
