INPUT_FILE = "bitcoin_price.csv"

from fbprophet import Prophet
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

color = sns.color_palette()


def plot_data(df,X,future,columns,title):
    future = future['ds'].apply(lambda x: mdates.date2num(x))
    fig, ax = plt.subplots(figsize=(12, 8))
    #plot the mean predicted value
    sns.tsplot(df[columns[0]], time=future, alpha=0.8, color="red", ax=ax)
    #plot the actual value
    plt.scatter(future, X,color="orange", alpha=0.3)
    #set the major axis as the date
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
    #configure the plot
    fig.autofmt_xdate()
    #plot the upper and lower bounded regions and fill it
    plt.plot_date(future,df[columns[1]],'-',color="#66b3ff")
    plt.plot_date(future, df[columns[2]],'-',color="#004080")
    plt.fill_between(future, df[columns[1]], df[columns[2]],facecolor='blue', alpha=0.1, interpolate=True)
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price in USD', fontsize=12)
    plt.title(title, fontsize=15)
    plt.show()



def predict_and_plot_closing_price():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], usecols=["Date", "Close"])
    df.columns = ["ds", "y"]
    df["y"] = np.log(df["y"])

    print df.head()

    m = Prophet()
    m.fit(df);
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] = np.e(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)
    # m.plot(forecast).show()

    predicted = forecast[['yhat', 'yhat_lower', 'yhat_upper']].applymap(np.exp)
    cols = ["Mean", "Lower_closing", "Upper_closing"]
    predicted.columns = cols
    actual = np.exp(df["y"])
    actual = actual[::-1]
    actual.name = "Actual"
    title = "Closing price distribution of bitcoin"
    plot_data(predicted, actual, future, cols,title)

def predict_and_plot_opening_price():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], usecols=["Date", "Open"])
    df.columns = ["ds", "y"]
    df["y"] = np.log(df["y"])

    print df.head()

    m = Prophet()
    m.fit(df);
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] = np.e(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)
    # m.plot(forecast).show()

    predicted = forecast[['yhat', 'yhat_lower', 'yhat_upper']].applymap(np.exp)
    cols = ["Mean", "Lower_Opening", "Upper_Opening"]
    predicted.columns = cols
    actual = np.exp(df["y"])
    actual = actual[::-1]
    actual.name = "Actual"
    title = "Opening price distribution of bitcoin"
    plot_data(predicted, actual, future, cols, title)

def predict_and_plot_high_price():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], usecols=["Date", "High"])
    df.columns = ["ds", "y"]
    df["y"] = np.log(df["y"])

    print df.head()

    m = Prophet()
    m.fit(df);
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] = np.e(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)
    # m.plot(forecast).show()

    predicted = forecast[['yhat', 'yhat_lower', 'yhat_upper']].applymap(np.exp)
    cols = ["Mean", "Lower_High", "Upper_High"]
    predicted.columns = cols
    actual = np.exp(df["y"])
    actual = actual[::-1]
    actual.name = "Actual"
    title = "High price distribution of bitcoin"
    plot_data(predicted, actual, future, cols, title)


def predict_and_plot_Low_price():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], usecols=["Date", "Low"])
    df.columns = ["ds", "y"]
    df["y"] = np.log(df["y"])

    print df.head()

    m = Prophet()
    m.fit(df);
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] = np.e(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(40)
    # m.plot(forecast).show()

    predicted = forecast[['yhat', 'yhat_lower', 'yhat_upper']].applymap(np.exp)
    cols = ["Mean", "Lower_Low", "Upper_Low"]
    predicted.columns = cols
    actual = np.exp(df["y"])
    actual = actual[::-1]
    actual.name = "Actual"
    title = "Low price distribution of bitcoin"
    plot_data(predicted, actual, future, cols, title)


predict_and_plot_Low_price()
predict_and_plot_closing_price()
predict_and_plot_high_price()
predict_and_plot_opening_price()