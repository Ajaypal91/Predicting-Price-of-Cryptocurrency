import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# file_name = "bitcoin_price.csv"
file_name = "ethereum_price.csv"

color = sns.color_palette()
df = pd.read_csv(file_name,parse_dates=['Date'])
df = df.iloc[::-1]
df['Date_mpl'] = df['Date'].apply(lambda x: mdates.date2num(x))

#closing price
fig, ax = plt.subplots(figsize=(12,8))
sns.tsplot(df.Close.values, time=df.Date_mpl.values, alpha=0.8, color=color[1], ax=ax)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
fig.autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price in USD', fontsize=12)
plt.title("Closing price distribution of bitcoin", fontsize=15)
plt.show()

#opening price
fig, ax = plt.subplots(figsize=(12,8))
sns.tsplot(df.Open.values, time=df.Date_mpl.values, alpha=0.8, color=color[5], ax=ax)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
fig.autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Opening Price in USD', fontsize=12)
plt.title("Opening price distribution of bitcoin", fontsize=15)
plt.show()

#High price
fig, ax = plt.subplots(figsize=(12,8))
sns.tsplot(df.High.values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
fig.autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('High Price in USD', fontsize=12)
plt.title("Highest price distribution of bitcoin", fontsize=15)
plt.show()

# price
fig, ax = plt.subplots(figsize=(12,8))
sns.tsplot(df.High.values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
fig.autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('High Price in USD', fontsize=12)
plt.title("Highest price distribution of bitcoin", fontsize=15)
plt.show()