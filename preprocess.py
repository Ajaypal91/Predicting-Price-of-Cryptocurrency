import numpy as np
import pandas as pd

#create DataFrame from CSV data
df = pd.read_csv('data/bitcoin_price.csv', parse_dates=['Date'])

#reverse the data using integer based
df = df.iloc[::-1]

#filter columns
features = df[["Open","Close"]].values
print features.shape

#save observation sequence
#features.to_csv("data/observations_3.csv", index=False)

price_variation = ((features[:,1]/features[:,0]) - 1)*100

observation = pd.DataFrame(price_variation)

observation.to_csv("data/observations_4.csv", index=False)
