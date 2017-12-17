INPUT_FILE = "bitcoin_dataset.csv"

from fbprophet import Prophet
import numpy as np
import pandas as pd

columns_to_read = ["Date","btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
cols = ["btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]


def predict_next_val(number_period):
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'], usecols=columns_to_read)

    predicated_vals = []
    for x in cols :
        df1 = df[["Date",x]]
        df1.columns = ["ds","y"]
        df1["y"] = np.log(df1["y"])

        m = Prophet()
        m.fit(df1)
        future = m.make_future_dataframe(periods=number_period)
        forecast = m.predict(future)
        # print forecast.tail(number_period+2)
        last_val = forecast.tail(number_period)
        predicated_vals.append(last_val[['yhat']].applymap(np.exp).values.tolist())

    return np.array(predicated_vals).T


# predicted_vals = predict_next_val(30)
# print predicted_vals