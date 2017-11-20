import numpy as np
import pandas as pd
import sklearn.decomposition as model
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import fbprophet2 as prophet
reload(prophet)

INPUT_FILE1 = "bitcoin_dataset.csv"
INPUT_FILE2 = "bitcoin_price.csv"
columns_to_read = ["Date","btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
cols2 = ["btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
Y_label = "High"
#split ratio for the test train split
split_ratio = 0.7
date = "Date"
bar_width = 0.2

#number of componends for PCA
number_of_components = 3
#SVM parameters
kernel = ['poly','rbf','sigmoid']
degree = 3
#user input value
my_high_val = [5000,6000]

#number of predictions for new days
periods = 10

#number of future vals added to dataset to overcome bias
no_future_vals = 200

colors = sns.color_palette()


#read csv file
def read_file(columns_to_read):
    df = pd.read_csv(INPUT_FILE1, parse_dates=['Date'], usecols=columns_to_read)
    return df

#Apply PCA to find the covariant data and the pricipal components
def get_transformed_features(df,number_of_components):
    PCA = model.PCA(n_components=number_of_components)
    PCA.fit(df)
    new_features = PCA.transform(df)
    assert new_features.shape[0] == len(df)
    return (PCA,new_features)

#get SVM classifier
def train_SVM(X,y,kernel,degree=None):
    clf = None
    if kernel == "poly":
        clf = svm.SVC(kernel=kernel,degree=degree)
    else :
        clf = svm.SVC(kernel=kernel)
    clf.fit(X,y)
    return clf


def get_final_dataframe():
    #read the file
    df = read_file(columns_to_read)
    df.set_index(pd.to_datetime(df[date]), inplace=True)

    #read y values
    df2 = pd.read_csv(INPUT_FILE2, parse_dates=[date], usecols=[date,Y_label])
    df2 = df2[::-1]
    #merge the two files based on the date collomn, final dataframe
    fdf = pd.merge(df,df2,how="inner", on=date, left_index=True)
    #shuffle dataframe
    fdf = fdf.sample(frac=1)
    # print fdf.head()
    return fdf



fdf = get_final_dataframe()
# feature columns dataframe
df = fdf[cols2]

#this was done to check the classifier. It turns out if the high_val set is too high the classifier is biased since there is less data
# msk = fdf["Date"] == "2017-11-07"
# test_data_new = fdf[msk]
# # print test_data_new
# test_data_new = np.log(test_data_new[cols2])


# transform the data. Apply log to smoothen the data
df1 = df[cols2].applymap(np.log)
# get PCA classifier new tranformed features fromd PCA
PCA_clf, new_features = get_transformed_features(df1, number_of_components)
# train SVM and test it
split_per = len(df1) * split_ratio
training, test = new_features[:split_per, :], new_features[split_per:, :]
# print training.shape

predicted_vals = prophet.predict_next_val(no_future_vals+10)[0]
#smoothen the new values received from fbprophet
smoothed_new_predicted_vals = np.log(predicted_vals)
# transform the new values using the trained PCA
tranformed_new_data = PCA_clf.transform(smoothed_new_predicted_vals)
#use 10 samples for real time prediction
future_new_features = tranformed_new_data[no_future_vals:,:]
# print tranformed_new_data.shape
training = np.concatenate((training,tranformed_new_data[:no_future_vals,:]))



for kern in kernel:
    for x in range(len(my_high_val)):
        my_high = my_high_val[x]

        # only Y labels column dataframe
        Y = fdf[[Y_label]]

        # create labels
        Y = np.array(Y[Y_label].apply(lambda x: 0 if x < my_high else 1))

        # train SVM and test it
        Y_train, Y_test = Y[:split_per], Y[split_per:]
        #concatenate number of future values
        Y_train = np.concatenate((Y_train,np.ones(no_future_vals)))

        clf = train_SVM(training, Y_train, kern, degree)
        Y_exp = clf.predict(test)
        Y_score = clf.decision_function(test)
        print accuracy_score(Y_exp, Y_test)
        # print Y_exp
        # print Y_test

        Y_real_time = clf.predict(future_new_features)
        print Y_real_time

        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print "Area under curve = "+ str(roc_auc)
