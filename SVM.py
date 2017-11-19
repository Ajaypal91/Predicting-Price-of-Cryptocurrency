import numpy as np
import pandas as pd
import sklearn.decomposition as model
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

INPUT_FILE1 = "bitcoin_dataset.csv"
INPUT_FILE2 = "bitcoin_price.csv"
columns_to_read = ["Date","btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
cols2 = ["btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
Y_label = "High"
#split ratio for the test train split
split_ratio = 0.7
date = "Date"

#number of componends for PCA
number_of_components = 3
#SVM parameters
kernel = 'poly'
degree = 2
#user input value
my_high_val = 6000


#read csv file
def read_file(columns_to_read):
    df = pd.read_csv(INPUT_FILE1, parse_dates=['Date'], usecols=columns_to_read)
    return df

#Apply PCA to find the covariant data and the pricipal components
def get_transformed_features(df,number_of_components):
    PCA = model.PCA(n_components=number_of_components)
    PCA.fit(df1)
    new_features = PCA.fit_transform(df1)
    assert new_features.shape[0] == len(df1)
    return new_features

#get SVM classifier
def train_SVM(X,y,kernel,degree=None):
    clf = None
    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel,degree=degree)
    else :
        clf = svm.SVC(kernel=kernel)
    clf.fit(X,y)
    return clf


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

Y = fdf[[Y_label]]
df = fdf[cols2]
#craete labels
Y = np.array(Y[Y_label].apply(lambda x: 0 if x < my_high_val else 1))

#transform the data. Apply log to smoothen the data
df1 = df[cols2].applymap(np.log)

#get new tranformed features fromd PCA
new_features = get_transformed_features(df1,number_of_components)
# print new_features[:3]

#train SVM and test it
split_per = len(Y)*split_ratio
training, test = new_features[:split_per,:], new_features[split_per:,:]
Y_train, Y_test = Y[:split_per], Y[split_per:]
msk = Y_train == 1
# print len(Y)
# print len(Y_test)
# print len(Y_train)
# print Y_train[msk]
clf = train_SVM(training,Y_train,kernel,degree)
Y_exp = clf.predict(test)
# print Y_exp
# print Y_test
print accuracy_score(Y_exp,Y_test)









