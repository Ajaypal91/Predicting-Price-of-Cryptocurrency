import numpy as np
import pandas as pd
import sklearn.decomposition as model
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE1 = "bitcoin_dataset.csv"
INPUT_FILE2 = "bitcoin_price.csv"
columns_to_read = ["Date","btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
cols2 = ["btc_avg_block_size", "btc_n_transactions","btc_n_transactions_total","btc_n_transactions_excluding_popular","btc_n_transactions_excluding_chains_longer_than_100","btc_output_volume"]
Y_label = "Open"
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
my_high_val = [1000,2000,3000,4000,5000,6000]

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

def plot_accuracy_plot(title, X, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.2
    Y = [str(x) for x in my_high_val]
    ind = np.arange(len(Y))

    ## the bars
    rects1 = ax.bar(ind, X[0], width,color='black')

    rects2 = ax.bar(ind + width, X[1], width, color='red')
    rects3 = ax.bar(ind + 2*width, X[2], width, color='blue')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.set_autoscalex_on(my_high_val)
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(Y)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    ax.legend((rects1[0], rects2[0], rects3[0]), ('poly', 'rbf', 'sigmoid'))
    plt.show()

def plot_roc_curves(false_positive_rate, true_positive_rate, title, roc_auc):
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#get accuracy plot based on #components, #kernels used and different high values
def get_accuracy_plot(fdf):
    accuracy = []
    aucs = []
    for kern in kernel:
        temp = []
        temp2 = []
        for x in range(len(my_high_val)):
            my_high = my_high_val[x]

            # only Y labels column dataframe
            Y = fdf[[Y_label]]
            # feature columns dataframe
            df = fdf[cols2]

            # create labels
            Y = np.array(Y[Y_label].apply(lambda x: 0 if x < my_high else 1))

            # transform the data. Apply log to smoothen the data
            df1 = df[cols2].applymap(np.log)

            # get PCA classifier new tranformed features fromd PCA
            PCA_clf, new_features = get_transformed_features(df1, number_of_components)
            # print new_features[:3]

            # train SVM and test it
            split_per = len(Y) * split_ratio
            training, test = new_features[:split_per, :], new_features[split_per:, :]
            Y_train, Y_test = Y[:split_per], Y[split_per:]

            clf = train_SVM(training, Y_train, kern, degree)
            Y_exp = clf.predict(test)
            Y_score = clf.decision_function(test)
            print my_high, kern
            print Y_exp
            print Y_test
            temp.append(accuracy_score(Y_exp, Y_test))

            # #calculate roc curve
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            temp2.append(roc_auc)

            #comment out below code to plot roc curves
            ## print roc_auc
            # title='Receiver Operating Characteristic for ' + Y_label + ' values using ' + kern + ' kernel'
            # plot_roc_curves(false_positive_rate,true_positive_rate,title,roc_auc)


        aucs.append(temp2)
        accuracy.append(temp)

    print "Accuracy = " + str(accuracy)
    print "Area under curve values = " + str(aucs)
    title = "Accuracy plot for different " + Y_label+ " values and Number of Principle components = " + str(number_of_components)
    plot_accuracy_plot(title,accuracy,"Accuracy")
    title = "AUC plot for different " + Y_label + " values and Number of Principle components = " + str(number_of_components)
    plot_accuracy_plot(title,aucs,"AUC")


fdf = get_final_dataframe()
get_accuracy_plot(fdf)









