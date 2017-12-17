# Predicting-Price-of-Cryptocurrency
Machine Leaning regression on cryptocurrency price prediction using SVM, HMM, PCA, fbProphet, continuous HMM, ARIMA and comparing the results 

<h1>What is it about</h1>
Ultimate goal for this project is to predict the price of bitcoin in near future. We analyzed the multiple scenarios like predicting the closing price for day, given opening price and predicting the all open, close, high, and low prices based on historical data.</br>

<h1> Data Representation</h1>
We used bitcoin and ethereum price data from kaggle. The data includes daily open, close, high, and low prices of bitcoin from April 28, 2013 till date. Variation in features of time series data is represented below:

<h2>Bitcoin Dataset</h2>
1) Closing price variation:

![alt text](Data_Analysis/1.PNG)


2) Opening price variation:

![alt text](Data_Analysis/2.PNG)

<h2>Ethereum Dataset</h2>
1) Closing Price Variation:

![alt text](Data_Analysis/3.PNG)

2) Opening Price Variation:

![alt text](Data_Analysis/4.PNG)


Data representation of High, Low features also looked similar to Open and Close values for that cryptocurrency as shown above.

NOTE:
In below approaches, the report focuses on OPEN and CLOSE value prediction only, to make the report succinct. However, experiments were also performed on HIGH and LOW feature values and the results were almost similar to OPEN and CLOSE. So, we are only focusing on OPEN and CLOSE features in the report. 

<h1>Approaches</h1>
<h2>1) FB Prophet</h2>

Prophet developed by Facebook, is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers.

<h3>RESULTS:</h3>
Following are the graphs for Bitcoin and Ethereum Open and Close feature values prediction using fbProphet. 
The graph represents the actual Open and Close feature trend over the time, with the lower bound, upper bound and mean predicted values for that time series data.
The red line in the graph represents the mean predicted values bounded by the lower and upper bound.

<h3>1) BITCOIN</h3>

![alt text](Data_Analysis/5.PNG)

![alt text](Data_Analysis/6.PNG)

<h3>1) ETHEREUM</h3>

![alt text](Data_Analysis/7.PNG)

![alt text](Data_Analysis/8.PNG)

<h2>2) HMM</h2>

When the opening and closing price is plotted on graph, we could see that there is close relation between them.

To leverage this relationship, we tried using HMM. We created observation sequence using historical data as follows:</br>
Calculate change in opening and closing price as  close-openopen</br>
Divide all obtained values in ranges and label them starting 0.</br>

We trained multiple HMM models using various number of states and various number of observation labels.</br>
For prediction, we created list of possible predictions using all the label created while obtaining observation sequence.</br>

We used solution to classic HMM problem that given a model and observation sequence, score the sequence in terms of likelihood. Hence, the problem becomes given HMM model and bitcoin values for d days along with open value for d+1st day, we need to compute close value for d+1st day.</br>
The predicted value Od+1 is iterated over all possible values and most likely prediction is selected as 

![alt text](Data_Analysis/9.PNG)

<h3>RESULTS</h3>

![alt text](Data_Analysis/10.PNG)

![alt text](Data_Analysis/11.PNG)

<b>TRYING TO MAKE SENSE OUT OF HMM:</b>
We can see even though, these models converge pretty good and we can take inferences from the transition matrix and observation matrix, the prediction part is not working well. Every observation gets the score very close to 1, so it does not help us predict correctly. The variance is so minute that while printing every score is printed as 1, but as we can see in last result (N=6), some scores are better than others by a very tiny margin.</br>

Another way of predicting values for (d+1)st day, would have been by looking at the B matrix and trying to predict the next best value. Such an approach would be like flipping a coin or randomly selecting a state for dth day. Based on the selected state we can look at the transition matrix to find the most probable state for (d+1)st day. On finding the most probable state for the (d+1)st day we can look at the B matrix and find the most probable observation range. Based on this range we can calculate closing value, when given opening value for (d+1)st day. 
Here is the catch ☹. If we look closely at the B matrix, irrespective of the value of N. The highest probabilities are concentrated at observation c (range -10% to 0%) and d (range 0% to 10%). So, irrespective of the whichever state we select for the (d+1)st day we will always land either on c or d observation, which is like flipping a coin.</br>

<h2>3) GMM-HMM (Gaussian Mixture Model)</h2>

After getting very low accuracy in simple HMM, we moved on to Gaussian Mixture based HMM.
Earlier in simple HMM approach we were dividing the observation (close-openopen) into different discrete ranges. The intuition behind trying GMM-HMM approach was moving from discrete observations to continuous observations. A C library with python wrappings ‘hmmlearn’ has GMMHMM capabilities. We trained the GMMHMM models as follows:
Observation sequence was created using vector observations. Observation at time t was given as
Ot= (close-open/open, high-open/open, open-low/open)

Hence Ot= (fractional-Close, fractional-High, fractional-Low)

We used Maximum a Posteriori algorithm to train the model. We used different number of states and different number of mixtures to train the models.
For prediction, we used following point as possible observations.

![alt text](Data_Analysis/12.PNG)

We used the function Od+1=argmaxOd+1 P(O1, O2, …,Od,Od+1|λ)  to compute most likely prediction. In this case also, we got equal scores for each of the observations. Sample result is as follows:

![alt text](Data_Analysis/13.PNG)

The values between the brackets [] are the observation being scored (fractional-Close, fractional-High, fractional-Low) and the next value is the predicted score for that observation 

<h2>4) SVM-PCA</h2>
In this technique, we tried to apply ML to answer specific business query i.e. will I earn profit selling the bitcoin tomorrow. In this question, a user has to specify the price at which the user wants to sell the bitcoin.</br>

To train the model, we used different features than earlier models.</br>
For bitcoin:</br>
btc_avg_block_size, btc_n_transactions, btc_n_transactions_total, btc_n_transactions_excluding_popular, btc_n_transactions_excluding_chains_longer_than_100, btc_output_volume</br>
For Ethereum:</br>
eth_supply, eth_hashrate, eth_difficulty, eth_blocks, eth_blocksize, eth_blocktime, eth_ethersupply</br>
Since we did not know the significance of each feature, we used PCA to narrow the dimensionality of the data. We found that, there are 3 most prominent eigen values. So, our experiments were focused on 3 and 2 most prominent bases.</br>

Then we label this data using known close and open price if more than user give price as +1 or -1. The next step was to train SVM based on this labeled data. After training the model on given historical data, we predicted the if the price would be higher than user specified one. The result was obtained as follows:</br>

<b>Bitcoin results:</b>

![alt text](Data_Analysis/14.PNG)

Here AUC represents the total area under ROC curve.

![alt text](Data_Analysis/15.PNG)

Looking at these values, which look pretty good, we had a closer look at the results. We found that the data is very skewed. Most of the data had the price range below the user specified price. In recent months only, the price of the bitcoin is booming. So, we used F1 score as it can better represent the accuracy of the model when data is skewed.<br>

![alt text](Data_Analysis/16.PNG)

We were not surprised by looking this bar chart. Sigmoid kernel gave the worst results, however polynomial and RBF kernel looked promising. 
fbProphet to the rescue:</br>
So, in order to solve the problem of data skewness, we randomly generated data using the predictions made by fbProphet for next 200 days and added this data to original pool before train the SVM.
Below are the results:</br>

![alt text](Data_Analysis/17.PNG)

![alt text](Data_Analysis/18.PNG)

![alt text](Data_Analysis/19.PNG)

We can clearly see that the result is improved when we tried to reduce the skewness of the data. </br>
We also plotted PR curves. Some sample representatives are as follows:</br>

![alt text](Data_Analysis/20.PNG)

![alt text](Data_Analysis/21.PNG)

![alt text](Data_Analysis/22.PNG)

<b>Note:</b> 
All the above graphs are for 3 PCA components and trained on bitcoin dataset. We have also done same experiments in different number PCA components and Ethereum dataset. Since, the results are similar to above once and to keep the report concise, we only presented subset of all experimental results.

<h1>Conclusion</h1>
To predict the future price of cryptocurrencies is tougher than it looks. We tried Facebook Prophet, HMM, GMM-HMM, PCA-SVM, and PCA-SVM-FBProphet. In some cases, the results look promising. But, there are a lot we can improve.

