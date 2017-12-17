# Predicting-Price-of-Cryptocurrency
Machine Leaning regression on cryptocurrency price prediction using SVM, HMM, PCA, fbProphet, continuous HMM, ARIMA and comparing the results 

<h1>What is it about</h1>
Ultimate goal for this project is to predict the price of bitcoin in near future. We analyzed the multiple scenarios like predicting the closing price for day, given opening price and predicting the all open, close, high, and low prices based on historical data.</br>

<h1> Data Representation</h1>
We used bitcoin and ethereum price data from kaggle. The data includes daily open, close, high, and low prices of bitcoin from April 28, 2013 till date. Variation in features of time series data is represented below:

<h2>Bitcoin Dataset</h2>
1) Closing price variation:

![alt text](Data_Analysis/architecture.PNG)


2) Opening price variation:

![alt text](Data_Analysis/architecture.PNG)

<h2>Ethereum Dataset</h2>
1) Closing Price Variation:

![alt text](Data_Analysis/architecture.PNG)

2) Opening Price Variation:

![alt text](Data_Analysis/architecture.PNG)


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

![alt text](Data_Analysis/architecture.PNG)

![alt text](Data_Analysis/architecture.PNG)

<h3>1) ETHEREUM</h3>

![alt text](Data_Analysis/architecture.PNG)

![alt text](Data_Analysis/architecture.PNG)

<h2>2) HMM</h2>

When the opening and closing price is plotted on graph, we could see that there is close relation between them.

To leverage this relationship, we tried using HMM. We created observation sequence using historical data as follows:</br>
Calculate change in opening and closing price as  close-openopen</br>
Divide all obtained values in ranges and label them starting 0.</br>

We trained multiple HMM models using various number of states and various number of observation labels.</br>
For prediction, we created list of possible predictions using all the label created while obtaining observation sequence.</br>

We used solution to classic HMM problem that given a model and observation sequence, score the sequence in terms of likelihood. Hence, the problem becomes given HMM model and bitcoin values for d days along with open value for d+1st day, we need to compute close value for d+1st day.</br>
The predicted value Od+1 is iterated over all possible values and most likely prediction is selected as 
\begin{equation}
O_d+1=argmax_Od+1 P(O1, O2, …,Od,Od+1|λ)
\end{equation}

<h3>RESULTS</h3>

![alt text](Data_Analysis/architecture.PNG)

![alt text](Data_Analysis/architecture.PNG)

<b>TRYING TO MAKE SENSE OUT OF HMM:<b>
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


![alt text](Data_Analysis/architecture.PNG)


