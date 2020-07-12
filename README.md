# House-Price-Prediction---regression

Kaggle competition : in top 25% of entries on leaderboard
House Prices: Advanced Regression Techniques
Predict sales prices and practice feature engineering, RFs, and gradient boosting


Using tools and techniques introduced and practiced through this course (Predictive Modelling) and previous courses, we were able to build a robust linear regression model to help predict house prices based on the variables in the test dataset.  In order to achieve our final result, which helped us achieve a leaderboard score in the 25% percentile on Kaggle, we started off with a simple linear regression model and built on to it in order to minimize the error rate on our test set.

Preliminary Analysis:

We first started off with some preliminary analysis on the dataset and did some background research on the real estate market and factors that are generally important. Generally, we know that things like location, # of rooms, quality of house, property age, and square footage are generally positively correlated with the sale price. We were able to confirm this using the dataset with basic graphs in tableau and R. 

Data Management (Feature Engineering, Outlier Analysis, Missing Values):

First of all, we start with importing the data into the tool of our choice, R. We are given two datasets: test and train. We merge these two and proceed with the data management piece where we will eventually work to complete this dataset for the purpose of our regression model.

Going through the dataset, we found several missing values (NA). To address these, we used data imputation methods such as mean imputation, mode imputation (for categorical variables) and also assigning new variables (based on data description: see Appendix). We also did some outlier analysis, density plots (look at distribution of columns) and other graphical analysis to view the data based on a variety of outlooks. 

Once the missing data and outliers were addressed, we proceeded with the feature engineering piece to create new features that would help to better predict the outcome. We started with a few features such as PropertyAge and Years since Remodelled and then came back and constructed more to help decrease our error rate on the test data. 

Model Building:

Now we get to the most important piece, building our linear regression model. We start off with creating dummy variables for the various categorical variables in our model. We also ensure the data types are correct based on context (this would have been done in the previous phase). Once we are comfortable, we split up the data into separate datasets to train and test our mod; and finally predict the Sales Price.  We revised the model through feature engineering, outlier analysis and leveraging techniques such as regularization to get to a better model. Here are the steps that were routinely done to get to our current model: 

i.	Use the linear regression tool in R & input independent & dependent variables into the formula to build the regression model. 
ii.	Predict the values and compare with the actual Sales Price using MAPE. 
iii.	Once we get an approximate score, we go back to identify improvements that can be made by either creating new meaningful features (using Feature Engineering) or fixing any issues with the underlying dataset. 
iv.	Note that we ended up using Lasso Regression to regularize the model and ensure ideal complexity to avoid overfitting.
Once we are satisfied with the model results on the test dataset, we use the regression model to predict the actual values for the test dataset (with no Sales Price). This was then exported and uploaded to the Kaggle competition page to retrieve our final results on the Leaderboard.
Analysis the quality of our results, we found the MAPE error to be approximately 9.36% on the test dataset, which was very good in my opinion. On the actual predict dataset, my score was ~12.6% (Root Mean Squared Logarithmic Error) which was impressively in the top 25% percentile in the public leaderboard on Kaggle!
