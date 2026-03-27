# Model Explanation

## Big Picture

The final model is a simple machine learning model that predicts the groundwater level of the next month by looking at:

- the recent past months
- the month of the year
- a few summary statistics of recent history

It is not a neural network. It is a linear regression model used in an autoregressive way.

That sounds technical, but the core idea is simple:

> Look at the recent history, look at the season of the year, and estimate the next value.

## 1. What Problem Are We Solving?

We have a sensor measuring underground water level. The raw training data contains daily measurements from 1995 to 2018. The assignment asks us to predict future values with monthly granularity.

So the real task is:

> Use the past to predict future monthly groundwater levels.

This is called **time series forecasting**.

- `time series`: data ordered in time
- `forecasting`: predicting future values

## 2. Why Do We Preprocess the Data?

The raw file is daily, but the assignment wants monthly predictions. Also, the raw data has missing values.

So we do two preprocessing steps:

1. Fill missing daily values with time interpolation
2. Resample daily data into monthly means

Why interpolation?

Because there are missing sensor values. If we ignore them, the monthly averages can become distorted. Time interpolation fills the missing point using the surrounding dates.

Why monthly means?

Because the target of the assignment is monthly forecasting. So we need one value per month, not one value per day.

After this step, the whole training history becomes a monthly series with 288 points.

## 3. Why Do We Split Into Train And Validation?

We do not want to train and test on the same months. That would make the model look better than it really is.

So we split the monthly series like this:

- older months -> training set
- last 48 months -> validation set

Meaning:

- `training set`: used to learn the model
- `validation set`: used to test whether the model works on unseen future months

This is especially important for time series. We must respect chronology:

- past for training
- future for validation

We must never let the model peek into the future.

## 4. What Is A Machine Learning Model, In Simple Words?

A machine learning model is a rule that learns from examples.

Each example has:

- an input
- a correct answer

In this project:

- `input` = recent history + calendar information
- `correct answer` = the next month groundwater level

So the model sees many past situations like:

> Here are the last 18 months and the month we want to predict.  
> The true answer was 106.7.

After seeing many such examples, it learns how to turn similar inputs into predictions.

## 5. What Does The Final Model Look At?

The final model uses three kinds of information:

1. The last 18 monthly values
2. The month of the year
3. Rolling summary statistics

## 6. The Last 18 Monthly Values

This is the autoregressive part.

**Autoregressive** means:

> predicting a value from earlier values of the same series

So if we want to predict a month, we first show the model the 18 months just before it.

Why 18 months?

Because:

- 12 months gives one full yearly cycle
- 18 months gives one full cycle plus some extra recent context

This lets the model see:

- what happened recently
- what happened around the same season last year

## 7. The Month Of The Year

The model also needs to know where it is in the annual cycle. Groundwater level is seasonal. For example, March and August do not behave the same way.

To encode the month, we use a one-hot vector:

- January -> `[1,0,0,0,0,0,0,0,0,0,0,0]`
- February -> `[0,1,0,0,0,0,0,0,0,0,0,0]`
- March -> `[0,0,1,0,0,0,0,0,0,0,0,0]`

and so on.

This is a simple way to say to the model:

- the target month is February
- the target month is August

Without this, the model would only see past values and might miss part of the seasonal structure.

## 8. Rolling Summary Statistics

The model also receives:

- average of the last 3 months
- average of the last 6 months
- average of the last 12 months
- standard deviation of the last 12 months

Why add these if the raw past months are already included?

Because summary features often help the model understand the situation more easily.

They tell it things like:

- what is the current level of the series
- whether recent months are generally high or low
- whether the recent pattern has been stable or variable

Think of it like this:

- the raw 18 values are the detailed history
- the rolling averages are the quick summary

## 9. What Is Linear Regression?

Linear regression is one of the simplest machine learning models.

It works like this:

`prediction = constant + weight1 * feature1 + weight2 * feature2 + ... + weightN * featureN`

This means:

- each input feature gets a learned importance weight
- the model multiplies each feature by its weight
- then it adds everything together

That final sum becomes the prediction.

So in plain language:

> The model learns how much each piece of information should matter.

For example, it may learn:

- last month matters a lot
- the 12-month average matters
- August usually lowers the expected level
- March usually raises it

## 10. Why Is It Called `linear_ar_18`?

Because:

- `linear` = linear regression
- `ar` = autoregressive
- `18` = uses the last 18 months as the main lag window

So the model name is actually a summary of how it works.

## 11. How Does Training Work?

Training means learning the weights.

We create many examples from the training history:

- input: previous 18 months, month of the year, rolling statistics
- target: the actual next month value

The model then searches for the weights that make its predictions as close as possible to the true target values.

How does it decide what is "close"?

It uses prediction error.

- if the model predicts badly, the error is high
- if it predicts well, the error is low

The learning process chooses weights that reduce this error over the training examples.

## 12. What Error Are We Minimizing?

The model is evaluated with squared error and RMSE.

Squared error:

- if you miss by 2, the error is 4
- if you miss by 5, the error is 25

So larger mistakes are punished much more strongly.

RMSE means:

1. compute squared errors
2. average them
3. take the square root

RMSE is useful because:

- it stays in the same unit as the target
- it punishes large mistakes
- it is easy to compare across models

## 13. How Do We Forecast More Than One Month?

The model is trained to predict one next month. But the assignment needs many future months.

So we use **recursive forecasting**.

That means:

1. predict the first future month
2. append that prediction to the history
3. predict the second future month using the updated history
4. append it again
5. repeat until all required months are predicted

Example:

- predict January
- use predicted January to help predict February
- use predicted January and February to help predict March

This is realistic, because in the real future we do not know the true next month yet. We only know our own previous predictions.

## 14. Why Is Recursive Forecasting Important?

Because a model can look good if you only ask for one isolated step. But in practice, forecasting far into the future means each new step depends partly on earlier predictions.

This creates a real challenge:

- if the model makes a mistake early
- that mistake can influence later predictions

So recursive forecasting is a tougher and more honest test.

That is why we ranked models mainly on the 48-month recursive validation RMSE.

## 15. Why Did This Simple Model Beat More Complex Models?

More complex does not automatically mean better.

Your monthly dataset is not huge:

- only 288 monthly observations in total

The series is also:

- fairly smooth
- strongly seasonal
- relatively stable

That kind of dataset often favors simpler models because:

- there is not enough data for big flexible models to shine safely
- simpler models overfit less
- the main signal is seasonal and can be captured directly

So the linear autoregressive model wins not because it is flashy, but because it generalizes better.

## 16. What Is Overfitting?

Overfitting means:

> the model learns the training data too specifically, including noise, and then performs worse on unseen data

A very flexible model can sometimes memorize quirks instead of learning the true pattern.

On small datasets, this risk is high.

Simpler models often generalize better because they are forced to focus on the strongest, most stable patterns.

## 17. Why Do We Compare With Seasonal Naive?

Seasonal naive is the simplest meaningful seasonal baseline.

It says:

- January this year = January last year
- February this year = February last year

and so on.

This is useful because the data clearly has yearly seasonality.

If a complicated model cannot beat seasonal naive, that is a bad sign.

Our final model beats seasonal naive by a large margin on validation, which makes it a credible improvement.

## 18. Why Do We Compare With Other Models Too?

Because good machine learning practice is not:

> pick a fancy model first and hope

It is:

- define a fair evaluation
- test multiple approaches
- compare them honestly
- choose the one that generalizes best

That is exactly what was done:

- naive seasonal baseline
- linear autoregressive models
- tree ensembles
- boosted trees
- AutoReg
- Holt-Winters
- SARIMAX
- Prophet

Then the best one on the shared validation protocol was selected.

## 19. What Are The Strengths Of The Final Model?

- simple and interpretable
- fast to train
- stable on small datasets
- explicitly captures seasonality
- works well in recursive forecasting
- easier to explain than an LSTM

## 20. What Are The Weaknesses Of The Final Model?

- it is linear, so it cannot learn very complex nonlinear effects
- it only uses past target values, not external variables like rainfall or weather
- recursive forecasting can still accumulate error over time
- if the real system changes suddenly, the model may react slowly

These are good points to mention if asked about limitations.

## 21. Why Is This Still Real Machine Learning If It Is Simple?

Because machine learning is not about using the most complicated model.

It is about:

- learning from data
- evaluating on unseen data
- selecting the model that generalizes best

A simpler model that wins fairly is a stronger scientific result than a complex model that only looks impressive.

## 22. What Should I Say If The Teacher Asks How The Model Works?

Short version:

> We first turned the daily sensor series into a monthly series by interpolating missing values and averaging by month. Then for each month to predict, we built features from the last 18 months, the target month in the calendar, and rolling statistics. A linear regression learned how to combine those features to predict the next month. To forecast several months ahead, we rolled the model forward recursively, using each prediction to help generate the next one.

## 23. What Should I Say If The Teacher Asks Why We Chose This Model?

You can say:

> We compared several model families on the same time-ordered validation block. The linear autoregressive model had the best overall recursive RMSE on the 48-month validation period, while staying simple and stable on a relatively small seasonal dataset.

## 24. What Should I Say If The Teacher Asks Why Not LSTM?

You can say:

> We did explore more complex approaches, but after monthly aggregation we only had 288 points. On that amount of data, the simpler autoregressive model generalized better and was more stable in recursive forecasting. So we selected it because it performed best on unseen validation months, not because it was the most sophisticated-looking model.

## 25. One-Line Memory Aid

If you want one single sentence to remember:

> The model predicts the next month from the last 18 months, the month of the year, and recent rolling averages, then repeats that step recursively to forecast the future.
