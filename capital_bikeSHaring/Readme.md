## Capital Bikeshare Demand Analysis and Prediction

# 1. Project Overview

This project analyses the Capital Bikeshare dataset to understand bike rental demand, identify the factors influencing usage patterns, and develop machine learning models capable of predicting daily rental demand.

The analysis uses two datasets:

- "day.csv": Daily rental records.
- "hour.csv": Hourly rental records.

The project follows an end-to-end data science workflow, covering data inspection, cleaning, exploratory data analysis (EDA), feature engineering, baseline modelling, model development, evaluation, hyperparameter tuning and time-series cross-validation.

# 2. Business Problem

Bike-sharing operators need to understand demand patterns to make better operational decisions.

Accurate demand estimates can support:

- Bicycle availability and fleet allocation.
- Operational and maintenance planning.
- Identification of peak demand periods.
- Resource allocation across different seasons.
- Better understanding of customer usage behaviour.

The primary objective was to identify the factors associated with rental demand and determine how accurately machine learning models could predict daily bike rentals.

# 3. Dataset Description

The datasets contain records of bike-sharing activity over 2011 and 2012.

Dataset| Observations| Features
Daily ("day.csv")| 731| 16
Hourly ("hour.csv")| 17,379| 17

The target variable was "cnt", representing the total number of bike rentals.

Other important variables included:

- "temp": Normalised temperature.
- "atemp": Normalised feels-like temperature.
- "hum": Normalised humidity.
- "windspeed": Normalised wind speed.
- "season": Season of the year.
- "weathersit": Weather conditions.
- "workingday": Whether the day was a working day.
- "holiday": Whether the day was a holiday.
- "yr": Year indicator.
- "mnth": Month.
- "weekday": Day of the week.

The target "cnt" is the sum of "casual" and "registered" rentals. Both component columns were excluded from the predictive features to prevent target leakage.

# 4. Data Inspection and Cleaning

Initial inspection revealed:

- No missing values in either dataset.
- No exact duplicate records.
- The "dteday" column was converted to datetime format.
- Temperature, feels-like temperature, humidity and wind speed were already normalised between 0 and 1.
- Binary variables were represented as 0 and 1.
- Seasonal and weather categories were represented numerically.

Additional date-based features were extracted, including year, month and day.

The daily dataset was sorted chronologically before the train-test split.

# 5. Exploratory Data Analysis

# 5.1 Daily versus hourly demand distribution

The daily rental count had a skewness of approximately -0.047, indicating an approximately symmetrical distribution.

The hourly rental count had a skewness of approximately 1.277, indicating a positively skewed distribution.

This difference reflects the nature of the datasets: daily records aggregate demand over an entire day, while hourly records capture demand within individual hours.

# 5.2 Customer behaviour

Average daily rentals were approximately:

- Registered users: 3,656.
- Casual users: 848.
- Total rentals: 4,504.

Registered users accounted for approximately 81% of rentals, while casual users accounted for approximately 19%.

This indicates that the service was used predominantly by registered customers.

# 5.3 Seasonal demand

Seasonal analysis revealed differences in rental activity:

- Spring recorded the lowest average demand.
- Summer experienced higher demand.
- Autumn recorded the highest average demand.
- Winter experienced a decline compared with autumn.

This suggests that seasonal conditions are relevant to bike-sharing demand.

# 5.4 Weather conditions

Average rental demand decreased as weather conditions deteriorated.

Clear and partly cloudy conditions recorded the highest average demand, while light rain and snow conditions were associated with substantially lower rentals.

This suggests that weather conditions influence customers' willingness to use the service.

# 5.5 Hourly demand patterns

Hourly analysis revealed distinct demand peaks:

- 8:00 am was a major morning peak.
- 5:00 pm recorded the highest evening demand.
- 6:00 pm also experienced substantial activity.

Registered users showed strong morning and evening commuter peaks.

Casual users recorded their highest demand around 3:00 pm, suggesting a greater association with leisure or flexible daytime usage.

# 5.6 Temperature and rental demand

Temperature showed a positive relationship with rental demand.

The Pearson correlation coefficients were:

Feature| Correlation with "cnt"
"temp"| 0.627
"atemp"| 0.631

Both variables demonstrated moderately strong positive linear relationships with rental demand.

Scatter plot analysis also suggested that the relationship may not be entirely linear, with demand increasing at moderate temperatures and appearing to level off at higher temperatures.

However, temperature alone could not explain all variations in rental activity.

# 5.7 Correlation analysis

Other notable correlations with daily rental demand included:

Feature| Correlation
"registered"| 0.94
"casual"| 0.67
"atemp"| 0.63
"temp"| 0.62
"yr"| 0.56
"season"| 0.40
"weathersit"| -0.29
"weekday"| 0.06
"workingday"| 0.061

The strong correlations involving "casual" and "registered" were expected because they are components of the target variable itself. They were therefore excluded from modelling.

The correlation between temperature and feels-like temperature also indicated substantial overlap in the information contained in these features.

# 6. Feature Engineering and Preprocessing

The following preprocessing decisions were made:

- Converted date columns to datetime.
- Extracted additional date components.
- Excluded "instant", "dteday", "casual", "registered" and "cnt" from the predictor set.
- Used a chronological 80/20 train-test split.
- Applied "StandardScaler" to continuous numerical variables for Linear Regression.
- Applied "OneHotEncoder" to categorical variables for Linear Regression.
- Passed binary variables through without scaling.

The Linear Regression model used "temp" while excluding "atemp" to reduce multicollinearity.

The Random Forest model used the numerical feature set without the same preprocessing pipeline.

# 7. Model Development and Evaluation

Three regression approaches were evaluated.

# 7.1 Dummy Regressor

A Dummy Regressor using the mean strategy was established as the baseline.

It predicts the mean rental count from the training data regardless of the input features.

Results:

Metric| Score
MAE| 2,290.41
RMSE| 2,560.55
R²| -0.865

The negative R² indicated that the baseline performed worse than using the mean of the test observations as predictions.

This established a benchmark that the actual predictive models needed to outperform.

# 7.2 Linear Regression

Linear Regression was developed using a preprocessing pipeline involving standardisation, one-hot encoding and linear regression.

Results:

Metric| Score
MAE| 804
RMSE| 1,103
R²| 0.653

The model substantially outperformed the baseline.

An R² of 0.653 indicates that the model explained approximately 65.3% of the variation in rental demand on the test set.

# 7.3 Random Forest Regressor

A Random Forest Regressor was developed using 100 decision trees.

Initial results:

Metric| Score
MAE| 911
RMSE| 1,114
R²| 0.6464

The model performed substantially better than the baseline, although its initial performance was slightly below Linear Regression.

# 8. Feature Importance

Random Forest feature importance identified the following leading predictors:

Feature: Importance
"temp": 0.29
"atemp": 0.26
"year": 0.13
"yr": 0.11

Together, these features accounted for approximately 79% of the model's reported feature importance.

Temperature and feels-like temperature were the leading predictors.

However, the overlap between "temp" and "atemp", as well as "year" and "yr", means their individual importance values should be interpreted cautiously.

Feature importance indicates predictive usefulness, not causation.

# 9. Hyperparameter Tuning

Manual hyperparameter experiments were conducted to investigate whether Random Forest performance could be improved.

The following parameters were explored:

- "n_estimators": Number of trees.
- "max_depth": Maximum tree depth.
- "min_samples_split": Minimum observations required to split a node.
- "min_samples_leaf": Minimum observations required in a leaf.

Increasing the number of trees did not produce a meaningful improvement.

The best observed configuration was:

- "n_estimators = 100"
- "max_depth = 15"
- "min_samples_split = 10"
- "min_samples_leaf = 1"

Results:

Metric| Tuned Random Forest
MAE| 886.71
RMSE| 1,097.02
R²| 0.6575

The tuned model marginally outperformed Linear Regression on the test set.

The improvement was relatively small, indicating that extensive parameter adjustments did not substantially change predictive performance.

# 10. Overfitting Assessment

The tuned Random Forest was evaluated on both training and test data.

Dataset| R²
Training| 0.96
Testing| 0.6575

The difference between the scores indicates a substantial generalisation gap.

The model fitted the training observations very closely but performed less effectively on unseen observations.

This suggested overfitting.

Increasing "min_samples_leaf" did not improve the test results, with the default value of 1 remaining the best observed setting.

# 11. Time-Series Cross-Validation

To assess performance across different chronological periods, five-fold expanding-window "TimeSeriesSplit" cross-validation was conducted on the training portion of the dataset.

The folds preserved chronological order:

Fold: Training indices: Validation indices
1: 0–98: 99–195
2: 0–195: 196–292
3: 0–292: 293–389
4: 0–389: 390–486
5: 0–486: 487–583

The resulting R² scores were approximately:

- Fold 1: -3.12
- Fold 2: 0.2148
- Fold 3: -0.751
- Fold 4: -0.625
- Fold 5: 0.067

Mean cross-validation R²: -0.8435

Standard deviation: approximately 1.2.

The results revealed substantial variation in predictive performance across validation periods.

Three of the five folds produced negative R² values, indicating that the model performed worse than the validation-period mean benchmark in those folds.

This showed that the model's favourable performance on the original test set did not translate consistently across the earlier validation periods.

# 12. Final Model Comparison

Model| MAE| RMSE| R²
Dummy Regressor| 2,290.41| 2,560.55| -0.865
Linear Regression| 804| 1,103| 0.653
Initial Random Forest| 911| 1,114| 0.6464
Tuned Random Forest| 886.71| 1,097.02| 0.6575

The tuned Random Forest achieved the highest test-set R², although its improvement over Linear Regression was marginal.

# 13. Conclusion

The Capital Bikeshare analysis demonstrated that rental demand is associated with several factors, particularly temperature, seasonality, weather conditions and time of day.

Exploratory analysis revealed that registered users accounted for the majority of rentals, with distinct commuter-hour peaks, while casual users showed stronger afternoon activity. Demand also varied considerably across seasons and weather conditions.

Temperature emerged as one of the most influential predictive variables, with both actual and feels-like temperature showing positive correlations with rental demand. However, the scatter plot suggested that the relationship was not perfectly linear, reinforcing the importance of investigating non-linear patterns during model development.

The modelling stage established a strong baseline comparison. Both Linear Regression and Random Forest substantially outperformed the Dummy Regressor. Linear Regression achieved an R² of 0.653, while the tuned Random Forest achieved 0.6575, with MAE and RMSE of 886.71 and 1,097.02 respectively.

Although Random Forest marginally outperformed Linear Regression on the original test set, further evaluation revealed a substantial difference between training and test performance. Its training R² of 0.96 compared with a test R² of 0.6575 indicated overfitting. More importantly, time-series cross-validation produced a negative mean R² of approximately -0.8435, with considerable variation between validation periods.

Consequently, the tuned Random Forest should not be considered a consistently reliable forecasting model based on the available evaluation. The results demonstrate why a single train-test split is insufficient for evaluating time-dependent data and why chronological cross-validation is important.

Overall, the project provided practical experience in regression modelling, feature engineering, exploratory analysis, feature importance, hyperparameter tuning, overfitting assessment and time-series validation. It also demonstrated that a slightly better test-set score does not necessarily indicate a more reliable model.

# 14. Limitations and Further Work

Potential improvements include:

1. Investigating the inconsistent cross-validation results and identifying the periods responsible for poor performance.
2. Ensuring that both models use comparable predictor information for a fair comparison.
3. Exploring cyclical encoding for month and weekday.
4. Investigating seasonal and weather interactions.
5. Evaluating alternative models such as Gradient Boosting, XGBoost and LightGBM.
6. Considering separate hourly demand forecasting.
7. Using additional chronological validation periods and carefully selected forecasting metrics.

# 15. Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

Project status: Completed — exploratory analysis, baseline modelling, regression modelling, hyperparameter experiments and chronological validation.
