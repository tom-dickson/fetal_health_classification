# Fetal health classification project

This repository contains my work in analyzing the Fetal Health Classification dataset I found on Kaggle (at https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

# In the "Updated" folder
The data.py file contains code where I collect and explore the data, select features, and resample to correct the large imbalance
in the original dataset. Then, classification.py contains my implementation of Gaussian naive Bayes and Random forest algorithms for classifying fetal health
based on the monitored health signals. 

The Random Forest model performed the best, predicting the outcome category with about 90% accuracy and precision 

# Other files
The feature selection folder has some graphs that assist in visualizing features from the original data to select. Both the original and resampled data sets are included in csv form

Going forward, I may look in to other classification strategies and do some hyperparameter tuning to see if the performance of the 
model can be improved further.
