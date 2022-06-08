import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Import and initial visualization of data
df = pd.read_csv('fetal_health.csv')


def plot_counts(col):
    plt.hist(df[col])
    plt.title(f'{col} distribution')
    plt.show()


def bar_corr(df, col):
    vals = []
    unique = df['fetal_health'].unique().tolist()
    unique.sort()
    for i in unique:
        subset = df[df.fetal_health == i]
        vec = subset[col]
        vals.append(vec.mean())
    plt.bar(unique, vals)
    plt.xticks(ticks = [1, 2, 3], labels=['Normal', 'Suspect', 'Pathological'])
    plt.title(f'Average {col} by fetal health label')
    plt.show()


def scatter(c1, c2):
    plt.scatter(df[c1], df[c2])
    plt.title(f'{c1} vs. {c2}')
    plt.show()


def explore(col):
    plot_counts(col)
    bar_corr(col)


df = df[['accelerations', 'uterine_contractions', 'abnormal_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'fetal_health']]

df.fetal_health.replace({1. : 'normal', 2. : 'suspect', 3. : 'pathological'}, inplace=True)



# Oversampling with imblearn smote

X = df.drop('fetal_health', axis=1)
y = df.fetal_health


smote = SMOTE(sampling_strategy='not majority', random_state=42)
X_ov, y_ov = smote.fit_resample(X, y)


#comparing the distributions of the input variables before and after resampling
def compare_samples(col):
    plt.hist(X[col])
    plt.title(f'{col} distribution actual')
    plt.show()
    plt.hist(X_ov[col])
    plt.title(f'{col} distribution resampled')
    plt.show()


resampled_df = pd.concat([X_ov, y_ov], axis=1)
resampled_df.to_csv('resampled_data.csv')
