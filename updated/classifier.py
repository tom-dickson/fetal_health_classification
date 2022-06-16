import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay



# Splitting and scaling the data
df = pd.read_csv('resampled_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

X = df.drop('fetal_health', axis=1)
y = df.fetal_health

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.fit_transform(X_test)


# Fitting classifier onscaled data
clf = RandomForestClassifier()
clf.fit(train_scaled, y_train)

# Testing model on test set
predicted = clf.predict(test_scaled)
probs = clf.predict_proba(test_scaled)


# Evaluating the model
print("Report:", classification_report(y_test, predicted))
print("Accuracy", accuracy_score(y_test, predicted))


# Constructing data frame with test set probabilities/predictions/actuals
prob_df = pd.DataFrame(probs, columns=clf.classes_)
prob_df['predicted'] = predicted
prob_df['actual'] = y_test


# Plots a confusion matrix
def plot_confusion(predictor):
    cm = confusion_matrix(y_test, predicted, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
    disp.plot()
    plt.title(f'{predictor} confusion matrix')
    plt.show()

# Plots the mean predicted probabilities for each outcome caregory
def prob_bars(label1):
    vals = []
    subset = prob_df[prob_df['actual'] == label1]
    print(subset.to_string())
    for col in subset.columns[0:3]:
        ave = subset[col].mean()
        vals.append(ave)
    plt.bar(subset.columns[0:3], vals)
    plt.title(f'Mean Predicted Probabilities when True Label is {label1}')
    plt.show()
