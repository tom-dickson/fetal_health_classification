import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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



# Fitting Gaussian NB on scaled data
clf = GaussianNB()
clf.fit(train_scaled, y_train)


predicted = clf.predict(test_scaled)
probs = clf.predict_proba(test_scaled)


# Evaluating the model
print("Confusion matrix:", confusion_matrix(y_test, predicted))
print("Report:", classification_report(y_test, predicted))
print("Accuracy", accuracy_score(y_test, predicted))


# Constructing data frame with test set probabilities/predictions/actuals
prob_df = pd.DataFrame(probs, columns=clf.classes_)
prob_df['predicted'] = predicted
prob_df['actual'] = y_train


def plot_confusion():
    cm = confusion_matrix(y_test, predicted, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
    disp.plot()
    plt.show()


def prob_bars(label1):
    vals = []
    subset = prob_df[(prob_df['actual'] == label1)]
    for col in subset.columns[0:3]:
        ave = subset[col].mean()
        vals.append(ave)
    plt.bar(subset.columns[0:3], vals)
    plt.show()
