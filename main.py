import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestRegressor


def predict():
    """Choose a model and predict using the data in diabetes.csv."""
    di = pd.read_csv('diabetes.csv')
    # di[pandas.isnull(di).any(axis=1)]
    y = di.Outcome.copy()
    X = di.drop(['Outcome'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test))
    y_test = y_test.reset_index(drop = True)
    z = pd.concat([y_test, y_pred], axis = 1)
    z.columns = ['True', 'Prediction']
    z.head()
    print_stats(y_pred, y_test)
    plot_graph(y_pred, y_test)


def plot_graph(y_pred, y_test):
    '''Plot a heatmap to visualize ta confustion matrix.'''
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y = 1.1)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


def print_stats(y_pred, y_test):
    """Prints accuracy, precision, recall for the model used."""
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))


if __name__ == "__main__":
    predict()