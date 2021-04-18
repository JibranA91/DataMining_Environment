# Decision Tree
from data_prep import data_prep


def decision_tree(x_train, x_test, y_train, y_test, label_encoder):
    import time
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier     # Import Decision Tree Classifier
    from sklearn import metrics                         # Import scikit-learn metrics module for accuracy calculation

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifier
    start_time = time.time()
    clf = clf.fit(x_train, y_train)
    train_time = time.time() - start_time

    # Predict the response for test dataset
    start_time = time.time()
    y_pred = clf.predict(x_test)
    predict_time = time.time() - start_time

    # Accuracy stats
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred))

    # Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual':    actual_values,
            'y_Predicted': predicted_values}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    return [accuracy_stats, confusion_matrix, [train_time, predict_time]]
