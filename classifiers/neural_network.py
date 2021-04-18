# Neural Network

def neural_network(x_train, x_test, y_train, y_test, label_encoder):
    from sklearn.neural_network import MLPClassifier
    import time
    import pandas as pd
    from sklearn import metrics                         # Import scikit-learn metrics module for accuracy calculation
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification

    clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, warm_start=True)

    # Train Neural Network Classifier
    start_time = time.time()
    # clf = pipe.fit(x_train, y_train)
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

    data = {'y_Actual': actual_values,
            'y_Predicted': predicted_values}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    return [accuracy_stats, confusion_matrix, [train_time, predict_time]]

