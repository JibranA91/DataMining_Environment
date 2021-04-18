#Random Forest

def random_forest(x_train, x_test, y_train, y_test, label_encoder):
    import time
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    # Model Prep
    clf = RandomForestClassifier(n_estimators=250, criterion="entropy", oob_score=True, warm_start=True)

    # Train the model using the training sets
    start_time = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = clf.predict(x_test)
    predict_time = time.time() - start_time

    # Model Prediction Statistics

    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred))

    #Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual':    actual_values,
            'y_Predicted': predicted_values
            }

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    return [accuracy_stats, confusion_matrix, [train_time, predict_time]]



