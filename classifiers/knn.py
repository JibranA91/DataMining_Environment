# K-Nearest Neighbors (K-NN)

def k_nn(x_train, x_test, y_train, y_test, label_encoder):

    import time
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    import seaborn as sn

    knn = KNeighborsClassifier(n_neighbors=10)

    # Train the KNN model
    start_time = time.time()
    knn.fit(x_train, y_train)
    train_time = time.time() - start_time

    # Predict the responses from the model
    start_time = time.time()
    y_pred = knn.predict(x_test)
    predict_time = time.time() - start_time

    # Accuracy stats
    print(metrics.classification_report(label_encoder.inverse_transform(y_test),
                                        label_encoder.inverse_transform(y_pred)))

    # Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual':    actual_values,
            'y_Predicted': predicted_values
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'],
                                   rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='g')

    # Accuracy Chart
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred))
    accuracy_val = []  # to store rmse values for different k
    for K in range(10):
        K = K+1
        knn = KNeighborsClassifier(n_neighbors=K)

        knn.fit(x_train, y_train)                           # fit the model
        y_pred = knn.predict(x_test)                        # make prediction on test set
        accuracy = metrics.accuracy_score(y_test, y_pred)   # calculate rmse
        accuracy_val.append(accuracy)                       # store rmse values
        print('Accuracy value for k =', K, 'is:', accuracy)

    curve = pd.DataFrame(accuracy_val)  # elbow curve
    curve.plot()

    return [accuracy_stats, confusion_matrix, [train_time, predict_time]]

