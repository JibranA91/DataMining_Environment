# K-Nearest Neighbors (K-NN)

def k_nn(x_train, x_test, y_train, y_test, label_encoder, n=10):

    import time
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    from math import sqrt
    import numpy as np

    import seaborn as sb
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    img = BytesIO()
    sb.set_style("dark")

    print("knn:: Executing K-Nearest Neighbor")
    knn = KNeighborsClassifier(n)

    # Train the KNN model
    start_time = time.time()
    knn.fit(x_train, y_train)
    train_time = time.time() - start_time

    # Predict the responses from the model
    start_time = time.time()
    y_pred = knn.predict(x_test)
    predict_time = time.time() - start_time

    # Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual':    actual_values,
            'y_Predicted': predicted_values
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'],
                                   rownames=['Actual'], colnames=['Predicted'])

    sb.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='g')
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    cm_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Accuracy Chart
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred))
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred),
                                                   output_dict=True)
    accuracy_stats = pd.DataFrame(accuracy_stats).transpose()

    # checking error for multiple values of k
    error_val = []
    k_val = []
    min_n = max(1, n-10)
    max_n = min_n + 10
    print("knn::Evaluating k_nn for n_neighbors from "+str(min_n)+" to "+str(max_n))
    for K in range(min_n, max_n):
        K = K+1
        knn = KNeighborsClassifier(n_neighbors=K)

        knn.fit(x_train, y_train)                # fit the model
        y_pred = knn.predict(x_test)             # make prediction on test set
        error = np.mean(y_pred != y_test)        # calculate error
        error_val.append(error)                  # store error values
        k_val.append(K)
        print('knn::Error value for k =', K, 'is:', error)

    plt.plot(k_val, error_val, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)       # elbow curve
    plt.title("Error Elbow Curve")
    plt.xlabel("N_Neighbors")
    plt.ylabel("Mean Error")
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    knn_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return [accuracy_stats, cm_plot_url, [train_time, predict_time], knn_plot_url]

