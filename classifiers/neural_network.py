# Neural Network

def neural_network(x_train, x_test, y_train, y_test, label_encoder):
    from sklearn.neural_network import MLPClassifier
    import time
    import pandas as pd
    from sklearn import metrics                         # Import scikit-learn metrics module for accuracy calculation
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification

    import seaborn as sb
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    img = BytesIO()
    sb.set_style("dark")

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
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred),
                                                   output_dict=True)
    accuracy_stats = pd.DataFrame(accuracy_stats).transpose()

    # Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual': actual_values,
            'y_Predicted': predicted_values}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sb.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='g')
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    cm_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return [accuracy_stats, cm_plot_url, [train_time, predict_time]]

