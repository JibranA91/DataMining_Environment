#Random Forest

def random_forest(x_train, x_test, y_train, y_test, label_encoder):
    import time
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    import seaborn as sb
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    img = BytesIO()
    sb.set_style("dark")

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
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred),
                                                   output_dict=True)
    accuracy_stats = pd.DataFrame(accuracy_stats).transpose()

    #Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual':    actual_values,
            'y_Predicted': predicted_values
            }

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sb.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='g')
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    cm_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return [accuracy_stats, cm_plot_url, [train_time, predict_time]]



