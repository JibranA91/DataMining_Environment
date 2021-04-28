#Random Forest

def random_forest(x_train, x_test, y_train, y_test, label_encoder, tune_hyperparameters=0):
    import time
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    import seaborn as sb
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    print("random_forest:: Executing Random Forest")
    img = BytesIO()
    sb.set_style("dark")
    best_parameters = "None"

    if tune_hyperparameters == 1:
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        import numpy as np

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [200, 500, 700, 1000]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [100, 200, 400]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print("random_forest:: "+str(random_grid))

        clf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=clf,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       cv=2,
                                       verbose=2,
                                       n_jobs=-1)
        # Hyperparameter Tuning
        start_time = time.time()
        rf_random.fit(x_train, y_train)
        train_time = time.time() - start_time

        # Predicting
        best_random = rf_random.best_estimator_
        start_time = time.time()
        y_pred = best_random.predict(x_test)
        predict_time = time.time() - start_time

        best_parameters = rf_random.best_params_
        print("random_forest:: "+str(best_parameters))

    else:
        # Model Prep
        clf = RandomForestClassifier(n_estimators=250, criterion="entropy", oob_score=True, warm_start=True)

        # Train the model using the training sets
        start_time = time.time()
        clf.fit(x_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred = clf.predict(x_test)
        predict_time = time.time() - start_time

        best_parameters = "None"

    # Model Prediction Statistics

    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred))
    accuracy_stats = metrics.classification_report(label_encoder.inverse_transform(y_test),
                                                   label_encoder.inverse_transform(y_pred),
                                                   output_dict=True)
    accuracy_stats = pd.DataFrame(accuracy_stats).transpose()
    print("random_forest::")
    print(accuracy_stats)

    # Confusion Matrix
    actual_values = list(label_encoder.inverse_transform(y_test))
    predicted_values = list(label_encoder.inverse_transform(y_pred))

    data = {'y_Actual':    actual_values,
            'y_Predicted': predicted_values
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sb.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='g')
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    cm_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return [accuracy_stats, cm_plot_url, [train_time, predict_time], best_parameters]



