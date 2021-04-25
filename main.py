from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from support_functions.correl_matrix import correl_matrix
from support_functions.pull_data import pull_data
from support_functions.data_prep import data_prep
from support_functions.model_compare import all_feature_models_validation
from support_functions.sortedgroupbar import sortedgroupedbar
from support_functions.data_summary import data_summary
from classifiers.random_forest import random_forest
from classifiers.support_vector_machine import  support_vm
from classifiers.naive_bayes import naive_bayes
from classifiers.neural_network import neural_network
from classifiers.decision_tree import decision_tree
from classifiers.knn import k_nn


mydata = None
feature_list = None
active_data = None
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("homepage.html")


@app.route('/feature_select', methods=['POST'])
def feature_select():
    global mydata
    uploaded_file = request.files['myfile']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    mydata = pull_data(uploaded_file.filename)
    column_names = list(mydata.columns)

    data_info = data_summary(mydata).to_html(index=False)

    return render_template("feature_select.html", column_names=column_names, data_info=data_info)


@app.route('/feature_select/datamine', methods=['GET', 'POST'])
def datamine():

    global feature_list, active_data

    if request.method == 'GET':
        if feature_list is None or active_data is None:
            return "<h1> Method Not Allowed </h1>"
        else:
            c_matrix = correl_matrix(active_data)
            return render_template("datamine.html", active_features=feature_list, c_matrix=c_matrix)

    if request.method == 'POST':
        x_axis = request.form.get('x_axis')
        y_axis = request.form.get('y_axis')
        split_on = request.form.get('split_on')

        if feature_list is None or active_data is None:
            feature_list = request.form.getlist('column_name')
            active_data = mydata[feature_list]
            if feature_list is None or active_data is None:
                return "<h1> Method Not Allowed </h1>"

        c_matrix = correl_matrix(active_data)
        rel_chart = 0
        if x_axis is not None and y_axis is not None and split_on is not None:

            unique_x = len(active_data[x_axis].unique())
            try:
                unique_split_on = len(active_data[split_on].unique())
            except KeyError:
                unique_split_on = 1

            if unique_x > 15 or unique_split_on > 15:
                return render_template("datamine.html", active_features=feature_list, c_matrix=c_matrix,
                                       rel_chart='Too Many Unique Values In X-Axis or Split-On Value')

            fig, ax = plt.subplots()
            rel_chart = sortedgroupedbar(ax, x=x_axis, y=y_axis, groupby=split_on, data=active_data)

        return render_template("datamine.html", active_features=feature_list, c_matrix=c_matrix, rel_chart=rel_chart)


@app.route('/feature_select/datamine/classify', methods=['POST'])
def classify():
    global feature_list, active_data
    feature_list = feature_list
    algorithm = request.form.get('datamine_algo')
    target_variable = request.form.get('dep_var')
    features = request.form.getlist('indep_var')
    train_test_split = request.form.get('train_test_split')
    tune_hyperparameters = request.form.get('tune_hyperparameters')
    tune_hyperparameters = "0" if tune_hyperparameters is None else tune_hyperparameters
    test_data_percentage = round((1.0 - int(train_test_split) / 100), 3)

    print('classify::splitting data into train_test...')
    train_test_data = data_prep(active_data, features, target_variable, test_data_percentage)

    print('classify::calling classification algorithm...')
    classifier_output = eval(algorithm + "(train_test_data[0],"
                                         "train_test_data[1],"
                                         "train_test_data[2],"
                                         "train_test_data[3],"
                                         "train_test_data[4]" +
                             ((',' + tune_hyperparameters) if algorithm == 'random_forest' else '') + ")")

    accuracy_stats = classifier_output[0].round(3).to_html()
    confusion_matrix = classifier_output[1]  # confusion matrix
    train_time = str(round((classifier_output[2][0]), 3))
    predict_time = str(round((classifier_output[2][1]), 3))
    best_parameters = str(classifier_output[3])
    best_parameters = best_parameters.replace(",", ",<br />", 15)
    print(best_parameters)

    algorithm = algorithm. \
        replace('decision_tree', 'Decision Tree'). \
        replace('naive_bayes', 'Naive Bayes'). \
        replace('random_forest', 'Random Forest'). \
        replace('neural_network', 'Neural Network').\
        replace('k_nn', 'K Nearest N').\
        replace('support_vm', 'Support Vector M')

    return render_template("classify.html",
                           active_features=feature_list, algorithm=algorithm,
                           target_variable=target_variable, features=features,
                           accuracy_stats=accuracy_stats, confusion_matrix=confusion_matrix,
                           train_time=train_time, predict_time=predict_time,
                           best_parameters=best_parameters)


@app.route('/feature_select/datamine/compare_models', methods=['GET', 'POST'])
def compare_models():
    global feature_list, active_data

    if request.method == 'GET':
        return render_template("compare_models.html",
                               active_features=feature_list,
                               model_compare_html='None')

    if request.method == 'POST':

        feature_list = feature_list
        is_decision_tree = request.form.get('is_decision_tree')
        is_random_forest = request.form.get('is_random_forest')
        is_naive_bayes = request.form.get('is_naive_bayes')
        is_neural_network = request.form.get('is_neural_network')
        features = request.form.getlist('indep_var')
        train_test_split = request.form.get('train_test_split')
        test_data_percentage = round((1.0 - int(train_test_split) / 100), 3)
        model_compare_df = all_feature_models_validation(active_data, features,
                                                         test_data_percentage=test_data_percentage,
                                                         is_decision_tree=is_decision_tree,
                                                         is_random_forest=is_random_forest,
                                                         is_naive_bayes=is_naive_bayes,
                                                         is_neural_network=is_neural_network)
        model_compare_html = model_compare_df.to_html(classes='table table-striped', index=False)

        fig, ax = plt.subplots()
        accuracy_variable = sortedgroupedbar(ax, x='target_variable', y='accuracy', groupby='class_algo',
                                             data=model_compare_df, is_data_grouped=1, xlabelrotation=90)

        fig, ax = plt.subplots()
        traintime_variable = sortedgroupedbar(ax, x='target_variable', y='train_time_s', groupby='class_algo',
                                              data=model_compare_df, is_data_grouped=1, xlabelrotation=90)

        return render_template("compare_models.html",
                               active_features=feature_list,
                               model_compare_html=model_compare_html,
                               accuracy_variable=accuracy_variable,
                               traintime_variable=traintime_variable)


if __name__ == '__main__':
    app.run(debug=True)
