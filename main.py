from flask import Flask, render_template, request
import sys
from support_functions.correl_matrix import correl_matrix
from support_functions.pull_data import pull_data
from support_functions.data_prep import data_prep
from support_functions.model_compare import all_feature_models_validation
from classifiers.decision_tree import decision_tree
from classifiers.knn import k_nn
from classifiers.naive_bayes import naive_bayes
from classifiers.neural_network import neural_network
from classifiers.random_forest import random_forest
from classifiers.support_vector_machine import support_vm


sys.path.insert(0, 'classifiers')

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
    return render_template("feature_select.html", column_names=column_names)


@app.route('/feature_select/datamine', methods=['POST'])
def datamine():
    global feature_list, active_data
    feature_list = request.form.getlist('column_name')
    active_data = mydata[feature_list]

    c_matrix = correl_matrix(active_data)
    return render_template("datamine.html", active_features=feature_list, c_matrix=c_matrix)


@app.route('/feature_select/datamine/classify', methods=['POST'])
def classify():
    global feature_list, active_data
    feature_list = feature_list
    algorithm = request.form.get('datamine_algo')
    target_variable = request.form.get('dep_var')
    features = request.form.getlist('indep_var')
    train_test_split = request.form.get('train_test_split')
    test_data_percentage = round((1.0 - int(train_test_split) / 100), 3)

    print('classify::splitting data into train_test...')
    train_test_data = data_prep(active_data, features, target_variable, test_data_percentage)

    print('classify::calling classification algorithm...')
    classifier_output = eval(algorithm + "(train_test_data[0],"
                                         "train_test_data[1],"
                                         "train_test_data[2],"
                                         "train_test_data[3],"
                                         "train_test_data[4])")

    accuracy_stats = classifier_output[0].to_html()
    print(accuracy_stats)
    confusion_matrix = classifier_output[1]  # confusion matrix
    train_time = str(round((classifier_output[2][0]), 3))
    predict_time = str(round((classifier_output[2][1]), 3))

    algorithm = algorithm. \
        replace('decision_tree', 'Decision Tree'). \
        replace('naive_bayes', 'Naive Bayes'). \
        replace('random_forest', 'Random Forest'). \
        replace('neural_network', 'Neural Network').\
        replace('k_nn', 'K Nearest N').\
        replace('support_vm', 'Support Vector M')

    return render_template("classify.html",
                           active_features=feature_list, algorithm=algorithm,
                           accuracy_stats=accuracy_stats, confusion_matrix=confusion_matrix,
                           train_time=train_time, predict_time=predict_time)


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

        return render_template("compare_models.html",
                               active_features=feature_list,
                               model_compare_html=model_compare_html)


if __name__ == '__main__':
    app.run(debug=True)

    """
    import seaborn as sns
    print('pulling data from file...')
    song_attributes = pull_data('data/msd_combined.csv')

    all_features_list = ['danceability', 'gender', 'genre_dortmund', 'genre_electronic',
                         'genre_rosamerica', 'genre_tzanetakis', 'ismir04_rhythm',
                         'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy',
                         'mood_party', 'mood_relaxed', 'mood_sad', 'moods_mirex', 'timbre',
                         'tonal_atonal', 'voice_instrumental']

    compare_df = all_feature_models_validation(song_attributes, all_features_list)

    
    features = ['mood_party', 'genre_dortmund',
                'genre_electronic', 'mood_acoustic', 'mood_aggressive', 'mood_electronic',
                'mood_happy', 'mood_relaxed', 'mood_sad', 'moods_mirex',
                'timbre', 'tonal_atonal', 'gender', 'voice_instrumental']
    target_variable = ['danceability']
    test_data_percentage = 0.3
    print('splitting data into train_test...')
    train_test_data = data_prep(song_attributes, features, target_variable, test_data_percentage)

    print('calling decision tree...')
    algorithm = 'decision_tree'
    classifier_output = eval(algorithm + "(train_test_data[0],"
                                         "train_test_data[1],"
                                         "train_test_data[2],"
                                         "train_test_data[3],"
                                         "train_test_data[4])")
    decision_tree_output = decision_tree(train_test_data[0],
                                         train_test_data[1],
                                         train_test_data[2],
                                         train_test_data[3],
                                         train_test_data[4])

    print(decision_tree_output[1])  # sn.heatmap(decision_tree_output[1], annot=True, cmap='viridis', fmt='g')
    print(decision_tree_output[0])
    print('Train Time: ' + str(round((decision_tree_output[2][0]), 3)) + ' seconds')
    print('Predict Time: ' + str(round((decision_tree_output[2][1]), 3)) + ' seconds')

    print('calling random forest...')
    random_forest_output = random_forest(train_test_data[0],
                                         train_test_data[1],
                                         train_test_data[2],
                                         train_test_data[3],
                                         train_test_data[4])

    print(random_forest_output[1])  # sn.heatmap(decision_tree_output[1], annot=True, cmap='viridis', fmt='g')
    print(random_forest_output[0])
    print('Train Time: ' + str(round((random_forest_output[2][0]), 3)) + ' seconds')
    print('Predict Time: ' + str(round((random_forest_output[2][1]), 3)) + ' seconds')

    print('calling naive bayes...')
    naive_bayes_output = naive_bayes(train_test_data[0],
                                     train_test_data[1],
                                     train_test_data[2],
                                     train_test_data[3],
                                     train_test_data[4])

    print(naive_bayes_output[1])  # sn.heatmap(decision_tree_output[1], annot=True, cmap='viridis', fmt='g')
    print(naive_bayes_output[0])
    print('Train Time: ' + str(round((naive_bayes_output[2][0]), 3)) + ' seconds')
    print('Predict Time: ' + str(round((naive_bayes_output[2][1]), 3)) + ' seconds')

    print('calling neural network...')
    neural_network_output = neural_network(train_test_data[0],
                                           train_test_data[1],
                                           train_test_data[2],
                                           train_test_data[3],
                                           train_test_data[4])

    print(neural_network_output[1])  # sn.heatmap(decision_tree_output[1], annot=True, cmap='viridis', fmt='g')
    print(neural_network_output[0])
    print('Train Time: ' + str(round((neural_network_output[2][0]), 3)) + ' seconds')
    print('Predict Time: ' + str(round((neural_network_output[2][1]), 3)) + ' seconds')

    """





