import seaborn as sn
from data_prep import data_prep
import sys
from model_compare import all_feature_models_validation

sys.path.insert(0, 'classifiers')
from decision_tree import decision_tree
from random_forest import random_forest
from naive_bayes import naive_bayes
from neural_network import neural_network


def pull_data(csv_path):
    import pandas as pd
    song_attributes = pd.read_csv(csv_path)
    return song_attributes


if __name__ == '__main__':
    print('pulling data from file...')
    song_attributes = pull_data('data/msd_combined.csv')

    all_features_list = ['danceability', 'gender', 'genre_dortmund', 'genre_electronic',
                         'genre_rosamerica', 'genre_tzanetakis', 'ismir04_rhythm',
                         'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy',
                         'mood_party', 'mood_relaxed', 'mood_sad', 'moods_mirex', 'timbre',
                         'tonal_atonal', 'voice_instrumental']

    compare_df = all_feature_models_validation(song_attributes, all_features_list)

    """
    features = ['mood_party', 'genre_dortmund',
                'genre_electronic', 'mood_acoustic', 'mood_aggressive', 'mood_electronic',
                'mood_happy', 'mood_relaxed', 'mood_sad', 'moods_mirex',
                'timbre', 'tonal_atonal', 'gender', 'voice_instrumental']
    target_variable = ['danceability']
    test_data_percentage = 0.3
    print('splitting data into train_test...')
    train_test_data = data_prep(song_attributes, features, target_variable, test_data_percentage)

    print('calling decision tree...')
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





