

def all_feature_models_validation(model_data, all_features_list, test_data_percentage=0.3,
                                  is_decision_tree=1, is_random_forest=1, is_naive_bayes=1, is_neural_network=1):
    import pandas as pd
    from data_prep import data_prep
    import sys
    sys.path.insert(0, 'classifiers')
    from decision_tree import decision_tree
    from random_forest import random_forest
    from naive_bayes import naive_bayes
    from neural_network import neural_network

    print('EXECUTING FUNCTION: all_feature_models_validation')
    model_compare_df = pd.DataFrame(columns=['class_algo', 'target_variable',
                                             'train_data_count', 'valid_data_count',
                                             'accuracy', 'train_time_s', 'predict_time_s'])
    loop_len = len(all_features_list)
    for i in range(loop_len):
        fmv_target_variable = all_features_list[i]
        fmv_features = list(all_features_list)
        del fmv_features[i]

        print("all_feature_models_validation::Training for and validating: "+fmv_target_variable)
        print("all_feature_models_validation::Splitting data into test and validation...")
        fmv_train_test_data = data_prep(model_data, fmv_features, fmv_target_variable, test_data_percentage)

        if is_decision_tree == 1:
            print("all_feature_models_validation::calling decision tree...")
            fmv_decision_tree_output = decision_tree(fmv_train_test_data[0],
                                                     fmv_train_test_data[1],
                                                     fmv_train_test_data[2],
                                                     fmv_train_test_data[3],
                                                     fmv_train_test_data[4])
            model_accuracy = fmv_decision_tree_output[0][fmv_decision_tree_output[0].find('accuracy'):(fmv_decision_tree_output[0].find('accuracy') + 39)][-4:]
            df_input_list = ['decision tree', fmv_target_variable,
                             (len(fmv_train_test_data[0])), (len(fmv_train_test_data[3])),
                             model_accuracy, (round((fmv_decision_tree_output[2][0]), 3)),
                             (round((fmv_decision_tree_output[2][1]), 3))]
            a_series = pd.Series(df_input_list, index=model_compare_df.columns)
            model_compare_df = model_compare_df.append(a_series, ignore_index=True)
            print('all_feature_models_validation::model accuracy: '+str(
                model_accuracy)+' with target variable: '+fmv_target_variable)

        if is_random_forest == 1:
            print("all_feature_models_validation::calling random forest...")
            fmv_random_forest_output = random_forest(fmv_train_test_data[0],
                                                     fmv_train_test_data[1],
                                                     fmv_train_test_data[2],
                                                     fmv_train_test_data[3],
                                                     fmv_train_test_data[4])
            model_accuracy = fmv_random_forest_output[0][fmv_random_forest_output[0].find('accuracy'):(fmv_random_forest_output[0].find('accuracy') + 39)][-4:]
            df_input_list = ['random forest', fmv_target_variable,
                             (len(fmv_train_test_data[0])), (len(fmv_train_test_data[3])),
                             model_accuracy, (round((fmv_random_forest_output[2][0]), 3)),
                             (round((fmv_random_forest_output[2][1]), 3))]
            a_series = pd.Series(df_input_list, index=model_compare_df.columns)
            model_compare_df = model_compare_df.append(a_series, ignore_index=True)
            print('all_feature_models_validation::model accuracy: ' + str(
                model_accuracy) + ' with target variable: ' + fmv_target_variable)

        if is_naive_bayes == 1:
            print("all_feature_models_validation::calling naive bayes...")
            fmv_naive_bayes_output = naive_bayes(fmv_train_test_data[0],
                                                 fmv_train_test_data[1],
                                                 fmv_train_test_data[2],
                                                 fmv_train_test_data[3],
                                                 fmv_train_test_data[4])
            model_accuracy = fmv_naive_bayes_output[0][fmv_naive_bayes_output[0].find('accuracy'):(fmv_naive_bayes_output[0].find('accuracy') + 39)][-4:]
            df_input_list = ['naive bayes', fmv_target_variable,
                             (len(fmv_train_test_data[0])), (len(fmv_train_test_data[3])),
                             model_accuracy, (round((fmv_naive_bayes_output[2][0]), 3)),
                             (round((fmv_naive_bayes_output[2][1]), 3))]
            a_series = pd.Series(df_input_list, index=model_compare_df.columns)
            model_compare_df = model_compare_df.append(a_series, ignore_index=True)
            print('all_feature_models_validation::model accuracy: ' + str(
                model_accuracy) + ' with target variable: ' + fmv_target_variable)

        if is_neural_network == 1:
            print("all_feature_models_validation::calling neural network...")
            fmv_neural_network_output = neural_network(fmv_train_test_data[0],
                                                       fmv_train_test_data[1],
                                                       fmv_train_test_data[2],
                                                       fmv_train_test_data[3],
                                                       fmv_train_test_data[4])
            model_accuracy = fmv_neural_network_output[0][fmv_neural_network_output[0].find('accuracy'):(fmv_neural_network_output[0].find('accuracy') + 39)][-4:]
            df_input_list = ['neural network', fmv_target_variable,
                             (len(fmv_train_test_data[0])), (len(fmv_train_test_data[3])),
                             model_accuracy, (round((fmv_neural_network_output[2][0]), 3)),
                             (round((fmv_neural_network_output[2][1]), 3))]
            a_series = pd.Series(df_input_list, index=model_compare_df.columns)
            model_compare_df = model_compare_df.append(a_series, ignore_index=True)
            print('all_feature_models_validation::model accuracy: ' + str(
                model_accuracy) + ' with target variable: ' + fmv_target_variable)

    return model_compare_df
