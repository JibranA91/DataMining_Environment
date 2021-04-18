# this function will split the input data into test/train for the target variable and features
# it will output a list that contains train and test dataframes for the dependent and independent variables
# as part of the list, it also outputs the label_encoder which is used to get back the encoded values
def data_prep(input_data, features, target_variable, test_data_percentage=0.3):

    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    features_encoded = input_data[features].apply(lambda x: x.factorize()[0])
    label_encoder = preprocessing.LabelEncoder()
    target_encoded = label_encoder.fit_transform(input_data[target_variable].values.ravel())

    x_train, x_test, y_train, y_test = train_test_split(features_encoded,
                                                        target_encoded,
                                                        test_size=test_data_percentage)
    return [x_train, x_test, y_train, y_test, label_encoder]
