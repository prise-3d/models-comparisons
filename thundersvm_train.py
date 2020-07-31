# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# models imports
from sklearn.utils import shuffle
from thundersvm import SVC
from sklearn.metrics.scorer import accuracy_scorer
from sklearn.model_selection import GridSearchCV

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

# variables and parameters
n_predict = 0


def main():

    parser = argparse.ArgumentParser(description="Train SKLearn model and save it into .joblib file")

    parser.add_argument('--data', type=str, help='dataset file prefix (without .train and .test)')
    parser.add_argument('--output', type=str, help='output file name desired for model (without .joblib extension)')

    args = parser.parse_args()

    p_data_file = args.data
    p_output    = args.output

    ########################
    # 1. Get and prepare data
    ########################
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)

    final_df_train = pd.concat([not_noisy_df_train[0:nb_noisy_train], noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test[0:nb_noisy_test], noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    final_df_train_size = len(final_df_train.index)
    final_df_test_size = len(final_df_test.index)

    # use of the whole data set for training
    x_dataset_train = final_df_train.iloc[:,1:]
    x_dataset_test = final_df_test.iloc[:,1:]

    y_dataset_train = final_df_train.iloc[:,0]
    y_dataset_test = final_df_test.iloc[:,0]

    #######################
    # 2. Construction of the model : Ensemble model structure
    #######################

    print("-------------------------------------------")
    print("Train dataset size: ", final_df_train_size)


    x_dataset_train = np.array(x_dataset_train, 'float32')
    y_dataset_train = np.array(y_dataset_train, 'uint8')

    def my_accuracy_scorer(*args):
        global n_predict
        score = accuracy_scorer(*args)
        print('{0} - Score is {1}'.format(n_predict, score))
        n_predict += 1
        return score

    def _get_best_model(X_train, y_train):

        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # Cs = [1, 2, 4, 8, 16, 32]
        # gammas = [0.001, 0.01]
        gammas = [0.001, 0.01, 0.1, 1, 5, 10, 100]
        param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

        svc = SVC()
        clf = GridSearchCV(svc, param_grid, cv=10, verbose=1, scoring=my_accuracy_scorer)

        clf.fit(X_train, y_train)

        model = clf.best_estimator_

        return model

    print('Start training model using thundersvm...')
    model = _get_best_model(x_dataset_train, y_dataset_train)

if __name__== "__main__":
    main()
