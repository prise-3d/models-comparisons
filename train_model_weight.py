# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import time

# models imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import sklearn.svm as svm
from sklearn.utils import shuffle
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import models as mdl

filetime_name = 'output_time.info'

# variables and parameters
saved_models_folder = cfg.saved_models_folder
models_list         = cfg.models_names_list

current_dirpath     = os.getcwd()
output_model_folder = os.path.join(current_dirpath, saved_models_folder)


def main():

    parser = argparse.ArgumentParser(description="Train SKLearn model and save it into .joblib file")

    parser.add_argument('--data', type=str, help='dataset file prefix (without .train and .test)')
    parser.add_argument('--output', type=str, help='output file name desired for model (without .joblib extension)')
    parser.add_argument('--choice', type=str, help='model choice from list of choices', choices=models_list)

    args = parser.parse_args()

    p_data_file = args.data
    p_output    = args.output
    p_choice    = args.choice

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

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
    # final_df_train = shuffle(final_df_train)
    # final_df_test = shuffle(final_df_test)

    final_df_train_size = len(final_df_train.index)
    final_df_test_size = len(final_df_test.index)

    # use of the whole data set for training
    # use only the fourth first elements and add weights
    #weights = [0.5, 0.3, 0.01, 0.01, 0.01, 0.01, 0.04, 0.1, 0.01, 0.01]
    # weights = [0.05088082, 0.03464818, 0.04999086, 0.05748639, 0.05145307, 0.04967657, \
    #             0.04891437, 0.03952947, 0.06644618, 0.04051915, 0.06741276, 0.03943296, \
    #             0.0423808,  0.03869016, 0.03591427, 0.,         0.03660649, 0.03911749, \
    #             0.00271328, 0.03857147, 0.03335941, 0.02617491, 0.01897268, 0.03659842, \
    #             0.01924748, 0.03526237]

    weights = np.ones(26)
    # print(weights)
    
    x_dataset_train = final_df_train.iloc[:,1:len(weights) + 1]
    
    for i, w in enumerate(weights):
        x_dataset_train.iloc[:,i] = x_dataset_train.iloc[:,i].apply(lambda x: w * x)
    
    x_dataset_test = final_df_test.iloc[:,1:len(weights) + 1]

    for i, w in enumerate(weights):
        x_dataset_test.iloc[:,i] = x_dataset_test.iloc[:,i].apply(lambda x: w * x)

    y_dataset_train = final_df_train.iloc[:,0]
    y_dataset_test = final_df_test.iloc[:,0]

    #######################
    # 2. Construction of the model : Ensemble model structure
    #######################

    print("-------------------------------------------")
    print("Train dataset size: ", final_df_train_size)

    start_time = time.time()
    model = mdl.get_trained_model(p_choice, x_dataset_train, y_dataset_train)

    trained_time = (time.time() - start_time)
    with open(filetime_name, 'a') as f:
        f.write(f'{p_output};{trained_time}\n')

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################
    val_scores = cross_val_score(model, x_dataset_train, y_dataset_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (val_scores.mean(), val_scores.std() * 2))

    ######################
    # 4. Test : Validation and test dataset from .test dataset
    ######################

    # we need to specify validation size to 20% of whole dataset
    val_set_size = int(final_df_train_size/3)
    test_set_size = val_set_size

    total_validation_size = val_set_size + test_set_size

    if final_df_test_size > total_validation_size:
        x_dataset_test = x_dataset_test[0:total_validation_size]
        y_dataset_test = y_dataset_test[0:total_validation_size]

    X_test, X_val, y_test, y_val = train_test_split(x_dataset_test, y_dataset_test, test_size=0.3, shuffle=True)

    y_train_model = model.predict(x_dataset_train)
    y_test_model = model.predict(X_test)
    y_val_model = model.predict(X_val)

    train_accuracy = accuracy_score(y_dataset_train, y_train_model)
    val_accuracy = accuracy_score(y_val, y_val_model)
    test_accuracy = accuracy_score(y_test, y_test_model)

    train_auc = roc_auc_score(y_dataset_train, y_train_model)
    val_auc = roc_auc_score(y_val, y_val_model)
    test_auc = roc_auc_score(y_test, y_test_model)

    # print('Train dataset 1 ', np.any(y_test_model == 1))
    # print('Train dataset 0 ', np.any(y_test_model == 0))

    # print('Val dataset 1 ', np.any(y_val_model == 1))
    # print('Val dataset 0 ', np.any(y_val_model == 0))

    # val_f1 = f1_score(y_val, y_val_model)
    # test_f1 = f1_score(y_test, y_test_model)

    ###################
    # 5. Output : Print and write all information in csv
    ###################

    print("Train dataset size ", len(y_train_model))
    print("Train: ", train_accuracy)

    print("Validation dataset size ", val_set_size)
    print("Validation: ", val_accuracy)

    print("Test dataset size ", test_set_size)
    print("Test: ", test_accuracy)


    ##################
    # 6. Save model : create path if not exists
    ##################

    if not os.path.exists(cfg.output_models):
        os.makedirs(cfg.output_models)

    joblib.dump(model, os.path.join(cfg.output_models, p_output + '.joblib'))

    ##################
    # 6. Save model perf into csv
    ##################

    if not os.path.exists(cfg.output_results_folder):
        os.makedirs(cfg.output_results_folder)

    results_filepath = os.path.join(cfg.output_results_folder, 'results.csv')

    # write header if necessary
    if not os.path.exists(results_filepath):
        with open(results_filepath, 'w') as f:
            f.write('name;train_acc;val_acc;test_acc;train_auc;val_auc;test_auc;\n')
            
    # add information into file
    with open(results_filepath, 'a') as f:
        line = p_output + ';' + str(train_accuracy) + ';' + str(val_accuracy) \
                        + ';' + str(test_accuracy) + ';' + str(train_auc) \
                        + ';' + str(val_auc) + ';' + str(test_auc) + '\n'
        f.write(line)

if __name__== "__main__":
    main()
