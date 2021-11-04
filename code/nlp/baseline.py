from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import torch
import numpy as np
import pickle
import sys

def get_valid_setting(start_seed, n_run, X_dict, y_dict, est_list, lr_list):
    X_train_ori = X_dict['train']
    X_valid = X_dict['valid']

    
    y_train_ori = y_dict['train']
    y_valid = y_dict['valid']

    
    final_auc = 0
    final_est = -1
    final_lr = -1
    final_model = None
    
    seed = start_seed
    for i in range(n_run):
        ros = RandomOverSampler(random_state=seed)

        X_train, y_train = ros.fit_resample(X=X_train_ori, y=y_train_ori)
        X_train = pd.DataFrame(data=X_train)
        X_valid = pd.DataFrame(data=X_valid)

        max_auc = 0
        max_est = None
        max_lr = None
        max_model = None

        for est in est_list:
            for lr in lr_list:
                xgb = XGBClassifier(n_estimators=est, learning_rate = lr, max_depth = 4, use_label_encoder=False)
                xgb.fit(X_train, y_train, eval_metric='logloss')
                xgb_pred = xgb.predict(X_valid)
                auc = roc_auc_score(y_valid, xgb_pred)

                if auc > max_auc:
                    max_auc = auc
                    max_est = est
                    max_lr = lr
                    max_model = xgb

        print("Valid MAX AUC:", max_auc, "Valid MAX EST:", max_est, "Valid MAX LR:", max_lr)
        sys.stdout.flush()
        if max_auc > final_auc:
            final_auc = max_auc
            final_est = max_est
            final_lr = max_lr
            final_model = max_model
            
        seed += 1
        
    print("\nValid Final AUC:", final_auc, "Valid Final EST:", final_est, "Valid MAX LR:", final_lr)
    sys.stdout.flush()
    return final_model, final_auc, final_est, final_lr

def test_xgboost(model, X_dict, y_dict):
    X_test = X_dict['test']
    y_test = y_dict['test']
    
    X_test = pd.DataFrame(data=X_test)
    xgb_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, xgb_pred)
    
    class_names = ['Negative', 'Positive']
    classification_dict = classification_report(y_test, xgb_pred, target_names=class_names, output_dict=True)

    print("---------------------------------")
    print("Test AUC:", auc)
    print(classification_report(y_test, xgb_pred, target_names=class_names))
    
    return auc, classification_dict

def get_one_hot_leaves(model, x):
    x_leaves = model.apply(pd.DataFrame(x))

    nb_classes = 32
    one_hot_x_leaves = []
    for l in x_leaves:
        one_hot_x_leaves.append(torch.Tensor(np.eye(nb_classes)[l.astype(int)]))

    return one_hot_x_leaves

def load_leaves(save_dir=None, name=str):
    if save_dir==None:
        save_dir = 'split_data'

    train_leaves = pd.read_pickle(f'./{save_dir}/X_train_{name}_xgboost_one_hot.pkl')
    valid_leaves = pd.read_pickle(f'./{save_dir}/X_valid_{name}_xgboost_one_hot.pkl')
    test_leaves = pd.read_pickle(f'./{save_dir}/X_test_{name}_xgboost_one_hot.pkl')

    return train_leaves, valid_leaves, test_leaves

def base_classify(X, y, save_dir=None, n_run=5, start_seed=123, store=True, name=str):
    def merge_in_out(x_out, x_in):
        x = []
        for i in range(len(x_out)):
            x_out_ = x_out[i].tolist()
            x_in_ = x_in[i].tolist()
            x_ = x_out_ + x_in_
            x.append(x_)
        return x

    if save_dir==None:
        save_dir = 'split_data'
    model_dir = 'saved_models'

    X_out_train, X_in_train, X_out_valid, X_in_valid, X_out_test, X_in_test = X
    X_train = merge_in_out(X_out_train, X_in_train)
    X_valid = merge_in_out(X_out_valid, X_in_valid)
    X_test = merge_in_out(X_out_test, X_in_test)

    y_train, y_valid, y_test = y

    X_dict = {'train': X_train, 'valid': X_valid, 'test': X_test}
    y_dict = {'train': y_train, 'valid': y_valid, 'test': y_test}
    
    est_list = [50, 100, 150]
    lr_list = [0.1, 0.05, 0.01]
    #n_run=1
    #est_list = [50]
    #lr_list = [0.1]

    model, _, _, _ = get_valid_setting(start_seed, n_run, X_dict, y_dict, est_list, lr_list)
    test_xgboost(model, X_dict, y_dict)

    one_hot_X_train_leaves = get_one_hot_leaves(model, X_train)
    one_hot_X_valid_leaves = get_one_hot_leaves(model, X_valid)
    one_hot_X_test_leaves = get_one_hot_leaves(model, X_test)

    if store:
        print('Save the one-hot encoded xgboost leaves of train, valid, test dataset.')
        with open(f'./{model_dir}/{name}_xgboost_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
        with open(f'./{save_dir}/X_train_{name}_xgboost_one_hot.pkl', 'wb') as f:
            pickle.dump(one_hot_X_train_leaves, f)
        with open(f'./{save_dir}/X_valid_{name}_xgboost_one_hot.pkl', 'wb') as f:
            pickle.dump(one_hot_X_valid_leaves, f)
        with open(f'./{save_dir}/X_test_{name}_xgboost_one_hot.pkl', 'wb') as f:
            pickle.dump(one_hot_X_test_leaves, f)

    return one_hot_X_train_leaves, one_hot_X_valid_leaves, one_hot_X_test_leaves
