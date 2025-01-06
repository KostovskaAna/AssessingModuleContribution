import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np 
import sys
import shap


def train_predict(conf, modAlgo):
    random_state = 42
    # /home/akostovska/GECCO_journal_extension
    performance_table_data = []

    for dim in [5, 30]:
        for budget in [50*dim, 100*dim, 300*dim, 500*dim, 1000*dim, 1500*dim]: 
            df_X = pd.read_csv(f"/home/akostovska/GECCO_journal_extension/data/landscape_data/ELA_{dim}D.csv", index_col=0)
            df_y = pd.read_csv(f"/home/akostovska/GECCO_journal_extension/data/performance_data/{modAlgo}/log/budget_{budget}_conf_{conf}_{dim}D.csv", index_col=0)
            df = df_X.join(df_y)
            X = df.drop(['target'], axis = 1)
            y = df.iloc[:,-1]
            # outer cv loop
            groups_outer = [i.split("_")[1] for i in df.index]
            logo_outer = LeaveOneGroupOut()
            r2_scores_train_outer = []
            r2_scores_test_outer = []
            test_preds = []
            test_preds_index = []
            for train_index_outer, test_index_outer in logo_outer.split(X, y, groups_outer):
                X_train_outer, X_test_outer = X.iloc[train_index_outer], X.iloc[test_index_outer]
                y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]
                # iterate over all possible hyperparameter configurations
                best_inner_model = None
                best_r2_score_test_inner = -10000
                for n_estimators in [10, 50, 100, 500]:
                    for max_features in [1.0, 'sqrt', 'log2']:
                        for max_depth in [4,8,15, None]:
                            for min_samples_split in [2, 5, 10]:
                                # inner cv loop
                                groups_inner = [i.split("_")[1] for i in X_train_outer.index]
                                logo_inner = LeaveOneGroupOut()
                                r2_scores_train_inner = []
                                r2_scores_test_inner = []
                                for train_index_inner, test_index_inner in logo_inner.split(X_train_outer, y_train_outer, groups_inner):
                                    X_train_inner, X_test_inner = X_train_outer.iloc[train_index_inner], X_train_outer.iloc[test_index_inner]
                                    y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]
                                    model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=-1, random_state=random_state)
                                    model.fit(X_train_inner, y_train_inner)
                                    # calculate performance metrics on the inner cv 
                                    y_train_inner_pred = model.predict(X_train_inner)
                                    y_test_inner_pred = model.predict(X_test_inner)
                                    r2_score_train_inner = r2_score(y_train_inner, y_train_inner_pred)
                                    r2_score_test_inner = r2_score(y_test_inner, y_test_inner_pred)
                                    r2_scores_train_inner.append(r2_score_train_inner)
                                    r2_scores_test_inner.append(r2_score_test_inner)
                                mean_of_r2_scores_test_inner = np.mean(r2_scores_test_inner)
                                # print(f"n_estimators={n_estimators}, max_features={max_features}, max_depth={max_depth}, min_samples_split={min_samples_split}, r2_train = { np.mean(r2_scores_train_inner)}, r2_test = { np.mean(r2_scores_test_inner)}")
                                if best_r2_score_test_inner < mean_of_r2_scores_test_inner:
                                    best_r2_score_test_inner = mean_of_r2_scores_test_inner
                                    best_inner_model = model
                best_inner_model.fit(X_train_outer, y_train_outer)
                y_train_outer_pred = best_inner_model.predict(X_train_outer)
                y_test_outer_pred = best_inner_model.predict(X_test_outer)
                test_preds.extend(y_test_outer_pred)
                test_preds_index.extend(y_test_outer.index)
                r2_score_train_outer = r2_score(y_train_outer, y_train_outer_pred)
                r2_score_test_outer = r2_score(y_test_outer, y_test_outer_pred)
                r2_scores_train_outer.append(r2_score_train_outer)
                r2_scores_test_outer.append(r2_score_test_outer)
                fold = list(X_test_outer.index)[0].split("_")[1]
                print(f'fold: {fold}, r2_train: {np.mean(r2_scores_train_outer)}, r2_test: {np.mean(r2_scores_test_outer)}')
                # get shapley values 
                explainer = shap.TreeExplainer(best_inner_model)
                shap_train = explainer.shap_values(X_train_outer)
                shap_test = explainer.shap_values(X_test_outer)
                df_shap_train = pd.DataFrame(shap_train, index = X_train_outer.index)
                df_shap_test = pd.DataFrame(shap_test, index = X_test_outer.index)
                # save shapley values on the outer test set
                shapley_folder = f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/shapley'
                if not os.path.exists(shapley_folder):
                    os.makedirs(shapley_folder)
                df_shap_train.to_csv(f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/shapley/shapley_budget_{budget}_conf_{conf}_{dim}D_fold_{fold}_train.csv')
                df_shap_test.to_csv(f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/shapley/shapley_budget_{budget}_conf_{conf}_{dim}D_fold_{fold}_test.csv')
                # save best model in each inner cv fold
                models_folder = f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/regression_models'
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                joblib.dump(best_inner_model, f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/regression_models/budget_{budget}_conf_{conf}_{dim}D_test_fold_{fold}.pkl', compress = 1)
            # save predictions on the outer test set
            predictions_folder = f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/regression_predictions'
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)
            test_preds_df = pd.DataFrame(test_preds, columns=["predicted"])
            test_preds_df.index = test_preds_index
            test_preds_df.to_csv(f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/regression_predictions/budget_{budget}_conf_{conf}_{dim}D_test_fold.csv')
            performance_table_data.append([conf, dim, budget, np.mean(r2_scores_train_outer), np.mean(r2_scores_test_outer)])
            print(f'r2_train: {np.mean(r2_scores_train_outer)}, r2_test: {np.mean(r2_scores_test_outer)}')
    performance_df = pd.DataFrame(performance_table_data, columns=['conf', 'dim', 'budget', 'train_r2', 'test_r2'])
    performance_df.to_csv(f'/home/akostovska/GECCO_journal_extension/results/{modAlgo}/regression_performance_tables/RForest_conf_{conf}.csv')

if __name__ == "__main__":
    conf = int(sys.argv[1])
    modAlgo = sys.argv[2]
    train_predict(conf, modAlgo)