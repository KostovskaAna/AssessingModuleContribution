import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np 
import sys
from sklearn.dummy import DummyRegressor


def train_predict():
    # /home/akostovska/GECCO_journal_extension
    performance_table_data = []

    for modAlgo in ['modCMA', 'modDE']:
        conf_num = 324 if modAlgo == 'modCMA' else 576
        for conf in range(0, conf_num):
            print(conf)
            for dim in [5, 30]:
                for budget in [50*dim, 100*dim, 300*dim, 500*dim, 1000*dim, 1500*dim]: 
                    df_X = pd.read_csv(f"./data/landscape_data/ELA_{dim}D.csv", index_col=0)
                    df_y = pd.read_csv(f"./data/performance_data/{modAlgo}/log/budget_{budget}_conf_{conf}_{dim}D.csv", index_col=0)
                    df = df_X.join(df_y)
                    X = df.drop(['target'], axis = 1)
                    y = df.iloc[:,-1]
                    groups_outer = [i.split("_")[1] for i in df.index]
                    logo_outer = LeaveOneGroupOut()
                    r2_scores_train_outer = []
                    r2_scores_test_outer = []
                    mse_scores_train_outer = []
                    mse_scores_test_outer = []
                    test_preds = []
                    test_preds_index = []
                    for train_index_outer, test_index_outer in logo_outer.split(X, y, groups_outer):
                        X_train_outer, X_test_outer = X.iloc[train_index_outer], X.iloc[test_index_outer]
                        y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]
                        
                        model = DummyRegressor(strategy='mean')  # Or choose 'median' or a constant value
                        model.fit(X_train_outer, y_train_outer)

                        y_train_outer_pred = model.predict(X_train_outer)
                        y_test_outer_pred = model.predict(X_test_outer)
                        test_preds.extend(y_test_outer_pred)
                        test_preds_index.extend(y_test_outer.index)


                        r2_score_train_outer = r2_score(y_train_outer, y_train_outer_pred)
                        r2_score_test_outer = r2_score(y_test_outer, y_test_outer_pred)
                        r2_scores_train_outer.append(r2_score_train_outer)
                        r2_scores_test_outer.append(r2_score_test_outer)

                        mse_score_train_outer = mean_squared_error(y_train_outer, y_train_outer_pred)
                        mse_score_test_outer = mean_squared_error(y_test_outer, y_test_outer_pred)
                        mse_scores_train_outer.append(mse_score_train_outer)
                        mse_scores_test_outer.append(mse_score_test_outer)
                        
                        
                        
                
                    performance_table_data.append([modAlgo, conf, dim, budget, np.mean(r2_scores_train_outer), np.mean(r2_scores_test_outer), np.mean(mse_scores_train_outer), np.mean(mse_scores_test_outer)])
    performance_df = pd.DataFrame(performance_table_data, columns=['modAlgo', 'conf', 'dim', 'budget', 'train_r2', 'test_r2', 'train_mse', 'test_mse'])
    performance_df.to_csv(f'dummy_regressor.csv')

if __name__ == "__main__":
    train_predict()