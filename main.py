from ML_Model import *
from TCN import TCNModel
from ALSTM import ALSTMModel
from Transformer import Transformer
from Portfolio import *

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp
import pickle
import joblib
import torch as th
from torch import optim
from torch.utils.data import Dataset, DataLoader
import optuna
from tqdm import trange
import quantstats as qs
from sklearn.metrics import mean_squared_error


# Variables
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
MAX_EPOCH = 10
RANDOM_STATE = 1


class Data(Dataset):
    """
    The simple Dataset object from torch that can produce batchs of data
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# read the pkl data
def read_pkl(dir: str):
    f = open(dir,'rb')
    data = pickle.load(f)
    f.close()
    return data

# write the pkl data
def write_pkl(dir: str, data):
    with open(dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

# fill the NA value by the average of its neighbors
def fill_na(df: pd.DataFrame):
    df = df.groupby(['instrument'], as_index=False).apply(lambda group: group.fillna(0)).reset_index(drop=False)
    return df


def train_ML_model(save_dir: str, modelLst: list,  start_year: int, end_year: int, train_window: int, test_window: int,
    hyperopt=False, hyperopt_dict=None):
    
    alpha = pd.read_csv('Alpha158.csv', index_col=[0,1], header=[0,1])
    alpha = alpha.dropna().reset_index(drop=False)
    alpha.rename(columns={'datetime': 'date', 'instrument': 'id'}, inplace=True)
    alpha['date'] = pd.to_datetime(alpha['date'])
    for model in modelLst:
        mseDF = []
        if not os.path.exists("{}/Model/{}".format(save_dir, model.name)):
            os.makedirs("{}/Model/{}".format(save_dir, model.name))

        if not os.path.exists("{}/Factor/{}".format(save_dir, model.name)):
            os.makedirs("{}/Factor/{}".format(save_dir, model.name))

        if not os.path.exists("{}/MSE/{}".format(save_dir, model.name)):
            os.makedirs("{}/MSE/{}".format(save_dir, model.name))

        for y in range(start_year, end_year-train_window-test_window+2):

            train_i = alpha.loc[(alpha['date'] >= pd.Timestamp("{}-01-01".format(y))) & (alpha['date'] <= pd.Timestamp("{}-12-31".format(y+train_window-1)))]
            test_i = alpha.loc[(alpha['date'] >= pd.Timestamp("{}-01-01".format(y+train_window))) & (alpha['date'] <= pd.Timestamp("{}-12-31".format(y+train_window+test_window-1)))]
            train_i.set_index(['id', 'date'], inplace=True)
            test_i.set_index(['id', 'date'], inplace=True)
            train_X, train_Y = train_i['feature'], train_i['label', 'LABEL1']
            test_X, test_Y = test_i['feature'], test_i['label', 'LABEL1']

            if os.path.exists("{}/Model/{}/{}_{}_{}.m".format(save_dir, model.name, y, y+train_window, y+train_window+test_window)):
                modelFitted = joblib.load("{}/Model/{}/{}_{}_{}.m".format(save_dir, model.name, y, y+train_window, y+train_window+test_window))
            else:
                if hyperopt:
                    model_dict = hyperopt_dict['{}'.format(model.name)]
                    uniform_dict, int_dict, choice_dict = model_dict['uniform_dict'], model_dict['int_dict'], model_dict['choice_dict']
                    model.hyperopt(train_X, train_Y, uniform_dict, int_dict, choice_dict)
                modelFitted = model.fit(train_X, train_Y)
                joblib.dump(modelFitted, "{}/Model/{}/{}_{}_{}.m".format(save_dir, model.name, y, y+train_window, y+train_window+test_window))

            pred_Y = modelFitted.predict(test_X)
            mse_i = mean_squared_error(pred_Y, test_Y)
            print('The MSE for model {} with data from {} to {} is {}'.format(model.name, y+train_window, y+train_window+test_window-1, mse_i))
            res = test_i['label', 'LABEL1'].to_frame()
            res['{}'.format(model.name)] = pred_Y
            res.reset_index(drop=False, inplace=True)
            res.to_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), index=False)
            test_i['pred'] = pred_Y

            def mse(x):
                return mean_squared_error(x['label', 'LABEL1'], x['pred'])

            test_i = test_i.groupby(by='id').apply(mse)
            test_i.name = '{}'.format(y)
            mseDF.append(test_i)
        
        mseDF = pd.concat(mseDF, axis=1)
        mseDF.to_csv("{}/MSE/{}/mse.csv".format(save_dir, model.name))


def train_DL_model(save_dir: str, modelLst: list,  start_year: int, end_year: int, train_window: int, test_window: int,
    hyperopt=False, learning_rate=2e-4, weight_decay=0):
    
    alpha = pd.read_csv('Alpha158.csv', index_col=[0,1], header=[0,1])
    alpha = alpha.dropna().reset_index(drop=False)
    alpha.rename(columns={'datetime': 'date', 'instrument': 'id'}, inplace=True)
    alpha['date'] = pd.to_datetime(alpha['date'])
    for model in modelLst:
        mseDF = []
        if not os.path.exists("{}/Model/{}".format(save_dir, model.name)):
            os.makedirs("{}/Model/{}".format(save_dir, model.name))

        if not os.path.exists("{}/Factor/{}".format(save_dir, model.name)):
            os.makedirs("{}/Factor/{}".format(save_dir, model.name))

        if not os.path.exists("{}/MSE/{}".format(save_dir, model.name)):
            os.makedirs("{}/MSE/{}".format(save_dir, model.name))

        for y in range(start_year, end_year-train_window-test_window+2):

            train_i = alpha.loc[(alpha['date'] >= pd.Timestamp("{}-01-01".format(y))) & (alpha['date'] <= pd.Timestamp("{}-12-31".format(y+train_window-1)))]
            test_i = alpha.loc[(alpha['date'] >= pd.Timestamp("{}-01-01".format(y+train_window))) & (alpha['date'] <= pd.Timestamp("{}-12-31".format(y+train_window+test_window-1)))]
            train_i.set_index(['id', 'date'], inplace=True)
            test_i.set_index(['id', 'date'], inplace=True)
            train_X, train_Y = train_i['feature'].values, train_i['label', 'LABEL1'].values
            test_X, test_Y = test_i['feature'].values, test_i['label', 'LABEL1'].values

            train_X, train_Y = th.from_numpy(train_X).unsqueeze(1), th.from_numpy(train_Y)
            test_X, test_Y = th.from_numpy(test_X).unsqueeze(1), th.from_numpy(test_Y)
            train_X, train_Y = train_X.to(th.float32), train_Y.to(th.float32)
            test_X, test_Y = test_X.to(th.float32), test_Y.to(th.float32)

            train_Dataset = Data(train_X, train_Y)
            train_Dataloader = DataLoader(train_Dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
            test_Dataset = Data(test_X, test_Y)
            test_Dataloader = DataLoader(test_Dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

            if os.path.exists("{}/Model/{}/{}_{}_{}.m".format(save_dir, model.name, y, y+train_window, y+train_window+test_window)):
                modelFitted = joblib.load("{}/Model/{}/{}_{}_{}.m".format(save_dir, model.name, y, y+train_window, y+train_window+test_window))
            else:
                if hyperopt:
                    def objective(trial):
                        #  decide the hyperparameters we want to tune
                        params = {
                                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
                                }

                        return train(model, train_Dataloader, lr=params['learning_rate'], weight_decay=params['weight_decay'])[-1]

                    # initialize the study
                    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
                    study.optimize(objective, n_trials=10)
                    # select the best trail and get the best hyperparameter
                    best_trial = study.best_trial
                    learning_rate = best_trial.params['learning_rate']
                    weight_decay = best_trial.params['weight_decay']
                modelFitted, _ = train(model, train_Dataloader, lr=learning_rate, weight_decay=weight_decay)
                joblib.dump(modelFitted, "{}/Model/{}/{}_{}_{}.m".format(save_dir, model.name, y, y+train_window, y+train_window+test_window))

            pred_Y = evaluate(modelFitted, test_Dataloader)
            mse_i = th.nn.MSELoss()(pred_Y, test_Y)
            print('The MSE for model {} with data from {} to {} is {}'.format(model.name, y+train_window, y+train_window+test_window-1, mse_i))
            res = test_i['label', 'LABEL1'].to_frame()
            res['{}'.format(model.name)] = pred_Y.detach().numpy()
            res.reset_index(drop=False, inplace=True)
            res.to_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), index=False)
            test_i['pred'] = pred_Y.detach().numpy()

            def mse(x):
                return mean_squared_error(x['label', 'LABEL1'], x['pred'])

            test_i = test_i.groupby(by='id').apply(mse)
            test_i.name = '{}'.format(y)
            mseDF.append(test_i)
        
        mseDF = pd.concat(mseDF, axis=1)
        mseDF.to_csv("{}/MSE/{}/mse.csv".format(save_dir, model.name))


def evaluate(model, dataloader):
    # choose criterion for the loss function
    criterion = th.nn.MSELoss()

    # enumerate mini batches
    print('Evaluating ...')
    test_data_size = len(dataloader)
    test_dataiter = iter(dataloader)
    model.eval()

    # initailize the loss
    Loss = 0
    pred = []

    # set the bar to check the progress
    with trange(test_data_size) as test_bar:
        for i in test_bar:
            test_bar.set_description('Evaluating batch %s'%(i+1))
            x_test, y_test = next(test_dataiter)

            # compute the model output without calculating the gradient
            y_pred = model(x_test)
            pred.append(y_pred)

            # calculate loss
            Loss += criterion(y_pred, y_test).item()

            # set information for the bar
            test_bar.set_postfix(evaluate_loss=Loss/(i+1))

        return th.cat(pred)
    

def train(model, train_dataloader, valid_dataloader=None, MAX_EPOCH=MAX_EPOCH, lr=2e-4, weight_decay=0):
    # set the criterion
    criterion = th.nn.MSELoss()

    # set the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print('{} epochs to train: '.format(MAX_EPOCH))
    best_loss = math.inf

    for epoch in range(1, MAX_EPOCH+1):

        # enumerate mini batches
        print('epoch {}/{}:'.format(epoch, MAX_EPOCH))
        train_data_size = len(train_dataloader)
        train_dataiter = iter(train_dataloader)
        model.train()

        Total_loss = 0
        min_loss = math.inf
        best_model = None

        # set the bar to check the progress
        with trange(train_data_size) as train_bar:
            for i in train_bar:
                train_bar.set_description('Training batch %s'%(i+1))
                x_train, y_train = next(train_dataiter)

                # clear the gradients
                optimizer.zero_grad()

                # compute the model output
                y_pred = model(x_train)

                # calculate loss
                train_loss = criterion(y_pred, y_train)
                Total_loss += train_loss.item()

                # credit assignment
                train_loss.backward()

                # update model weights
                optimizer.step()

                # set information for the bar
                train_bar.set_postfix(train_loss=Total_loss/(i+1))

            # see whether the trained model after this epoch is the currently best
            if valid_dataloader is not None:
                model.eval()
                loss = evaluate(model, valid_dataloader)
                if loss < min_loss:
                    best_model = model
                    min_loss = loss
                model.train()
            else:
                best_model = model
            
            if Total_loss < best_loss:
                best_loss = Total_loss

    return best_model, best_loss


def backtest(save_dir: str, modelLst: list,  start_year: int, end_year: int, train_window: int, test_window: int):

    for model in modelLst:
        if not os.path.exists("{}/Report/{}".format(save_dir, model.name)):
            os.makedirs("{}/Report/{}".format(save_dir, model.name))

        df = pd.DataFrame()
        for y in range(start_year, end_year-train_window-test_window+2):
            df_i = pd.read_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), header=[0])
            df_i = df_i.loc[1:,:]
            df = df.append(df_i)
        df.reset_index(drop=True, inplace=True)
        dateLst = df['date'].values
        # when yyyy-mm-dd
        try:
            df['date'] = [pd.Timestamp(int(date_i[:4]), int(date_i[5:7]), int(date_i[8:])) for date_i in dateLst]
        # when yyyy/mm/dd
        except:
            dateLst = [i.split('/') for i in dateLst]
            df['date'] = [pd.Timestamp(int(date_i[0]), int(date_i[1]), int(date_i[2])) for date_i in dateLst]
        df[['label', '{}'.format(model.name)]] = df[['label', '{}'.format(model.name)]].astype(np.float64)

        def get_ret(x):
            true_y = x['label'].mean()
            pred_y = x['{}'.format(model.name)]
            pred_y = (pred_y - pred_y.min()) / (pred_y.max() - pred_y.min())
            pred_y = (pred_y * x['label'].values / pred_y.sum()).sum()
            return true_y, pred_y

        def get_top_half(x):
            true_ret = x['label'].values
            true_y = true_ret.mean()
            pred_y = x['{}'.format(model.name)].values
            idx = pred_y.argsort()
            pred_y = pred_y[idx][-5:]
            true_ret = true_ret[idx][-5:]
            pred_y = (pred_y - pred_y.min()) / (pred_y.max() - pred_y.min())
            pred_y = (pred_y * true_ret / pred_y.sum()).sum()
            return true_y, pred_y

        # df = df.groupby(by='date').apply(get_ret)
        df = df.groupby(by='date').apply(get_top_half)
        df.name = 'return'
        df = df.to_frame()
        df['benchmark'] = df['return'].apply(lambda x: x[0])
        df['strategy'] = df['return'].apply(lambda x: x[1])
        del df['return']
        print(df)
        # create the report under the path
        report_dir = "{}/Report/{}/{}_{}.html".format(save_dir, model.name, start_year, end_year)
        qs.reports.html(df['strategy'], df['benchmark'],
            title='Report of ML strategy with model {}'.format(model.name),
            output=report_dir)
        print('Report saved in %s' % (report_dir))


def get_signal(save_dir: str, modelLst: list,  start_year: int, end_year: int, train_window: int, test_window: int):

    for model in modelLst:
        print("Getting the signal predicted by model {} ...".format(model.name))

        if not os.path.exists("{}/Signal/{}".format(save_dir, model.name)):
            os.makedirs("{}/Signal/{}".format(save_dir, model.name))

        if not os.path.exists("{}/Label/{}".format(save_dir, model.name)):
            os.makedirs("{}/Label/{}".format(save_dir, model.name))

        df = pd.DataFrame()
        for y in range(start_year, end_year-train_window-test_window+2):
            df_i = pd.read_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), header=[0])
            df_i = df_i.loc[1:,:]
            df = df.append(df_i)
        df.reset_index(drop=True, inplace=True)
        dateLst = df['date'].values
        # when yyyy-mm-dd
        try:
            df['date'] = [pd.Timestamp(int(date_i[:4]), int(date_i[5:7]), int(date_i[8:])) for date_i in dateLst]
        # when yyyy/mm/dd
        except:
            dateLst = [i.split('/') for i in dateLst]
            df['date'] = [pd.Timestamp(int(date_i[0]), int(date_i[1]), int(date_i[2])) for date_i in dateLst]
        df[['label', '{}'.format(model.name)]] = df[['label', '{}'.format(model.name)]].astype(np.float64)

        def get_weights(x):
            pred_y = x.values
            idx = pred_y.argsort()
            idx = idx[-5:]
            ret = np.zeros(len(pred_y))
            pred_y = pred_y[idx]
            pred_y = (pred_y - pred_y.min()) / (pred_y.max() - pred_y.min())
            np.put(ret, idx, pred_y)
            return pd.Series(ret, index=x.index)

        # df_min = df.groupby(by='date').agg({'{}'.format(model.name):'min'}).rename(columns={'{}'.format(model.name): 'min'}).reset_index()
        # df_max = df.groupby(by='date').agg({'{}'.format(model.name):'max'}).rename(columns={'{}'.format(model.name): 'max'}).reset_index()
        # df = pd.merge(df, df_min, on='date', how='left')
        # df = pd.merge(df, df_max, on='date', how='left')
        # df['signal'] = (df['{}'.format(model.name)] - df['min']) / (df['max'] - df['min'])
        # df['pct'] = df.groupby(by='date')['signal'].rank(pct=True)
        # df['pct'].loc[df['pct'] >= 0.5] = 1
        # df['pct'].loc[df['pct'] < 0.5] = 0
        # df['signal'] = df['signal'] * df['pct']

        df['signal'] = df.groupby(by='date')['{}'.format(model.name)].apply(get_weights)

        label = df[['id', 'date', 'label']]
        df = df[['id', 'date', 'signal']]
        label.set_index(['id', 'date'], inplace=True)
        df.set_index(['id', 'date'], inplace=True)
        label = label.unstack(level=0)
        df = df.unstack(level=0)
        label.columns = [i[-1] for i in label.columns]
        df.columns = [i[-1] for i in df.columns]
        df.fillna(0, inplace=True)
        label.to_csv("{}/Label/{}/Label.csv".format(save_dir, model.name))
        df.to_csv("{}/Signal/{}/Signal.csv".format(save_dir, model.name))


def backtest_signal(save_dir: str, modelLst: list, gamma, update_period, alpha, period_len):

    if not os.path.exists("{}/Report/Portfolio".format(save_dir)):
        os.makedirs("{}/Report/Portfolio".format(save_dir))

    label = pd.read_csv("{}/Label/Label.csv".format(save_dir), index_col=[0], header=[0])
    label_df = label.stack().to_frame()
    label_df.index.names = ['date', 'id']
    label_df.columns = ['label']

    for model in modelLst:
        print('Get signals predicted by model {}'.format(model.name))

        signal = pd.read_csv("{}/Signal/{}/Signal.csv".format(save_dir, model.name), index_col=[0], header=[0])

        signal_df = signal.stack().to_frame()
        signal_df.index.names = ['date', 'id']
        signal_df.columns = ['signal']

        data = pd.merge(label_df, signal_df, on=['id', 'date'], how='left')

        def get_value(x):
            return (x['label'] * x['signal']).sum() / x['signal'].sum()

        data = data.groupby(by='date').apply(get_value)
        data.name = '{}'.format(model.name)
        label = pd.merge(label, data, on='date', how='left')

    nameLst = [model.name for model in modelLst]
    data = label[nameLst]
    print(data)
    portfolio = mean_variance_sliding_result_large(data, gamma, update_period, alpha, period_len)
    # portfolio = mean_variance_sliding_result_large(label, 10, 5, 0.1, 20)
    portfolio = portfolio.stack().to_frame()
    portfolio.index.names = ['date', 'id']
    portfolio.columns = ['weights']

    label = label.stack().to_frame()
    label.index.names = ['date', 'id']
    label.columns = ['label']

    df = pd.merge(portfolio, label, on=['id', 'date'], how='left')

    df.reset_index(drop=False, inplace=True)
    dateLst = df['date'].values
    df['date'] = [pd.Timestamp(int(date_i[:4]), int(date_i[5:7]), int(date_i[8:])) for date_i in dateLst]

    label.reset_index(drop=False, inplace=True)
    dateLst = label['date'].values
    label['date'] = [pd.Timestamp(int(date_i[:4]), int(date_i[5:7]), int(date_i[8:])) for date_i in dateLst]

    def get_true_ret(x):
        return x['label'].mean()

    def get_pred_ret(x):
        pred_y = x['weights']
        pred_y = (pred_y * x['label'].values).sum()
        return pred_y

    label = label.groupby(by='date').apply(get_true_ret)
    label.name = 'benchmark'
    df = df.groupby(by='date').apply(get_pred_ret)
    df.name = 'strategy'
    label = label.to_frame()
    df = df.to_frame()
    df = pd.merge(df, label, on='date', how='left')
    print(df)
    # create the report under the path
    report_dir = "{}/Report/Portfolio/result_{}_{}_{}_{}.html".format(save_dir, gamma, update_period, alpha, period_len)
    qs.reports.html(df['strategy'], df['benchmark'],
        title='Report of ML strategy with model {}'.format('+'.join(nameLst)),
        output=report_dir)
    print('Report saved in %s' % (report_dir))


def get_metrics(save_dir: str, modelLst: list,  start_year: int, end_year: int, train_window: int, test_window: int):

    ICdfLst = []
    RankICdfLst = []
    fctLst = []

    for model in modelLst:
        df = pd.DataFrame()
        for y in range(start_year, end_year-train_window-test_window+2):
            df_i = pd.read_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), header=[0])
            df_i = df_i.loc[1:,:]
            df = df.append(df_i)
        df['label'] = df['label'].astype(np.float64)
        df.reset_index(drop=True, inplace=True)

        # calculate the IC (pearson correlation)
        def IC(x):
            return sp.pearsonr(x[model.name], x['label'])[0] * 100
        # calculate the Rank IC (spearman correlation)
        def RankIC(x):
            return sp.spearmanr(x[model.name], x['label'])[0] * 100

        # match current factors with return on the next period
        ICdf = df.groupby('date').apply(IC)
        RankICdf = df.groupby('date').apply(RankIC)

        ICdfLst.append(ICdf)
        RankICdfLst.append(RankICdf)
        fctLst.append(model.name)

    show_metrics(ICdf, RankICdf, fctLst, save_dir, 'The Metrics for Different ML Strategies')


def show_metrics(ICdfLst: list, RankICdfLst: list, fctLst: list, save_dir: str,  title=''):
    # initialize the columns
    cellText = []
    colLabels = ['IC mean', 'Rank IC mean']
    # get metrics
    for i in range(len(fctLst)):
        print('Process factor %s' % fctLst[i])
        metrics = []
        ICdf, RankICdf = ICdfLst[i], RankICdfLst[i]
        metrics.append('%.4f' % ICdf.mean())
        metrics.append('%.4f' % RankICdf.mean())
        cellText.append(metrics)
    # plot the table
    plt.table(cellText=cellText, colLabels=colLabels, rowLabels=fctLst, loc="center")
    plt.axis('tight')
    plt.axis('off')
    plt.title(title)
    plt.savefig('%s/result.png' % save_dir)
    plt.show()


def show_corr(save_dir: str, modelLst: list,  start_year: int, end_year: int, train_window: int, test_window: int):
    dfLst = []
    for model in modelLst:
        print('Loading the factor predicted by {}'.format(model.name))
        df = pd.DataFrame()
        for y in range(start_year, end_year-train_window-test_window+2):
            df_i = pd.read_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), header=[0])
            df_i = df_i.loc[1:,:]
            df = df.append(df_i)
        df.reset_index(drop=True, inplace=True)
        temp = df[['id', 'date']]
        dfLst.append(df['{}'.format(model.name)])
    dfLst.append(temp)
    data = pd.concat(dfLst, axis=1)
    nameLst = [model.name for model in modelLst]
    data = data.groupby(by='date')[nameLst].corr().unstack()
    data = data.mean(axis=0).to_frame().unstack()
    data.columns = [i[-1] for i in data.columns]
    print(data)

    plt.figure(figsize=(11,8))
    sns.heatmap(data, cmap="Greens",annot=True)
    plt.savefig("corr.png")
    plt.show()


def show_mse(save_dir: str, modelLst: list):
    dfLst = []
    for model in modelLst:
        print('Loading the mse data predicted by {}'.format(model.name))
        df_i = pd.read_csv("{}/MSE/{}/mse.csv".format(save_dir, model.name), index_col=[0])
        df_i = df_i.values.reshape(-1)
        df_i = pd.DataFrame({'mse': df_i, 'model': model.name})
        dfLst.append(df_i)
    df = pd.concat(dfLst, axis=0)
    # use to set style of background of plot
    sns.set(style="whitegrid")
    
    # loading data-set
    sns.boxplot(x = 'model', y = 'mse',
                    data = df,
                    linewidth=2.5)
    plt.title("The MSE of different model")
    plt.savefig("{}/mse.png".format(save_dir))
    plt.show()


def show_prediction(save_dir: str, modelLst: list, start_year: int, end_year: int, train_window: int, test_window: int, stock='000905.SZ'):

    for model in modelLst:
        print('Loading the factor predicted by model {}'.format(model.name))
        df = pd.DataFrame()
        for y in range(start_year, end_year-train_window-test_window+2):
            df_i = pd.read_csv("{}/Factor/{}/{}_{}.csv".format(save_dir, model.name, y+train_window, y+train_window+test_window), header=[0])
            df_i = df_i.loc[1:,:]
            df = df.append(df_i)
        df['label'] = df['label'].astype(np.float64)
        df = df.loc[df['id'] == stock]
        df.sort_values('date', inplace=True)
        g = sns.lineplot(x='date', y='{}'.format(model.name), data=df)
        g.set(xticklabels=[])
    sns.lineplot(x='date', y='label', data=df)
    plt.show()


if __name__ == "__main__":
    
    start_year = 2016
    end_year = 2021
    save_dir = '.'
    train_window = 2
    test_window = 1
    in_feature = 158
    out_feature = 1
    gamma = [0.001,0.005,0.01,0.05,0.1,0.5,1,3,5,7,8,9,10,12,15]
    # gamma = [1]
    alpha = [0.2]
    # alpha = np.linspace(0.1,0.9,9)
    update_period = [5,10,15,20]
    update_period = [5]
    # len_period = [20,30,50,60,120]
    len_period = [120]

    ridge = Ridge_Regression()
    lasso = Lasso_regression()
    lasso.set_params(alpha=1e-5)
    en = Elastic_Net()
    en.set_params(alpha=1e-5)
    rf = RF()
    adaboost = Adaboost()
    xgb = XGboost()
    # xgb.set_params(n_jobs=8)
    catboost = Catboost()
    lgbm = LightGBM()
    baseline = Baseline()

    hyperopt_dict = {
        "Ridge": {'uniform_dict': {'alpha': (1e-10, 10)}, 'int_dict': {}, 'choice_dict': {}},
        "Lasso": {'uniform_dict': {'alpha': (1e-10, 10)}, 'int_dict': {}, 'choice_dict': {}},
        "Elastic Net": {'uniform_dict': {'alpha': (1-10, 10), 'l1_ratio': (0, 1)}, 'int_dict': {}, 'choice_dict': {}},
        "RF": {'uniform_dict': {}, 'int_dict': {'max_depth': (10, 100), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 10), 'n_estimators': (2, 200)}, 'choice_dict': {'bootstrap': [True], }},
        "Adaboost": {'uniform_dict': {'learning_rate': (0.01, 1)}, 'int_dict': {'n_estimators': (2, 200)}, 'choice_dict': {}},
        "XGboost": {'uniform_dict': {"lambda": (0.1,1.0), 'learning_rate': (0.1,1.0)}, 'int_dict': {}, 'choice_dict': {"booster": ["gbtree", "gblinear", "dart"], }},
        "Catboost": {'uniform_dict': {'learning_rate': (1e-4, 1)}, 'int_dict': {}, 'choice_dict': {'depth': [3, 5, 6, 7], 'l2_leaf_reg': [3, 5, 7], }},
        "LightGBM": {'uniform_dict': {'learning_rate': (1e-4, 1)}, 'int_dict': {}, 'choice_dict': {'max_depth': [3, 5, 6, 7], 'lambda_l2': [0.1, 0.3, 0.5], }},
    }

    MLmodelLst = [ridge, lasso, en, rf, adaboost, xgb, catboost, lgbm]
    MLmodelLst = [baseline, ridge, lasso, en]
    MLmodelLst = [ridge, adaboost]

    alstm = ALSTMModel(in_feature)
    tcn = TCNModel(in_feature, out_feature, [32,64,64], 2, 0.5)
    transformer = Transformer(in_feature)

    DLmodelLst = [alstm, transformer]
    # DLmodelLst = [alstm, transformer]
    # DLmodelLst = []

    # train_ML_model(save_dir, MLmodelLst, start_year, end_year, train_window, test_window, True, hyperopt_dict)
    # train_DL_model(save_dir, DLmodelLst, start_year, end_year, train_window, test_window)
    # backtest(save_dir, MLmodelLst+DLmodelLst, start_year, end_year, train_window, test_window)
    # get_signal(save_dir, MLmodelLst+DLmodelLst, start_year, end_year, train_window, test_window)
    for g in gamma:
        for i in update_period:
            for j in alpha:
                for k in len_period:
                    backtest_signal(save_dir, MLmodelLst+DLmodelLst, g, i, j, k)
    # get_metrics(save_dir, MLmodelLst+DLmodelLst, start_year, end_year, train_window, test_window)
    # show_corr(save_dir, MLmodelLst+DLmodelLst, start_year, end_year, train_window, test_window)
    # show_mse(save_dir, MLmodelLst+DLmodelLst)
    # show_prediction(save_dir, MLmodelLst+DLmodelLst, start_year, end_year, train_window, test_window, '600015.SH')