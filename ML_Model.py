from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from deepforest import CascadeForestRegressor
from hyperopt import fmin, tpe, hp
from hyperopt.pyll.base import scope

import warnings
# used to ignore the warnings
warnings.filterwarnings("ignore")


class Model(object):
    """
    Model used for classification task
    Currently updated models:
    1. Ridge Regression
    2. Lasso Regression
    3. Elastic Net
    4. Random Forest
    5. Adaboost
    6. XGboost
    7. Catboost
    8. Light Gradient Boosting Machine
    """
    def __init__(self, model):
        self.model = model
        self.name = None

    # set parameters for the model
    def set_params(self, **params):
        self.model.set_params(**params)

    # fit the model on the data
    def fit(self, train_X, train_Y):
        return self.model.fit(train_X, train_Y)

    # do cross validation to see the performance
    def cross_validation(self, train_X, train_Y, cv=5, scoring="neg_mean_squared_error", verbose=False):
        """
        Cross validation

        Args:
            cv: the number of splits for the cross validation
            scoring: the scroing method of the cross validation
                     for regression: "neg_mean_absolute_error", "neg_mean_squared_error", "r2", ...
        """
        score = cross_val_score(self.model, train_X, train_Y, cv=cv, scoring=scoring).mean()
        if verbose:
            print("The {} of cross validation is {}".format(scoring, score))
        return score

    def hyperopt(self, train_X, train_Y, uniform_dict, int_dict, choice_dict, maximum=True, max_evals=10, cv=5, scoring="neg_mean_squared_error"):
        """
        hyperparameter optimization

        Args:
            uniform_dict: the dictionary contains the hyperparameters in float form
            int_dict: the dictionary contains the hyperparameters in int form
            choice_dict: the dictionary contains the hyperparameters in other discrete form
        """
        space, int_key, choice_key = {}, [], []
        # define the type of the hyperparameters
        for key, value in uniform_dict.items():
            space.update({key:hp.uniform(key,value[0],value[1])})
        for key, value in int_dict.items():
            space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
            int_key.append(key)
        for key, value in choice_dict.items():
            space.update({key:hp.choice(key,value)})
            choice_key.append((key,value))

        # define the loss function
        def loss(params):
            self.model.set_params(**params)
            if maximum:
                return -self.cross_validation(train_X, train_Y, cv=cv, scoring=scoring)
            else:
                return self.cross_validation(train_X, train_Y, cv=cv, scoring=scoring)

        # process for hyperparameter pruning
        optparams = fmin(fn=loss, space=space, algo=tpe.suggest, max_evals=max_evals)
        for key in int_key:
            optparams[key] = int(optparams[key])
        for item in choice_key:
            optparams.update({item[0]:item[1][optparams[item[0]]]})
        # set the best hyperparameters to the model
        self.model.set_params(**optparams)
        print("The optimal parameters of model {} in terms of {} is {}".format(self.name, scoring, optparams))


class Ridge_Regression(Model):
    """
    Ridge Regression
    """
    def __init__(self):
        Model.__init__(self, Ridge())
        self.name = "Ridge"


class Lasso_regression(Model):
    """
    Lasso Regression
    """
    def __init__(self):
        Model.__init__(self, Lasso())
        self.name = "Lasso"


class Elastic_Net(Model):
    """
    Elastic Net
    """
    def __init__(self):
        Model.__init__(self, ElasticNet())
        self.name = "Elastic Net"


class RF(Model):
    """
    Random Forest Model
    """
    def __init__(self):
        Model.__init__(self, RandomForestRegressor())
        self.name = "RF"


class Adaboost(Model):
    """
    Adaboost Model
    """
    def __init__(self):
        Model.__init__(self, AdaBoostRegressor())
        self.name = "Adaboost"


class XGboost(Model):
    """
    XGboost Model
    """
    def __init__(self):
        Model.__init__(self, XGBRegressor())
        self.name = "XGboost"


class Catboost(Model):
    """
    Catboost Model
    """
    def __init__(self):
        Model.__init__(self, CatBoostRegressor())
        self.name = "Catboost"


class LightGBM(Model):
    """
    Light Gradient Boosting Machine Model
    """
    def __init__(self):
        Model.__init__(self, LGBMRegressor())
        self.name = "LightGBM"


class Baseline(Model):
    """
    Baseline
    """
    def __init__(self):
        Model.__init__(self, None)
        self.name = "baseline"



# Example about how to apply the model (cross validation and hyperparameter optimization)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
if __name__ == "__main__":
    data = load_diabetes()
    X, Y = data.data, data.target
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33)
    xgb = XGboost()
    xgb.cross_validation(train_X, train_Y)
    xgb.hyperopt(train_X, train_Y, uniform_dict={"lambda": (0.1,1.0)}, int_dict={"max_depth": (5,20)}, choice_dict={"booster": ["gbtree", "gblinear", "dart"]})
    xgbTrained = xgb.fit(train_X, train_Y)
    pred_Y = xgbTrained.predict(test_X)
    res = mean_squared_error(test_Y, pred_Y)
    print("The MSE is {}".format(res))

    ridge = Ridge()
    ridgeTrained = ridge.fit(train_X, train_Y)
    pred_Y = ridgeTrained.predict(test_X)
    res = mean_squared_error(test_Y, pred_Y)
    print("The MSE is {}".format(res))