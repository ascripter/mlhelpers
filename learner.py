# -*- coding: utf-8 -*-
"""
Wrapper(s) and helpers for easily creating and visualising sklearn models
"""

__all__ = ["Learner", "LearnerSK", "LearnerStat"]
__version__ = "0.1"
__author__ = "Andreas Pfrengle"

import copy
from itertools import combinations, chain
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy

from sklearn import impute, preprocessing, model_selection
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_selection import RFE, RFECV, f_regression
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import abline_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import LinAlgError

from mlhelpers.dataframe import DataFrameModifier


class Learner:
    """Baseclass for more specific learners like :class:`LearnerSK`.
    
    :param pandas.DataFrame df: Underlying DataFrame containing
        predictors and target(s). The DataFrame will be modified inplace
        to save memory. If that doesn't fit your needs pass a copy
        instead using ``df.copy(deep=True)``.
    :param modifier: DataFrameModifier instance that will be applied
        to ``df`` before splitting train and test data or 
        applying any models.
    :type modifier: mlhelpers.dataframe.DataFrameModifier, optional
    :param y_columns: Column name of target or list of column names
        if the problem is a multitarget problem (not yet supported)
    :type y_columns: list or str, optional
    :param imputer: Imputer_ to estimate missing data. Default is to replace
        missing data cells with the columns median.
    :type imputer: sklearn.impute.<ImputerInstance>, optional
    :param normalizer: Normalizes data of all columns. Default is to use
        sklearn's RobustScaler_ which normalizes to the IQR
        (interquartile range)
    :type normalizer: sklearn.preprocessing.<Normalizer instance>, optional
    :param bool multiclass: Set ``True`` if the problem is categorical
        and has at least one target variable with > 2 classes
    :param float test_size: Fraction of dataset that will be used
        as test set.
    :param test_df: Overrides `test_size`. Explicitly hands over a DataFrame
        as test set. The 1st argument ``df`` is then interpreted as
        training set
    :type test_df: pd.DataFrame, optional
        
    .. todo::
        - Multitarget Modelling
        - Stratification of train/test split
        
    .. _Imputer: https://scikit-learn.org/stable/modules/impute.html
    .. _RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    """    
    def __init__(self, df, modifier=None, y_columns=[],
                 imputer=impute.SimpleImputer(missing_values=np.nan, strategy='median'),
                 normalizer=preprocessing.RobustScaler(copy=False),
                 multiclass=False,
                 test_size=0.25, test_df=None,
                 random_state=None):
        y_columns = [y_columns] if type(y_columns) is str else y_columns
        if modifier is None:
            modifier = DataFrameModifier(drop_nan_y=False, time_to_float=False)
        if id(df) not in modifier._applied_to_df:
            modifier.apply_to(df, y_columns)
        elif y_columns:
            warnings.warn("Modfier was already applied to df. `y_columns`"
                          " argument is ignored")
        
        self.df = df
        self.modifier = modifier
        self.test_size = test_size
        self.test_df = test_df
        self.x_columns = self.modifier.x_columns
        self.y_columns = y_columns
        
        cat_col = [_['name'] for _ in modifier._categorical_columns]
        is_cat = np.array([y in cat_col for y in y_columns])
        if is_cat.any() and not is_cat.all():
            warnings.warn("If mutliple outputs are to be modelled, all target "
                          "columns need to be either categorical or continuous. "
                          "Model training will most likely raise errors.")
        elif is_cat.all():
            self.is_cat = True
        else:
            self.is_cat = False
        self.imputer = imputer
        self.normalizer = normalizer
        self.is_multiclass = multiclass
        self.is_multitarget= len(y_columns) > 1
        self.mdls = []
        self.loss_train = []
        self.loss_test = []
        self.feature_importances = []
        self.accuracy_train = []
        self.accuracy_test = []
        self._is_pruned = False
        self._X_train = []    # series of data if multiple models 
        self._X_test = []     # with different splits are trained
        self._Y_train = []
        self._Y_test = []

    def inspect_df(self, mode="text"):
        """Runs :meth:`DataFrameModifier.inspect` on the underlying
        DataFrame.
        """
        return self.modifier.inspect(self.df)

    def get_XY(self, df, test_size=0.25, shuffle=True, random_state=None):
        """
        Applies imputation, then normalization, then performing train/test
        split and returns tuple of ``(X_train, X_test, Y_train, Y_test)``
        if ``test_size`` is given, otherwise ``(X, Y)``

        :param pandas.DataFrame df: DataFrame to be split into X and Y
        :param test_size: Fraction of rows to be used as test set.
            If ``None`` is passed, there will be no train/test split
        :type test_size: float or type(None)
        :param bool shuffle: Whether train/test split is shuffled or not
        :param random_state: random state
        :type random_state: int or type(None)

        """
        X = df.loc[:, self.x_columns]
        Y = df.loc[:, self.y_columns]
        
        if self.imputer is not None:
            X.loc[:, :] = self.imputer.fit_transform(X)
        
        if test_size is not None:
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                    X, Y, test_size=test_size, shuffle=shuffle,
                    random_state=random_state)
        else:
            X_train, X_test, Y_train, Y_test = X, None, Y, None
        #else:
        #    X_train, X_test = X[np.logical_not(X.index.isin(self.test_rows)), :], X.loc[self.test_rows, :]
        #    Y_train, Y_test = Y[np.logical_not(Y.index.isin(self.test_rows)), :], Y.loc[self.test_rows, :]
        if self.normalizer is not None:
            X_train = self.normalizer.fit_transform(X_train)
            if X_test is not None:
                X_test = self.normalizer.transform(X_test)
        if len(self.y_columns) == 1:
            # convert column vector to 1d array
            Y_train = Y_train.values[:,0]
            if Y_test is not None:
                Y_test = Y_test.values[:,0]
        if X_test is not None:
            print("X_train shape: {}, Y_train shape: {}".format(X_train.shape, Y_train.shape))
            print("X_test shape:  {}, Y_test shape:  {}".format(X_test.shape, Y_test.shape))
        return X_train, X_test, Y_train, Y_test
    
    def set_train_test(self, random_state=None):
        if self.test_df is not None:
            X_train, _, Y_train, _ = self.get_XY(self.df, test_size=None, random_state=random_state)
            X_test, _, Y_test, _ = self.get_XY(self.test_df, test_size=None, random_state=random_state)
        else:
            X_train, X_test, Y_train, Y_test = \
                self.get_XY(self.df, test_size=self.test_size, random_state=random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
    def plot_loss(self, loss_train, loss_test):
        x = range(len(loss_train))
        plt.plot(x, loss_train, 'b.-', x, loss_test, 'r.-')
        plt.yscale('log')
        plt.title("Train (blue) and test loss (red); MSE")
        plt.show()       

    def plot_residuals_histogram(self, mdl_fitted, X_train, X_test, Y_train, Y_test):
        Y_all = np.concatenate([Y_train, Y_test], axis=0)
        
        # Y- and residuals-histogram
        plt.title("Residuals Histogram (Baseline vs. Model, Test set)")
        data_residuals = np.mean(Y_all) - Y_all
        model_residuals = mdl_fitted.predict(X_test) - Y_test
        _min = min(np.min(data_residuals), np.min(model_residuals))
        _max = max(np.max(data_residuals), np.max(model_residuals))
        plt.hist(model_residuals, bins=np.linspace(_min, _max, 50))
        plt.hist(data_residuals, bins=np.linspace(_min, _max, 50), histtype="step")
        plt.show()
        
    def plot_pred_vs_var(self, mdl_fitted, X_test, Y_test):
        #Prediction vs. Variable
        plt.title("Prediction vs. Variable (Test set)")
        plt.scatter(Y_test, mdl_fitted.predict(X_test), marker=".")
        y = np.linspace(np.min(Y_test), np.max(Y_test), 2)
        plt.plot(y, y, c="black", lw=2)
        plt.show()
        print("Test score (R²): {}".format(mdl_fitted.score(X_test, Y_test)))
    
    def plot_accuracy(self, means, std=0):
        plt.title("Prediction Accuracy (Train vs. Test set)")
        plt.bar([0, 1], means, tick_label=["train", "test"], yerr=std)
        plt.text(0, 0.1, f"{means[0]:.3f}", ha="center")
        plt.text(1, 0.1, f"{means[1]:.3f}", ha="center")
        plt.show()

    def plot_roc_curve(self, mdl_fitted, X_test, Y_test):
        """Plot the `Receiver Operating Characteristic`_ on the test set
        
        .. _`Receiver Operating Characteristic`: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        """
        y_score = mdl_fitted.decision_function(X_test)
        if self.is_multiclass:
            raise NotImplementedError("ROC curve for multiclass not implemented")
        fpr, tpr, _ = roc_curve(Y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.title(f"ROC curve (area = {roc_auc:.2f})")
        plt.plot(fpr, tpr, color="darkorange", lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()                

    def plot_classifier(self, mdl_fitted, x1, x2):
        for x_name in (x1, x2):
            if x_name not in self.x_columns:
                raise ValueError(f"'{x_name}' is not a valid column")
        i1, i2 = self.x_columns.index(x1), self.x_columns.index(x2)
        X = self.df.loc[:, [x1, x2]].to_numpy()
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        hx = (x_max - x_min) / 100
        hy = (y_max - y_min) / 100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
        
        Xmean = self.df.loc[:, self.x_columns].mean(axis=0).to_numpy()
        Xpred = np.empty((10000, Xmean.shape[0]))
        Xpred[:, :] = Xmean
        Xpred[:, i1] = xx.ravel()
        Xpred[:, i2] = yy.ravel()
        Z = mdl_fitted.predict(Xpred)
        
        for y_name in self.y_columns:
            Y = self.df[y_name].to_numpy()
            
            
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
            
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
            plt.title('Classification using Support Vector Machine (2D projection)')
            plt.axis('tight')
            plt.show()
        
        
    def sorted_feature_importances(self, feature_importances, n=10, errors=None):
        cols_ordered = [x for _, x in sorted(zip(feature_importances, self.x_columns))][-n:]
        errors_ordered = [x for _, x in sorted(zip(feature_importances, errors))][-n:]\
            if errors is not None else [0] * min(n, len(cols_ordered))
        fi_ordered = sorted(feature_importances)[-n:]
        return cols_ordered, fi_ordered, errors_ordered
        
    def plot_feature_importances(self, feature_importances, n=10, errors=None):
        cols, fi, err = self.sorted_feature_importances(feature_importances, n, errors)
        plt.barh(cols, fi / max(fi), xerr=err / max(fi))
        if errors is not None:
            plt.title("Relative Variable Importance with 1s error bars")
        else:
            plt.title("Relative Variable Importance")
        plt.show()
     
    def print_feature_importances(self, feature_importances, n=10, errors=None):
        cols, fi, errors = self.sorted_feature_importances(feature_importances, n, errors)
        w = 3 + max([len(col) for col in cols])
        for c, f, e in zip(reversed(cols), reversed(fi), reversed(errors)):
            print("{:{w}s}: {:0.4f}   +/- {:0.4f}".format(c, f, e, w=w))    

    def prune_by_vif(self, vif_thresh=5):
        """Modifies underlying DataFrame inplace by eliminating multi collinear
        columns of exogenous / predictive variables depending on the
        `Variance Inflation Factor`_ (VIF), i.e. prune columns where
        ``VIF > vif_thresh``. The VIF of a given variable is calculated as
        $\frac{1}{1 - R^2}$ where $R^2$ is the correlation coefficient
        of the linear regression of this variable on all other variables.
        The algorithm will successively eliminate the column with
        the highest VIF until ``vif_thresh`` is reached.
                
        :param vif_thresh: Threshold for Variance Inflation Factor
        :type vif_thresh: float, optional
        
        _`Variance Inflation Factor`: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
        """
        X = self.df[self.x_columns].to_numpy()
        if self.imputer is not None:
            X = self.imputer.fit_transform(X)        
        
        variables = list(range(X.shape[1]))  # we need list to be able to delete elements
        column_names = copy.copy(self.x_columns)  # avoid changing input inplace
        
        dropped_column = True
        while dropped_column:
            dropped_column = False
            vif = []
            for ix in range(len(variables)):
                try:
                    vif.append(variance_inflation_factor(X[:, variables], ix))
                except LinAlgError:
                    print("LinAlgError on column {}: {}".format(ix, column_names[ix]))
                    vif.append(1)
                
            for _vif, ix in reversed(sorted(zip(vif, range(len(variables))))):
                # pick variable with highest vif that's not in self._dont_prune_columns
                if _vif > vif_thresh and column_names[ix] not in \
                        self.modifier._dont_prune_columns:
                    print(f'dropping col {variables[ix]} "{column_names[ix]}"'
                          f'with VIF={_vif:.1f}')
                    del variables[ix]
                    del column_names[ix]
                    dropped_column = True
                    break            
    
        print(f'{len(column_names)} Remaining variables: {column_names}')
        #self._pruning_variables_remaining = variables
        self.df = self.df[column_names + self.y_columns]
        self.x_columns = column_names
        self._is_pruned = True
      
    def identify_multicollinearity(self, factors_max=3, top=None, R2_min=0.8,
                                   out_path='identify_multicollinearity.csv'):
        """Conduct multiple linear regressions, regressing each variable
        of the underlying DataFrame (including target variable(s))
        on all possible combinations of ``factors_max`` other variables
        (brute force multicollinearity). This allows to identify *all*
        linear correlations that are highly predictive for each variable.
        
        Returns a score-table where each row represents
        one regression. Columns marked ``x`` are linear predictors on the
        column marked ``y``. Resulting $R^2$ and $R_adj^2$ are reported
        where $R_adj^2$ is adjusted by the size of the subspace ``k`` vs.
        total number of dimensions ``n``:
        $$1 - R_adj^2 = (1 - R^2) * \frac{n + 1}{n + 1 - k}$$

        :param int factors_max: Maximum number of predictors
            in each combination
        :param top: Can be an integer to limit the output table
            to the ``top`` highest ranking results
        :type top: int or NoneType
        :param float R2_min: Regressions that don't reach that score
            are dropped from the result table
        :param out_path: Target file (csv) to save the results, or ``None``
            to not save any result but just return it
        :type out_path: :ref:`path-like object` or NoneType
        
        :return: pd.DataFrame containing the regression results
        
        .. _`path-like object`: https://docs.python.org/3/glossary.html#term-path-like-object
        """
        def log(iterations):
            print("    {}: {:6d} regressions".format(
                    time.strftime("%H:%M:%S", time.localtime()), iterations))
        if self.modifier and self.modifier._categorical_columns:
            for cat in self.modifier._categorical_columns:
                name, mode = cat['name'], cat['mode']
                if mode in self.modifier.CAT_MODES_LABEL:
                    warnings.warn(
                        f"Categorical column '{name}' should be 1-hot encoded instead of label encoded. "
                         "Use DataFrameModifier.cat_column(name, mode='1-hot'). "
                         "Results for categorical columns are wrong if more than 2 labels are present."
                    )
        df = self.df.loc[:,[_ for _ in self.df.columns if _ not in self.y_columns]]
        n = len(df.columns)
        n_factors = range(1, factors_max + 1)
        n_subspaces = sum([scipy.special.comb(n, k + 1, exact=True) for k in n_factors])
        n_regressions = sum([(k + 1) * scipy.special.comb(n, k + 1, exact=True) for k in n_factors])
        
        if n_regressions >= 10000:
            print(f"Calculating {n_regressions} regression analyses to determine "
                  "multi-collinearity may take some time")
            log(0)
        
        if self.imputer is not None:
            arr = self.imputer.fit_transform(df.to_numpy())
            X = pd.DataFrame(arr, index=df.index)
        else:
            X = df
        row = 0
        result = np.zeros((n_regressions, df.shape[1])).astype('U1')
        result[:,:] = ""
        resultR = np.zeros((n_regressions, 2)).astype(np.float32)
        resultn = np.zeros((n_regressions, 1)).astype(np.int8)
        for k in n_factors:
            for subspace in combinations(range(n), k + 1):
                for predictors in combinations(subspace, k):
                    idx_x = list(predictors)
                    idx_y = [_ for _ in subspace if _ not in predictors][0]
                    idx_row = df.notna().iloc[:, idx_y]  # exclude NaN-rows
                    x = X.loc[idx_row, :].iloc[:, idx_x]
                    y = df.loc[idx_row, :].iloc[:, idx_y]
                    mdl = LinearRegression().fit(x, y)
                    R2 = mdl.score(x, y)
                    if (row + 1) % 10000 == 0:
                         log(row+ 1)
                    R2_adj = 1 - (1 - R2) * (n + 1) / (n + 1 - k)
                    result[row, idx_x] = "x"
                    result[row, idx_y] = "y"
                    resultR[row, 0] = R2
                    resultR[row, 1] = R2_adj
                    resultn[row, 0] = len(predictors)
                    row += 1
        resultdf = pd.concat([pd.DataFrame(result, columns=list(df.columns)),
                              pd.DataFrame(resultR, columns=["R2", "R2_adj"]),
                              pd.DataFrame(resultn, columns=["n_factors"])], axis=1)
        resultdf.sort_values("R2_adj", axis=0, ascending=False, inplace=True)
        resultdf = resultdf.loc[resultdf.R2 >= R2_min, :]
        if top is not None:
            resultdf = resultdf.head(top)
        
        # drop columns that are unused
        resultdf = resultdf.loc[:, (resultdf.to_numpy() != "").any(axis=0)]
        if out_path:
            resultdf.to_csv(out_path, sep=";", index=False, float_format="%.3f")
        return resultdf
    
class LearnerSK(Learner):
    MODELS = ["elasticnet", "svm", "boosting", "linear"]
    def _append_loss(self, mdl_fitted, X_train, X_test):
        loss_train, loss_test = [], []
        if self.is_cat:
            for y_pred in mdl_fitted.staged_decision_function(X_train):
                loss_train.append(mdl_fitted.loss_(self.Y_train, y_pred))
            for y_pred in mdl_fitted.staged_decision_function(X_test):
                loss_test.append(mdl_fitted.loss_(self.Y_test, y_pred))
        else:
            for y_pred in mdl_fitted.staged_predict(X_train):
                loss_train.append(np.mean((y_pred - self.Y_train)**2)) 
            for y_pred in mdl_fitted.staged_predict(X_test):
                loss_test.append(np.mean((y_pred - self.Y_test)**2))
        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        
    def elasticnet(self):
        self.set_train_test()
        mdl = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                           alphas=[1, 2, 5, 10, 15, 20, 30, 50, 100],
                           tol=1e-4, cv=5,
                           verbose=0)
        mdl = RFECV(mdl, verbose=2)
        mdl.fit(self.X_train, self.Y_train)
        return mdl
    
    def RFECV(self, mdl, rfecv):
        """Feature selection with recursive feature elimination and cross-validated
           selection of the best number of features"""
        minf = getattr(mdl, 'max_features', 1)
        step = (self.X_train.shape[1] - minf) // 10
        selector = RFECV(mdl, cv=rfecv, verbose=0, min_features_to_select=minf, step=step)
        selector = selector.fit(self.X_train, self.Y_train)
        rk = selector.ranking_
        X_train, X_test = self.X_train[:,rk == 1], self.X_test[:,rk == 1]
        print("{} variables removed by recursive feature elimination ({} remaining)".format(
            rk[rk > 1].size, rk[rk == 1].size))
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        plt.show()
        mdl = selector.estimator_
        return mdl, X_train, X_test

    def linear(self, rfecv=False, plot=True, **mdl_params):
        """Apply :meth:`LinearRegression` on cardinal or
        :meth:`LogisticRegression` on categorical target respectively
        """
        self.set_train_test()
        params = {}
        params.update(mdl_params)
        if self.is_cat:
            mdl = LogisticRegressionCV(**params)
        else:
            mdl = LinearRegression(**params)
        return self._process_model(mdl, rfecv=rfecv,
                                   plot=plot)        


    def svm(self, rfecv=False, plot=True, **mdl_params):
        """Perform Support Vector Modelling (either regression or
        classification depending on target). Default kernel is ``'linear'``,
        if this is not complex enough set ``kernel='rbf'``
        """
        self.set_train_test()
        params = {'kernel': 'linear'}
        params.update(mdl_params)
        if self.is_cat:
            mdl = SVC(**params)
        else:
            mdl = SVR(**params)
        return self._process_model(mdl, rfecv=rfecv,
                                   plot=plot)
        

    def boosting(self, rfecv=False, plot=True, n_features=10, **mdl_params):
        """Gradient Tree Boosting for classification_ or regression_ tasks.
        
        :param bool rfecv: ``True``: Apply feature ranking with recursive
            feature elimination and cross-validated selection of the best
            number of features. See :meth:`sklearn.feature_selection.RFECV`
        :param bool plot: ``True``: Create plots
        :param int n_features: Number of features analysed in the
            feature importance graph
        :param \*\*mdl_params: Any other keyword arguments are passed on
            to the model instantiation
        
        .. _classification: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        .. _regression: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        """
        # weak-learner boosting
        
        self.set_train_test()
        params = {
                'n_estimators': 2000,
                'learning_rate': .1,
                'min_samples_leaf': 0.01,
                'max_depth': 3,  # see https://en.wikipedia.org/wiki/Gradient_boosting#Size_of_trees
                'max_features': 3,
                'validation_fraction': .25,
                'n_iter_no_change': 50,
                'tol': 1e-8,
                'verbose': 0}
        params.update(mdl_params)
        if self.is_cat:
            mdl = GradientBoostingClassifier(**params)
        else:
            mdl = GradientBoostingRegressor(**params)
        return self._process_model(mdl, rfecv=rfecv,
                                   plot=plot, n_features=n_features)

    def _process_model(self, mdl,
                       rfecv=False, plot=True, n_features=10):       
        # Wrap MutliOutputregressor if there is > 1 target variable
        if self.is_cat and self.is_multitarget:
            mdl = MultiOutputRegressor(mdl)

        if rfecv:
            mdl, X_train, X_test = self.RFECV(mdl, rfecv)
        else:
            X_train, X_test = self.X_train, self.X_test
            mdl.fit(X_train, self.Y_train) 
        self.mdls.append(mdl)
        self._X_train.append(X_train)
        self._X_test.append(X_test)
        self._Y_train.append(self.Y_train)
        self._Y_test.append(self.Y_test)
        
        if hasattr(mdl, 'staged_decision_function') or \
                hasattr(mdl, 'staged_predict'):
            self._append_loss(mdl, X_train, X_test)
        
        if hasattr(mdl, 'feature_importances_'):
            self.feature_importances.append(mdl.feature_importances_)
        
        if self.is_cat:
            accuracy_train = mdl.score(X_train, self.Y_train)
            accuracy_test = mdl.score(X_test, self.Y_test)
            self.accuracy_train.append(accuracy_train)
            self.accuracy_test.append(accuracy_test)
        
        if plot:
            self._plot_all()
        
        return mdl
 
    def _plot_all(self, n_features=10, n_plot=3):
        mdl = self.mdls[0]
        for i in range(min(n_plot, len(self.loss_train))):
            self.plot_loss(self.loss_train[i], self.loss_test[i])
            
        if hasattr(mdl, 'feature_importances_'):
            if len(self.feature_importances) > 1:
                fi = np.array(self.feature_importances)
                fi_mean = np.mean(fi, axis=0)
                fi_std = np.std(fi, axis=0)
            else:
                fi_mean = mdl.feature_importances_
                fi_std = None
            self.plot_feature_importances(fi_mean, n=n_features, errors=fi_std)
            print("Feature importance (non-normalized)")
            self.print_feature_importances(fi_mean, n=n_features, errors=fi_std)
        
        if self.is_cat:
            for i in range(min(n_plot, len(self.mdls))):
                self.plot_roc_curve(self.mdls[i], self._X_test[i], self._Y_test[i])
            if len(self.accuracy_train) > 1:
                accuracy_means = (np.mean(self.accuracy_train), np.mean(self.accuracy_test))
                accuracy_std = (np.std(self.accuracy_train), np.std(self.accuracy_test))
            else:
                accuracy_means = (self.accuracy_train[0], self.accuracy_test[0])
                accuracy_std = None
            self.plot_accuracy(accuracy_means, accuracy_std)
        
        else:
            for i in range(min(n_plot, len(self.mdls))):
                self.plot_residuals_histogram(self.mdls[i],
                                              self._X_train[i], self._X_test[i],
                                              self._Y_train[i], self._Y_test[i])
                self.plot_pred_vs_var(self.mdls[i], self._X_test[i], self._Y_test[i])
                    

    def multiples(self, method, n_models=10, n_features=10, n_plot=3, **mdl_params):
        """Wrapper for other methods to repeatedly fit similar models
        on different splits of train / test data to evaluate the robustness
        of model.
        
        :param str method: Any ``LearnerSK`` method building a model
        :param int n_modesl: Number of models from which mean and std
            of appropriate metrics will be calculated and plotted
        """
        if method not in self.MODELS:
            raise AttributeError(f"The method {method} doesn't exist. Choose one of {self.MODELS}")
        print(f"Repeat {method} with {n_models} different train/test splits to evaluate robustness of model")
        model_fn = getattr(self, method)
        for i in range(n_models):
            self.mdls.append(model_fn(rfecv=False, plot=False, **mdl_params))
        self._plot_all(n_features=n_features, n_plot=n_plot)
        return self.mdls
        
class LearnerStat(Learner):
    """Learner for statsmodels regression models
    """

    def _set_f_columns(self):
        self._df_columns = [_ for _ in self.df.columns]
        self.df.columns = [re.sub("\W", "_", _) for _ in self.df.columns]
    
    def _restore_columns(self):
        self.df.columns = self._df_columns

    def plot_adjusted(self, x_names=None, y_names=None, ncols=2):
        x_names = self.x_columns if x_names is None else x_names
        y_names = self.y_columns if y_names is None else y_names
        nrows = (len(x_names) - 1) // ncols + 1
        nrows = (len(x_names) + 1) // ncols
        figsize = (4 * ncols, 3 * nrows)
        for y_name in y_names:
            fig = plt.figure(figsize=figsize)
            plt.subplots_adjust(hspace=0.25, wspace=0.3)
            for i, x_name in enumerate(x_names):
                x_other = self.df.loc[:, [_ for _ in self.x_columns if _ != x_name]]
                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.set_title("")
                res_yaxis = self.Mdl(self.df[y_name], x_other, family=self.family).fit()
                y_resid = res_yaxis.resid_generalized
                x = self.df[x_name]
                ax.plot(x, y_resid, 'o')
                fitted_line = sm.OLS(y_resid, x, family=self.family).fit()
                fig = abline_plot(0, fitted_line.params[0], color='k', ax=ax)
                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)
            fig.suptitle("Adjusted Response Plot (generalized residuals)", fontsize="large")
            
    
    def linear(self, formula=None, plot=True):
        self.set_train_test()
        if self.is_cat and not self.is_multiclass:
            self.Mdl = sm.Logit
            self.family = sm.families.Binomial()
        elif self.is_cat and self.is_multiclass:
            raise NotImplementedError("Logistic Regression for multiclass problem not implemented")
        else:
            self.Mdl = sm.OLS
            self.family = sm.families.Binomial()
        
        for y_name in self.y_columns:
            if formula is None:
                mdl = self.Mdl(self.df[y_name], self.df[self.x_columns]).fit()
            else:
                self._set_f_columns()
                mdl = smf.glm(formula=formula, data=self.df,
                              family=self.family).fit()
                self._restore_columns()
            self.mdls.append(mdl)
        return self._process_models_and_plot(plot=plot)
    
    def _process_models_and_plot(self, plot=True):
        for mdl in self.mdls:
            print(mdl.summary())
            if plot:
                n = int(np.sqrt(len(self.x_columns)))
                figsize = (4 * n, 4 * n)
                fig = plt.figure(figsize=figsize)
                sm.graphics.plot_partregress_grid(mdl, fig=fig)
                self.plot_adjusted(ncols=4)
        if len(self.mdls) == 1:
            return self.mdls[0]
        return self.mdls
        