# -*- coding: utf-8 -*-
"""
Wrapper(s) and helpers for easily handling DataFrames in the context
of creating sklearn models. 
"""

__all__ = ["DataFrameModifier", "read_excel", "read_csv"]
__version__ = "0.1"
__author__ = "Andreas Pfrengle"

import copy
import logging
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn import preprocessing
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#from statsmodels.tsa.stattools import LinAlgError


clock_fn = lambda a: (pd.to_numeric(a) % (1E9 * 3600 * 24)).astype(float) / 1E9 # seconds since midnight
time_fn = lambda a, b: (pd.to_numeric(a) - pd.to_numeric(b)).astype(float) / 1E9  # delta seconds
day_plus_hour = lambda d, h: pd.to_numeric(d) / 1E9 + clock_fn(h)  # date in seconds since epoch


def read_excel(path, sheet, datetime_columns=[], header=0, **kwargs):
    """Read Excel file, correctly interpreting Date and Time columns that
    can be addressed by name instead of column index.
    
    :param path: Path to source file
    :type path: str or path-like object
    :param str sheet: Name of the sheet you're interested in
    :param list datetime_columns: Columns with these names are parsed
        as date-columns
    :param int header: The 0-based index of the header row in the
        source file
    :param \*\*kwargs: Further parameters are passed on
        to `pandas.read_excel()`_
    :return: pd.DataFrame
    
    .. _`pandas.read_excel()`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
    """
    h = pd.read_excel(path, sheet, header=header).columns
    parse_dates = [i for i, name in enumerate(h) if name in datetime_columns]
    df = pd.read_excel(path, sheet, header=header,
                     parse_dates=parse_dates, **kwargs)
    return df


def read_csv(path, datetime_columns=[], header=0, **kwargs):
    """Read CSV file, correctly interpreting Date and Time columns that
    can be addressed by name instead of column index.
    
    :param path: Path to source file
    :type path: str or path-like object
    :param list datetime_columns: Columns with these names are parsed
        as date-columns during file reading
    :param int header: The 0-based index of the header row in the
        source file
    :param \*\*kwargs: Further parameters are passed on
        to `pandas.read_csv()`_
    :return: pd.DataFrame
    
    .. _`pandas.read_csv()`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    """    
    encoding = kwargs.get('encoding', 'utf8')
    h = pd.read_csv(path, header=header, encoding=encoding).columns
    parse_dates = [i for i, name in enumerate(h) if name in datetime_columns]
    df = pd.read_csv(path, header=header, encoding=encoding,
                     parse_dates=parse_dates, **kwargs)
    return df
   

class DataFrameModifier:
    """Defines modifications to be performed on a raw pandas DataFrame
    in the sense of a data preprocessor if you have i.e. a heterogeneous
    Excel table as source that contains different data types and descriptive
    or other non-relevant columns to be excluded, categorical columns
    to be converted or want to filter some data.
    
    Advantages are:
        
    - Being able to use a pd.DataFrame as input for sklearn and other
      ML models by performing all the relevant preprocessing
      on the DataFrame instead of a naked values array (allows to access
      data by their column names).
    - High level functions for data interaction / transformation
    
    Disadvantages:
    
    - Possible overhead of the data structure that may lack performance
      for large data sets
      
    Typically you will use:
        
    - :meth:`exclude_column` explicitly excludes columns.
    - :meth:`include_column` alternatively for a complementary definition
      of relevant columns    
    - :meth:`cat_columns` explicitly tells which columns are of
      categorical data type
    - :meth:`filters` will mask certain values (i.e. set to ``np.nan``)(out-of-tolerance, outliers 
    
    :param bool drop_nan_y: Drop rows where target column is nan
    :param bool drop_nan_x: Drop rows where *any* predictor column is nan
    :param bool time_to_float: Automatically convert datetime and
        timedelta columns to float
        
    .. _`Variance Inflation Factor`: https://en.wikipedia.org/wiki/Variance_inflation_factor
    """
    CAT_MODES_LABEL = ("label", "int", "ordinal", "ord")
    CAT_MODE_LABEL = "label"
    CAT_MODES_1HOT = ("1-hot", "1hot", "onehot", "one-hot", "binary", "bin")
    CAT_MODE_1HOT= "1-hot"
    CAT_MODES = CAT_MODES_LABEL + CAT_MODES_1HOT
    def __init__(self,
                 drop_nan_y=True,
                 drop_nan_x=False,
                 time_to_float=True,
                 ):
        self._calc_columns = OrderedDict()
        self._exclude_columns = []
        self._include_columns = []
        self._dont_prune_columns = []
        self._categorical_columns = []
        self._categorical_columns_map = {}
        self._categorical_columns_encoder = {}
        self._filters = {}
        self.x_columns = self.xi_columns = self.y_columns = self.yi_columns = None
        self._baseline_mean = []

        self._drop_nan_y = drop_nan_y
        self._drop_nan_x = drop_nan_x
        self._time_to_float = time_to_float
        #self._pruning_variables_remaining = None
        self._applied_to_df = []   # list of object-ids the modifier was already applied to
        
    def __str__(self):
        s = "<DataFrameModifer"
        for name, attr in (('New columns', '_calc_columns'),
                           ('Exclude columns', '_exclude_columns'),
                           ('Include columns', '_include_columns'),
                           ('Never prune columns', '_dont_prune_columns'),
                           ('Categorical columns', '_categorical_columns'),
                           ('Filters', '_filters'),
                           ('Transformations', '_baseline_mean'),
                           ('Drop rows where target is NaN', '_drop_nan_y'),
                           ('Drop rows where any predictor is NaN', '_drop_nan_x'),
                           ('Time to float', '_time_to_float')):
            val = getattr(self, attr)
            if val:
                s += f" - {name}: {val}"
        if s == "<DataFrameModifer ":
            s += " - No Modifications Defined"
        s += ">"
        return s
    
    def __repr__(self):
        return self.__str__()
        
    def new_column(self, name, fn, *args):
        """Add new column based on other columns of the DataFrame. Example:
            
        >>> df = pd.DataFrame([[1,2,3], [4,5,6]], columns=list('abc'))
        >>> m = DataFrameModifier()
        >>> m.new_column('d', lambda x, y, z: x * y + z, 'a', 'b', 'c')
        >>> m.apply_to(df)
        >>> df
           a  b  c   d
        0  1  2  3   5
        1  4  5  6  26
        
        :param str name: Name of the new column
        :param function fn: Function taking as many variables as there
            are columns involved in the formula
            (i.e. ``lambda x, y, z: x * y + z``)
        :param str \*args: column names in the original DataFrame
            representing the columns that will be positional arguments
            to function ``fn``
        :return: None
        """
        self._calc_columns[name] = lambda df: fn(*[df[col] for col in args]) 
    
    def exclude_column(self, names):
        """Columns named here will be excluded from a DataFrame where
        the Modifer will be applied to. Complementary to
        :meth:`include_column`
        
        :param names: Column name or list of column names
        :type names: str or list
        :return: None
        """
        if type(names) is str:
            names = [names]
        if self._include_columns:
            warnings.warn("The DataFrameModifier has already include_columns"
                          " defined. This setting has precedence,"
                          " exclude_columns will be ignored")
        self._exclude_columns.extend(names)
    
    def exclude_columns(self, names):
        """Alias for :meth:`exclude_column`
        """
        self.exclude_column(names)

    def include_column(self, names):
        """This is the complementary definition of :meth:`exclude_column`.
        *Only* the columns listed here included in the data, all others
        are excluded. Has precedence over :meth:`exclude_column`
        
        :param names: Column name or list of column names
        :type names: str or list
        :return: None
        """
        if type(names) is str:
            names = [names]
        if self._exclude_columns:
            warnings.warn("The DataFrameModifier has already exclude_columns"
                          " defined. include_columns has precedence,"
                          " so exclude_columns will be ignored")
        self._include_columns.extend(names)
    
    def include_columns(self, names):
        """Alias for :meth:`include_column`
        """
        self.exclude_column(names)
    
    def dont_prune_column(self, names):
        """Columns defined here are never pruned by any automated pruning
        algorithm of downstream learners.
        Use this on columns you want to enforce being part of the model
        even if they'll be statistically non-optimal 
        
        :param names: Column name or list of column names
        :type names: str or list
        :return: None
        """
        if type(names) is str:
            names = [names]
        self._dont_prune_columns.extend(names)
    
    def dont_prune_columns(self, names):
        """Alias for :meth:`dont_prune_column`
        """
        self._dont_prune_column(names)
    
    def cat_column(self, names, mode, impute=None, inplace=True, dropfirst=True):
        """Declare column(s) as categorical data type. ``mode`` tells whether
        a categorical column shall be encoded as labels_ or as
        ``n-1`` binary `1-hot`_ columns for ``n`` different labels.
        
        >>> df = pd.DataFrame(list('bbaccc'), columns=['labelvar'])
        >>> df['onehotvar'] = list('cddee') + [np.nan]
        >>> m = DataFrameModifier()
        >>> m.cat_column('labelvar', 'label')
        >>> m.cat_column('onehotvar', '1-hot', impute='d')
        >>> m.apply_to(df)
        >>> df
           labelvar  onehotvar1  onehotvar2
        0       1.0         0.0         1.0
        1       1.0         0.0         0.0
        2       0.0         0.0         0.0
        3       2.0         1.0         0.0
        4       2.0         1.0         0.0
        5       2.0         0.0         0.0
        
        :param names: Column name or list of column names
        :type names: str or list
        :param str mode:
        
            - ``"label"``: will append / replace the column's contents
              by integers.
              If ``inplace=False``, a column of the same name + suffix ``0``
              will be appended, otherwise the column will be replaced.
              Missing values are represented as ``NaN``. Category indices
              follow alphabetical order of classes.
            - ``"1-hot"``: Append as many columns as there are classes - 1,
              named like ``{name}0``, ``{name}1``, ``...``, where ``{name}0``
              will represent the most frequent label, the rest in descending
              order. However the first column with index ``0`` is dropped
              to prevent perfect multi collinearity of the encoded columns.
              So if every column is ``0``, this will represent the most
              frequent label (use :meth:`inspect` for details)
              
        :param str impute:
            
            - ``None`` (default): Eliminate rows where this column's value
              value is missing / NaN. 
            - ``value``: Replace missing value with ``value`` and then encode
              the data.
              
        :param bool inplace: 
            
            - ``True`` (default): For label encoded columns, the original
              column is replaced. For 1-hot encoded columns, the original
              column is dropped and only the indexed columns remain,
              appended at the end of the DataFrame.
            - ``False``: The original columns are kept. Any downstream
              learner would get the raw data (i.e. strings) passed
              into the model which will most likely raise errors.
              **This setting should always be** ``True`` **except for
              debugging reasons / raw data analysis**.
            
        :param bool dropfirst: 
            
            - ``True`` (default): In case of 1-hot encoding determines that
              the first column shall indeed be dropped as described above
            - ``False``: The first column will be kept instead (which will
              introduce multi-collinearity between the encoded columns.)
        
        :return: None
        
        .. _labels: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        .. _`1-hot`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        """
        if type(names) is str:
            names = [names]
        if type(mode) is str:
            mode = [mode] * len(names)
        if type(impute) not in (list, tuple):
            impute = [impute] * len(names)
        if type(inplace) is bool:
            inplace = [inplace] * len(names)
        if type(dropfirst) is bool:
            dropfirst = [dropfirst] * len(names)
        for i, m in enumerate(mode):
            if m not in self.CAT_MODES:
                raise TypeError(f"``mode`` must be either 'label' or '1-hot': {m}")
            mode[i] = self.CAT_MODE_1HOT if m in self.CAT_MODES_1HOT else m
            mode[i] = self.CAT_MODE_LABEL if m in self.CAT_MODES_LABEL else m
        for i, ip in enumerate(inplace):
            if type(ip) is not bool:
                raise TypeError(f"``inplace`` must be bool: {ip}")
        for i, d in enumerate(dropfirst):
            if type(d) is not bool:
                raise TypeError(f"``dropfirst`` must be bool: {d}")
        for name, m, im, ip, d in zip(names, mode, impute, inplace, dropfirst):
            self._categorical_columns.append({'name': name, 'mode': m,
                                              'impute': im, 'inplace': ip,
                                              'dropfirst': d})
    
    def cat_columns(self, *args, **kwargs):
        """Alias for :meth:`cat_column`
        """
        self.cat_column(*args, **kwargs)

    def get_cat_encoder(self, column):
        """Returns a tuple of the Encoder instance and categories / classes
        for the passed column name(s)
        
        :param str column: Column name or list of column names. If a list
            is passed, a list of ``(encoder, classes)`` will be returned
        :type column: str or list
        :return: tuple(s) of ``(encoder, classes)`` where ``encoder`` is 
            either a OneHotEncoder_ or LabelEncoder_ instance and ``classes``
            a `pd.Categorical`_ or ``encoder.classes_`` respectively.
        
        .. _LabelEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        .. _OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        .. _`pd.Categorical`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Categorical.html
        """
        if isinstance(column, str):
            return self._categorical_columns_encoder[column]
        return [self._categorical_columns_encoder[col] for col in column]

    def filters(self, filterdict):
        """Define one or more filters per column, i.e. to apply
        hard thresholds.
        
        :param dict filterdict: Dictionary of ``{column: function(df, x)}``
            where ``column`` is the name of a DataFrame's column and
            ``function(df, x)`` a function that accepts the DataFrame
            as first argument and the data values of ``column``
            as second argument. It should return a boolean array.
            In rows where this array evaluates to ``False`` the column
            will be set to ``np.nan``. Examples for filter-functions:
                
                - ``lambda df, x: x < 10.5``
                - ``lambda df, x: np.logical_and(0 < x, x < 100)``
                - ``lambda df, x: x < df['other_column']`` may modify ``x``
                  depending on another column's data
        
        :return: None
        """
        self._filters = filterdict

    def _apply_filters(self, df):
        for col, func in self._filters.items():
            if col not in df.columns:
                continue
            print("Apply filter on '{}'".format(col))
            df.loc[np.logical_not(func(df, df[col])), col] = np.nan

    def transform_by_category_mean(self, target_column,
                                   categorical_column,
                                   category_ref=None):
        """Normalize all values of ``target_column`` (*float*) depending
        on the values of ``categorical_column``. For each category, the 
        mean is calculated and subtracted, effectively normalizing each
        category to zero-mean. However if ``category_ref`` is provided,
        each category will be normalized to the mean of this reference
        category::
        
          df[x] += "mean of x where categorical_column = category_ref"
                 - "mean of x where categorical_column = current row's category"
        
        Doesn't affect rows where ``categorical_column`` is missing / NaN
        """
        self._baseline_mean.append((target_column, categorical_column, category_ref))  
        
    def _apply_transform_by_category_mean(self, df):
        for x, ccol, cat in self._baseline_mean:
            means = df.groupby(ccol).mean()
            if cat:
                mean_ref = means.loc[cat, x]   # scalar
            else:
                mean_ref = 0
            row_ix = df.loc[pd.notna(df[ccol]), ccol]        
            mean_this = means.loc[row_ix, x].values  # 1d array
            df.loc[pd.notna(df[ccol]), x] += mean_ref - mean_this    
        #return df

# =============================================================================
#     def prune_by_vif(self, X):
#         """
#         pass matrix of all exogenous variables and prune columns that have
#         a Variance Inflation Factor larger than `self._vif_thresh`.
#         
#         return (`pruned_matrix`, `indices`) with
#             `pruned_matrix`: matrix where collinear columns are removed.
#             `indices`: list of indices of the remaining columns
#                        (referring indices of the original matrix)
#         
#         https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
#         """
#         variables = list(range(X.shape[1]))  # we need list to be able to delete elements
#         if self.x_columns is not None:
#             assert len(self.x_columns) == len(variables), \
#             "len(self.x_columns) == {}, but must be same as X.shape[1] == {}".format(
#                     len(self.x_columns), len(variables)
#             )
#             column_names = copy.copy(self.x_columns)  # avoid changing input inplace
#         else:
#             column_names = variables
#         
#         dropped_column = True
#         while dropped_column:
#             dropped_column = False
#             vif = []
#             for ix in range(len(variables)):
#                 try:
#                     vif.append(variance_inflation_factor(X[:, variables], ix))
#                 except LinAlgError:
#                     print("LinAlgError on column {}: {}".format(ix, column_names[ix]))
#                     vif.append(1)
#                 
#             for _vif, ix in reversed(sorted(zip(vif, range(len(variables))))):
#                 # pick variable with highest vif that's not in self._dont_prune_columns
#                 if _vif > self._vif_thresh and column_names[ix] not in self._dont_prune_columns:
#                     print('dropping col {} "{}" with VIF={:.1f}'.format(
#                     variables[ix], column_names[ix], _vif))
#                     del variables[ix]
#                     del column_names[ix]
#                     dropped_column = True
#                     break            
#     
#         print('{} Remaining variables:'.format(len(column_names)))
#         #print(df.columns[variables])
#         print(column_names)
#         #self._pruning_variables_remaining = variables
#         self._is_pruned = True
#         return X[:, variables], variables
# =============================================================================

    def _set_categorical(self, df):
        """Encoding of categorical variables as dtype ``np.float32``.
        Encoder will be accessible via
        ``self._categorical_columns_encoder[col]``, where ``col``
        may be either be the old or the new column name.
    
        :param pandas.DataFrame df: Input DataFrame
        :return: None
        """
        y_columns_new = copy.copy(self.y_columns)
        
        # first iteration may drop rows and may hence influence labels on ALL
        # columns. Second iteration transforms the remainder (NaNs are removed)
        for d in self._categorical_columns:
            name, impute = d['name'], d['impute']
            if name not in df.columns:
                warnings.warn(f"Skipping column {name}: Not found in DataFrame")
                continue
            if impute is None:
                df.drop(index=df.loc[pd.isna(df[name]), :].index, inplace=True)
            else:
                df.loc[pd.isna(df[name]), name] = impute
        for d in self._categorical_columns:
            name, mode, inplace = d['name'], d['mode'], d['inplace']
            dropfirst = "first" if d['dropfirst'] else None
            if name not in df.columns:
                continue
            if mode.lower() in self.CAT_MODES_1HOT:
                # .value_counts sorts by frequency descending;
                # -1 for NaN values, but since we removed NaNs this won't occur.
                c = df[name].value_counts().index
                cat = pd.Categorical(df[name], categories=c)
                col_int = cat.codes
                
                # drop="first" means that there will be no column for code 0,
                # (most frequent label), i.e. all columns will be zero then
                enc = preprocessing.OneHotEncoder(dtype=np.float32, sparse=False,
                                                  categories="auto", drop=dropfirst)
                x = col_int.reshape(-1, 1)
                out = enc.fit_transform(x)
                
                # the column suffix is equivalent to the i-th label representation
                names = []
                for i in range(out.shape[1]):
                    a = enc.inverse_transform([[(1 if _ == i else 0) for _ in \
                                                range(out.shape[1])]])[0,0]
                    names.append(f"{name}{a}")
                    df[f"{name}{a}"] = out[:,i]
                    
                #df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
                #df.columns = df.columns.values.tolist()[:-out.shape[1]] + names
                if inplace:
                    df.drop(name, axis=1, inplace=True)
                for n in names:
                    self._categorical_columns_map[n] = name
                self._categorical_columns_encoder[name] = (enc, cat)
                if name in self.y_columns:
                    # redefine y_columns if they got renamed during set_categorical
                    if inplace:
                        del y_columns_new[y_columns_new.index(name)]
                    y_columns_new.extend(names)
            elif mode.lower() in self.CAT_MODES_LABEL:
                enc = preprocessing.LabelEncoder()
                df[name] = df[name].astype(np.str_)
                enc.fit(df[name])
                if inplace:
                    col_new = name
                else:
                    col_new = f"{name}0"
                df[col_new] = enc.transform(df[name]).astype(np.float32)
                self._categorical_columns_map[col_new] = name
                self._categorical_columns_encoder[name] = (enc, enc.classes_)
                if name in self.y_columns and not inplace:
                    # redefine y_columns if they got renamed during set_categorical
                    y_columns_new.append(col_new)

        self.y_columns = y_columns_new


    def inspect(self, df, mode="text"):
        """Inspect the structure of the DataFrame. Typical usage is first
        defining modifiers, applying them via :meth:`apply_to` and then
        inspecting the result given the modifier. Example:
        
        >>> df = pd.DataFrame([[1, 'text', 3.5]], columns=list('abc'))
        >>> m = DataFrameModifier()
        >>> m.cat_column('b', 'label')
        >>> m.apply_to(df)
        >>> print(m.inspect(df))
        col   name   dtype     categories
          0   a      int64   
          1   b      float64   'text': 0
          2   c      float64   
        
        :param pandas.DataFrame df: Target DataFrame to be inspected        
        :param str mode: Whether to return *str* (default) or an equivalent
            pd.DataFrame containing meta data on the structure
            if ``mode != 'text'``
        :return: Either str or pd.DataFrame depending on ``mode``
        """
        #df = self.df
        #m = self.modifier
        m = self
        cols_per_dtype = {}
        structure = pd.DataFrame()
        max_col_length = min(40, max([4] + [len(col) for col in df.columns]))
        cat_cols = {_['name']: _ for _ in m._categorical_columns}
        modes = {_['mode'] for _ in m._categorical_columns}
        
        out = f"col   {'name':{max_col_length}s}   dtype     "
        out += (f"categories\n{'':{max_col_length+19}s}(1-hot ordered by freq.)" \
                if m.CAT_MODE_1HOT in modes else "categories")
        out += "\n"
        
        for i, col in enumerate(df.columns):            
            if df[col].dtype not in cols_per_dtype:
                cols_per_dtype[df[col].dtype] = []
            cols_per_dtype[df[col].dtype].append((i, col))
        for dtype, cols in cols_per_dtype.items():
            for i, col in cols:
                cat, cat_str = "", ""
                if m and col in m._categorical_columns_map.keys():
                    base_col = m._categorical_columns_map[col]
                    cat_mode = cat_cols[base_col]['mode']
                    enc, categ = m.get_cat_encoder(base_col)
                    if cat_mode in m.CAT_MODES_LABEL:
                        cat = dict(zip(categ, enc.transform(categ)))
                        cat_str = str(cat).replace("{", "").replace("}", "")\
                            .replace(",", f"\n{'':{max_col_length+18}s}")
                    elif cat_mode in m.CAT_MODES_1HOT:
                        # Since column suffix is equivalent to the label code,
                        # we can just "invert" the number (NaN is all cols zero)
                        n = int(col[len(base_col):])
                        cat = categ.from_codes([n], categ.categories)[0]
                        cat_str = cat
                        if n == 1 and cat_cols[base_col]['dropfirst']:
                            # We virtually prepend the dropped n = 0 case
                            # of the most frequent label
                            col_ = f"{base_col}0"
                            col_str_ = str(col_).replace('\n', ' ').strip()
                            cat_ = categ.from_codes([0], categ.categories)[0]
                            cat_ = f"{cat_} ..."
                            structure = pd.concat([structure, pd.DataFrame(
                                    [[col_, dtype, cat_]], index=["..."])])
                            out += (f"...   {col_str_:{max_col_length}s}   "
                                   f"{df.dtypes[col]}   {cat_}\n")
                structure = pd.concat([structure, pd.DataFrame(
                        [[col, dtype, cat]], index=[i])])
                col_str = str(col).replace('\n', ' ').strip()
                out += f"{i:3d}   {col_str:{max_col_length}s}   {df.dtypes[col]}   {cat_str}\n"
        out = out[:-1]   # remove trailing newline
        structure.columns=['column', 'dtype', 'categorical']
        if mode == "text":
            return out 
        return structure
        

    def apply_to(self, df, y_columns=[]):
        """
        Applies modifications of self to DataFrame ``df`` inplace.
        Explicitly the method will:
            
        - Create new columns
        - Exclude / include defined columns        
        - Drop rows depending on the settings of ``drop_nan_y``
          and ``drop_nan_x``
        - Apply filters
        - Apply transformations
        - Transform categorical columns
        - Transform date, datetime and timedelta columns
        - Move ``y_columns`` at the end of the DataFrame
        
        No imputation (except for explicitly defined imputation for
        categorical columns via :meth:`cat_column`), normalization
        or pruning is performed up to here.
        These options will be provided by the :class:``Learner`` class
        
        :param pandas.DataFrame df: DataFrame that shall be modified
            / transformed
        :param y_columns: Column(s) intended as target variable(s)
            for a downstream learner
        :type y_columns: str or list
        :return: None
        """
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        self.y_columns = y_columns
        for col in self.y_columns:
            assert col in df.columns, 'column "{}" doesn\'t exist'.format(col)
        
        if id(df) in self._applied_to_df:
            warnings.warn("This modifier was already applied to this DataFrame")

        # new columns
        new_names = [_[0] for _ in self._calc_columns]
        for name, fn in self._calc_columns.items():
            #print("Adding new column '{}'".format(name))
            df[name] = fn(df)
        
        # exclude / include columns
        if self._include_columns:
            incl_cols = [col for col in df.columns if col in \
                         self._include_columns + self.y_columns + new_names]
            incl_cols_not_found = [col for col in self._include_columns \
                                   if col not in df.columns]
            df.drop(columns=[_ for _ in df.columns if _ not in incl_cols],
                    inplace=True)
            if incl_cols_not_found:
                warnings.warn("The following columns are to be included, but weren't "
                             f"present in the DataFrame: {incl_cols_not_found}")
        else:
            excl_cols = [col for col in self._exclude_columns if col in df.columns]
            excl_cols_not_found = [col for col in self._exclude_columns \
                                   if col not in df.columns]
            df.drop(excl_cols, axis=1, inplace=True)
            if excl_cols_not_found:
                warnings.warn("The following columns are to be excluded, but weren't "
                             f"present in the DataFrame: {excl_cols_not_found}")
        
        # drop rows with missing data
        if self._drop_nan_y and self.y_columns:
            # only keep rows where ALL y_columns are not NaN    
            #df = df.loc[np.all(pd.notna(df[self.y_columns]), axis=1), :]
            df.drop(df.loc[np.any(pd.isna(df[self.y_columns]), axis=1), :].index,
                           inplace=True)
        if self._drop_nan_x:
            # only keep rows where ALL x_columns are not NaN
            x_columns = [col for col in df.columns if col not in self.y_columns]
            #df = df.loc[np.all(pd.notna(df[x_columns]), axis=1), :]
            df.drop(df.loc[np.any(pd.isna(df[x_columns]), axis=1), :].index,
                           inplace=True)

        #apply filters *before* baseline_model, so outliers won't affect baseline
        self._apply_filters(df)
        
        # apply baseline model; may use new columns and always raw and untransformed values
        # also needs to be calculated before transforming categorical columns
        # since otherwise they're not easily accessible by name
        self._apply_transform_by_category_mean(df)
        
        # categorical columns. Also need to redefine x_columns
        self._set_categorical(df)
        self.x_columns = [col for col in df.columns if col not in self.y_columns]
        # self.y_columns is redefined within the _set_categorical method

        # convert any datetime and timedelta to float
        if self._time_to_float:
            for col in df.columns:
                try:
                    if df[col].dtype == 'datetime64[ns]':
                        df[col] = pd.to_numeric(df[col]).astype(float)
                        print("column '{}' was type 'datetime' and automatically converted to float".format(col))
                    if df[col].dtype == 'timedelta64[ns]':
                        df[col] = df[col].dt.total_seconds().astype(float)
                        print("column '{}' was type 'timedelta' and automatically converted to float".format(col))
                except (ValueError, TypeError) as exc:
                    raise type(exc)(exc.__str__() + "(when processing column '{}'. "
                       "Please convert it to caterogical column)".format(col))            
        
        # put y_columns at the end of the DataFrame by successively prepending
        # x_columns as index and resetting it again (can be done inplace)
        for name in self.x_columns[::-1]:
            df.set_index(name, inplace=True)
            df.reset_index(inplace=True)
        
        self._applied_to_df.append(id(df))

    
def __test():
    #df = pd.DataFrame(np.array([[1,2,3,4,5], [10,1,10,0,1], [1,1,0,0,0]]).T, columns=['a', 'b', 'c'])
    df = pd.DataFrame(list('bbaccc'), columns=['labelvar'])
    df['onehotvar'] = list('cddee') + [np.nan]
    df['abc'] = [1,5,6,7,8,9]
    m = DataFrameModifier()
    #m.new_column('d', lambda a,b,c: a*b+c, 'a', 'b', 'c')
    m.cat_column('labelvar', 'label')
    m.cat_column('onehotvar', '1-hot', impute='d', dropfirst=False)
    m.apply_to(df)
    
    print(df)
    print(m.inspect(df, "text"))
if __name__ == "__main__":
    __test()