def custom_cross_validation(training_data, n_splits=5):
    '''creates n_splits sets of training and validation folds

    Args:
      training_data: the dataframe of features and target to be divided into folds
      n_splits: the number of sets of folds to be created

    Returns:
      A tuple of lists, where the first index is a list of the training folds, 
      and the second the corresponding validation fold
    '''
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    training_folds = []
    validation_folds = []

    for train_index, val_index in kf.split(training_data):

        train_fold = training_data.iloc[train_index].copy()
        val_fold = training_data.iloc[val_index].copy()


        # Compute city means on the training folds

        city_means = train_fold.groupby('city_encoded')['price'].mean().rename('city_mean_price')


        # Join these values to both training and validation folds

        train_fold = train_fold.merge(city_means, on='city_encoded', how='left')
        val_fold = val_fold.merge(city_means, on='city_encoded', how='left')


        # Fill NaN values that might appear due to missing cities in validation fold

        train_fold['city_mean_price'] = train_fold['city_mean_price'].fillna(
            train_fold['price'].mean())
        val_fold['city_mean_price'] = val_fold['city_mean_price'].fillna(
            train_fold['price'].mean())

        training_folds.append(train_fold)
        validation_folds.append(val_fold)

    return training_folds, validation_folds


def hyperparameter_search(training_folds, validation_folds, param_grid, model_name):
    '''outputs the best combination of hyperparameter settings in the param grid, 
    given the training and validation folds

    Args:
      training_folds: the list of training fold dataframes
      validation_folds: the list of validation fold dataframes
      param_grid: the dictionary of possible hyperparameter values for the chosen model

    Returns:
      A tuple including a list of the best hyperparameter settings and the best score using
      mean squared error.
    '''
    import numpy as np
    from sklearn.model_selection import ParameterGrid
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    best_params = None
    best_score = float('inf')

    for params in ParameterGrid(param_grid):

        mse_scores = []

        for train_fold, val_fold in zip(training_folds, validation_folds): # runs through 
            if model_name == 'decision_tree':
              model = DecisionTreeRegressor(**params, random_state=42)
            if model_name == 'random_forest':
              model = RandomForestRegressor(**params, random_state=42)

            X_train = train_fold.drop(columns=['price'])
            y_train = train_fold['price']
            X_val = val_fold.drop(columns=['price'])
            y_val = val_fold['price']

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_scores.append(mse)

        avg_score = np.mean(mse_scores)

        if avg_score < best_score: # checks if better than best score
            best_score = avg_score
            best_params = params

    return best_params, best_score