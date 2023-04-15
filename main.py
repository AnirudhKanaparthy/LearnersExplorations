import algorithms.linear_regression as linreg
import algorithms.gda as gda
import algorithms.logistic_regression as logreg

import utils.dataset_loaders as loaders
import utils.model_testers as testers

import matplotlib.pyplot as plt
import seaborn as sns


def train_split(all_X, all_y, train_ratio = 0.8):
    trainX, testX = all_X[ : int(all_X.shape[0] * train_ratio), :], all_X[int(all_X.shape[0] * train_ratio) :, :]
    trainY, testY = all_y[ : int(all_y.shape[0] * train_ratio), :], all_y[int(all_y.shape[0] * train_ratio) :, :]

    return trainX, trainY, testX, testY


def run_gda():
    data_path = './datasets/Binary Classification/stan_cs229.csv'
    all_X, all_y = loaders.load_stan_cs229(data_path)
    trainX, trainY, testX, testY = train_split(all_X, all_y)

    gda_stan = gda.GaussianDiscriminantAnalysis(trainX, trainY)
    gda_stan.fit()

    gda_stan.print_params()
    print(f'error: {testers.classification_test(gda_stan, testX, testY)}')


def run_linreg():
    data_path = './datasets/Linear Regression/linreg_sample.csv'

    all_X, all_y = loaders.load_linreg_sample(data_path)
    trainX, trainY, testX, testY = train_split(all_X, all_y)

    lr_norm = linreg.LinearRegression(trainX, trainY)
    lr_newt = linreg.LinearRegression(trainX, trainY)

    initial_error = lr_norm.mse_test(testX, testY)
    print(f'Initial: {initial_error}')

    lr_norm.fit_svd()
    lr_newt.fit_newt()

    print(f'Normal Gradient Ascent:\n{lr_norm.theta}\n')
    print(f'Newton\'s Method:\n{lr_newt.theta}\n')

    total_error = lr_norm.mse_test(testX, testY)
    total_error2 = lr_newt.mse_test(testX, testY)

    print(f'Normal:\n Error: {total_error}\nCorrect: {100 - total_error}')
    print(f'Newton\'s Method:\n Error: {total_error2}\nCorrect: {100 - total_error2}')


def run_logreg():
    data_path = './datasets/Binary Classification/abalone.csv'
    all_X, all_y = loaders.load_abalone(data_path)
    trainX, trainY, testX, testY = train_split(all_X, all_y)

    lr_norm = logreg.LogisticRegression(trainX, trainY, 0.1)
    lr_newt = logreg.LogisticRegression(trainX, trainY)

    error_norm = lr_norm.fit()
    error_newt = lr_newt.fit_newt()

    print(f'Normal Gradient Ascent:\n{lr_norm.theta}\n')
    print(f'Newton\'s Method:\n{lr_newt.theta}\n')

    total_error = testers.classification_test(lr_norm, testX, testY)
    total_error2 = testers.classification_test(lr_newt, testX, testY)

    print(f'Normal:\n Error: {total_error}\nCorrect: {100 - total_error}')
    print(f'Newton\'s Method:\n Error: {total_error2}\nCorrect: {100 - total_error2}')

    sns.lineplot(x = list(range(len(error_norm))) , y = error_norm)
    sns.lineplot(x = list(range(len(error_newt))) , y = error_newt)

    plt.show()


def main():
    run_linreg()

if __name__ == '__main__':
    main()