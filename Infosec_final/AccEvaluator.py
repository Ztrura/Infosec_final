import itertools
import multiprocessing

import numpy as np
import pandas as pd

from const import project_folder
from logger import log

from sklearn.metrics import classification_report, confusion_matrix


class AccEvaluator:
    PIN_LENGTH = None

    def __init__(self, PIN_LENGTH):
        AccEvaluator.PIN_LENGTH = PIN_LENGTH

    @staticmethod
    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return x, y

    @staticmethod
    def get_attempts(func_args):
        pin_n, predictions, y_test = func_args
        # NB this 5 must not be changed. It must be a constant, since it depends on our dataset (which has 5-digit pins)
        local_pin = pin_n * 5
        sub_matrix = predictions[local_pin: local_pin + AccEvaluator.PIN_LENGTH, 0: 10]
        all_key_prob = pd.DataFrame(itertools.product(*sub_matrix))
        all_pin_prob = all_key_prob.prod(axis=1)
        sorted_prob = all_pin_prob.sort_values(ascending=False)
        num_test = ""
        for num in range(AccEvaluator.PIN_LENGTH):
            num_test += str(y_test[local_pin + num])
        num_test = int(num_test)
        index_pos = sorted_prob.index.get_loc(num_test)
        return index_pos, np.sum(sorted_prob[:3])

    @staticmethod
    def _cfm(clf, X_test, y_test):
        predictions = np.empty(len(X_test))
        for i, sample in enumerate(X_test):
            predictions[i] = int(np.argmax(clf.predict(sample)[0], axis=-1))

        np.savetxt(f"{project_folder}/y_pred.txt", predictions)
        np.savetxt(f"{project_folder}/y_test.txt", y_test)

        log.info('Confusion Matrix')
        log.info(confusion_matrix(y_test, predictions, normalize='true'))
        log.info('Classification Report')
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        log.info(classification_report(y_test, predictions, target_names=classes))

    @staticmethod
    def print_cfm(y_pred, y_test):
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        log.info('Confusion Matrix')
        log.info(confusion_matrix(y_test, y_pred, normalize='true'))
        log.info('Classification Report')
        log.info(classification_report(y_test, y_pred, target_names=classes))

    @staticmethod
    def get_predictions(clf, X_test, y_test, name_dataset, PIN_LENGTH):
        predictions = np.empty((len(X_test), 10))
        for i, sample in enumerate(X_test):
            predictions[i] = clf.predict(sample)[0]
        np.savetxt('results/predicted_{}_{}PIN.csv'.format(name_dataset, PIN_LENGTH), predictions, delimiter=',')
        np.savetxt('results/y_test_{}_{}PIN.csv'.format(name_dataset, PIN_LENGTH), np.array(y_test), fmt="%d",
                   delimiter=',')
        return predictions

    @staticmethod
    def in_best_x_s(predictions, y_test, howmany=3):
        counter = np.zeros(howmany, dtype='float')

        tot_cycle = (len(predictions) // 5)

        range_used = range(0, tot_cycle)
        len_range = len(range_used)
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results_pool = pool.map(AccEvaluator.get_attempts,
                                zip(range_used, [predictions] * len_range, [y_test] * len_range))
        attempts, top3_confidences = zip(*results_pool)
        pool.close()

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for idx, prediction in enumerate(predictions):
            lll = [(classes[i], val) for i, val in enumerate(prediction) if val != 0.0]
            values = [x[0] for x in sorted(lll, key=lambda x: x[1], reverse=True)]
            target_class = y_test[idx]
            if target_class in values:
                for x in range(values.index(target_class), howmany):
                    counter[x] += 1

        return counter / len(y_test), attempts, top3_confidences

    @staticmethod
    def print_acc_s(predictions, y_test_final, savetxt=False):
        key_top_n_acc, attempts, top3_confidences = AccEvaluator.in_best_x_s(predictions, y_test_final)
        x, y = AccEvaluator.ecdf(attempts)
        if savetxt:
            np.savetxt(f"{project_folder}/ecdf_x.txt", x)
            np.savetxt(f"{project_folder}/ecdf_y.txt", y)
        x_search = list(map(int, x))
        idx1 = np.searchsorted(x_search, 1)
        idx2 = np.searchsorted(x_search, 2)
        idx3 = np.searchsorted(x_search, 3)
        if idx1 == 0:
            idx1_result = 0
        else:
            idx1_result = y[idx1 - 1]
        if idx2 == 0:
            idx2_result = 0
        else:
            idx2_result = y[idx2 - 1]
        if idx3 == 0:
            idx3_result = 0
        else:
            idx3_result = y[idx3 - 1]
        pin_top_n_acc = [idx1_result, idx2_result, idx3_result]
        return key_top_n_acc, pin_top_n_acc, attempts, top3_confidences
