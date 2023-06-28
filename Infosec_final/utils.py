import random
import string

import cv2
import numpy as np
from const import *
# from sys import exit
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from tensorflow.keras.utils import to_categorical


# not used anymore
def random_string(length=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


# this function extends a single sequence
def extend_short_sequence(sequence, edge_case, max_seq_length):
    black_frame = np.zeros(shape=sequence[0].shape)

    if edge_case == 1:
        diff = max_seq_length - len(sequence)
        for _ in range(diff):
            # filling first half with black frames
            sequence.insert(0, black_frame)
    else:
        diff = max_seq_length - len(sequence)
        for _ in range(diff):
            # filling second half with black frames
            sequence.append(black_frame)

    return sequence


# extends all the sequences in the @sequences param.
# this function is used in the load_dataset function (i.e. the old way to load data)
def extend_short_sequences(sequences, edge_cases, max_seq_length):
    """This functions extends shorter sequences by adding black frames.
    Black frames are added at the beginning of the sequence (edge case 1) or at the end (edge cases 0 and 2)

    :param sequences: input examples
    :param edge_cases: either, 0, 1 or 2
    :param max_seq_length: the length of the longest sequence of frames
    :return:
    """

    # create placeholder frame
    black_frame = np.zeros(shape=sequences[0][0].shape)

    for i, seq in enumerate(sequences):
        edge_case = edge_cases[i]

        if len(seq) < max_seq_length:
            # we need to extend the number of frames to equal `max_num_frames`
            # edge_case 1 -> missing the first half
            # edge_case 2 -> missing the second half
            # edge_case 0 -> has both parts but it's just shorter, i'll handle it like edge_case 2

            if edge_case == 1:
                diff = max_seq_length - len(seq)
                for _ in range(diff):
                    # filling first half with black frames
                    seq.insert(0, black_frame)
            else:
                diff = max_seq_length - len(seq)
                for _ in range(diff):
                    # filling second half with black frames
                    seq.append(black_frame)

    sequences = np.asarray(sequences)
    return sequences


def basic_preproc_pipeline(input_frame, cam_id, reduced_size, coverage, pin_pad):
    """Expected BGR input (standard for cv2), not RGB (standard for PIL images)
    cam_id can be used for custom preprocessing based on the camera, useful when cropping.

    BGR2GRAY -> Normalize -> Crop -> Resize
    """
    output_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    output_frame = cv2.normalize(output_frame, None, 0, 255, cv2.NORM_MINMAX)
    output_frame = output_frame / 255
    # cropping, getting ROI coordinates based on cam_id
    roi = CROP_ROI[cam_id]
    output_frame = output_frame[roi["y1"]: roi["y2"], roi["x1"]: roi["x2"]]
    output_frame = cv2.resize(output_frame, (reduced_size, reduced_size), interpolation=cv2.INTER_CUBIC)

    if pin_pad == 1:
        if coverage == 25:
            output_frame[50:(reduced_size // 100) * 48, 50:200] = 0
        if coverage == 50:
            output_frame[50:(reduced_size // 100) * 61, 50:200] = 0
        if coverage == 75:
            output_frame[50:(reduced_size // 100) * 76, 50:200] = 0
        if coverage == 100:
            output_frame[50:(reduced_size // 100) * 95, 50:200] = 0
    if pin_pad == 2:
        if coverage == 25:
            output_frame[50:(reduced_size // 100) * 59, 50:200] = 0
        if coverage == 50:
            output_frame[50:(reduced_size // 100) * 72, 50:200] = 0
        if coverage == 75:
            output_frame[50:(reduced_size // 100) * 86, 50:200] = 0
        if coverage == 100:
            output_frame[50:(reduced_size // 100) * 104, 50:200] = 0

    # cv2.imshow("post resize", output_frame)
    output_frame = np.expand_dims(output_frame, axis=-1)
    # cv2.waitKey(0)

    return output_frame


def encode_labels(label_list: list, num_classes: int = 10):
    # removed sklearn label encoder
    # one-hot encoding
    y = to_categorical(label_list, num_classes=num_classes)
    return y


def custom_split_Z(X, Z, y, test_size: float = 0.2, random_state: int = 1337):
    """This is not used anymore, ignore
    """
    assert 0 <= test_size <= 1, "Test size has to be in [0; 1]"
    # np.random.seed(random_state)

    arr_rand = np.random.rand(X.shape[0])
    indexes = arr_rand < np.percentile(arr_rand, 100 - test_size * 100)

    X_train = X[indexes]
    Z_train = Z[indexes]
    y_train = y[indexes]

    X_test = X[~indexes]
    Z_test = Z[~indexes]
    y_test = y[~indexes]

    return X_train, Z_train, y_train, X_test, Z_test, y_test


def get_max_seq_length(rows):
    max_seq_length = 0

    for row in rows:
        # headers: [target_frame, starting_frame, ending_frame, interkeystroke_1, interkeystroke_2, edge_case, directory, key]
        starting_frame_no = int(row[1])
        ending_frame_no = int(row[2])

        # number of frames of the current frame sequence
        # save the length of the longest frame sequence so we can make them all equal
        current_seq_length = ending_frame_no - starting_frame_no + 1
        if current_seq_length > max_seq_length:
            max_seq_length = current_seq_length

    return max_seq_length


def manual_split(rows, train_names, val_names, test_names):
    if len(set.intersection(train_names, val_names)) != 0:
        print("Intersection not null train-val")
        exit()

    if len(set.intersection(train_names, test_names)) != 0:
        print("Intersection not null train-test")
        exit()

    if len(set.intersection(val_names, test_names)) != 0:
        print("Intersection not null val-test")
        exit()
        
    train_indexes = []
    val_indexes = []
    test_indexes = []

    for i, row in enumerate(rows):
        username = row[7]
        if username in train_names:
            train_indexes.append(i)
        elif username in val_names:
            val_indexes.append(i)
            if username in BLACKLIST1 or username in BLACKLIST2:
                print("User in blacklist put in val")
                exit()
        elif username in test_names:
            test_indexes.append(i)
            if username in BLACKLIST1 or username in BLACKLIST2:
                print("User in blacklist put in test")
                exit()
        else:
            print("WARNING USER NOT USED:", username)

    train = rows[train_indexes]
    val = rows[val_indexes]
    test = rows[test_indexes]
    # print("train:")
    # print(train)
    # print("val:")
    # print(val)
    # print("test:")
    # print(test)

    return train, val, test


def custom_split(rows, test_size: float = 0.3, random_state: int = None, user_independent: bool = False, k_fold=False,
                 which_split=0):
    assert 0 <= test_size <= 1, "[ASSERTION ERROR] -> Test size has to be in [0; 1]"

    if random_state is not None:
        np.random.seed(random_state)

    if user_independent:
        groups = np.empty((len(rows)), dtype=object)
        for i, row in enumerate(rows):
            username = row[7]
            groups[i] = username

        tot_candidates = len(np.unique(groups))
        train_candidates = round(tot_candidates * (1 - test_size))

        # we do not need to set random_state for gss as we have already fixed
        # the seed in the beginning.
        if k_fold:

            group_kfold = GroupKFold(n_splits=4)
            splits_idx = []
            for train_idx, test_idx in group_kfold.split(rows, None, groups):
                splits_idx.append((train_idx, test_idx))

            train_indexes, test_indexes = splits_idx[which_split]

        else:

            gss = GroupShuffleSplit(n_splits=1, train_size=train_candidates)
            train_indexes, test_indexes = next(gss.split(rows, None, groups))
    else:

        arr_rand = np.random.rand(rows.shape[0])
        train_indexes = arr_rand < np.percentile(arr_rand, 100 - test_size * 100)
        test_indexes = ~train_indexes

    train = rows[train_indexes]
    test = rows[test_indexes]

    return train, test
