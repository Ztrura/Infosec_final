import os
import csv

import numpy as np
import tensorflow as tf

from const import *
# from keras_models.multi_frame_cnn import build_multiframe_model
from multi_frame_cnn import build_multiframe_model
from logger import log
from multiframe_data_generator import MultiFrameDataGenerator
from utils import manual_split

from AccEvaluator import AccEvaluator
# split_filename = sys.argv[1]

split_filename = "exp1/new.csv"
if split_filename.count('/') != 1:
    print("Only one slash in filename")
    print("ERROR!")
    # exit()
# print("*")
split_csv_filename = f"{EXP_SPLIT_FOLDER}/{split_filename}"
if not os.path.isfile(split_csv_filename):
    print("Indicated csv file do not exists")
    print("ERROR!")
    # exit()
# print("*")

FRAMES_PER_SIDE = 5
BATCH_SIZE = 16
EPOCHS = 180

# Valid coverages 0 25 50 75 100
COVERAGE = 0
# default 250
REDUCED_SIZE = 250
DATA_AUGMENTATION = True
COMPUTE_TIMESTAMPS = False
USER_INDEPENDENT = True
SUBSAMPLING = False
K_FOLD = False
WHICH_SPLIT = 0  # between 0 and 3, only if k-fold true
PIN_LENGTH = 5
CAM_TYPE_1 = "center"
CAM_TYPE_2 = "left"
CAM_TYPE_3 = "right"
# print("*")
TESTING = False
SAVE_MODEL_NAME = "180_FINAL_MAXACC_LSTM_split"
if __name__ == "__main__":
    log.info("[+] Start")
    # How many cam to consider
    cam_list = [CAM_ID[CAM_TYPE_1], CAM_ID[CAM_TYPE_2], CAM_ID[CAM_TYPE_3]]
    cam_list = [CAM_ID[CAM_TYPE_1], ]
    log.info("[+] Loading CSV file")
    train_rows_all = []
    val_rows_all = []
    test_rows_all = []
    for cam_id in cam_list:
        with open(f"{OUT_FOLDER}/{cam_id}_input_data.csv", "r", newline="") as f:
            rows = np.array(list(csv.reader(f)))
        # print(rows)
        with open(split_csv_filename, "r", newline="") as f:
            split_list = np.array(list(csv.reader(f)))
        train_names = list(split_list[:, 0])
        train_names = set(list(filter(None, train_names)))
        val_names = split_list[:, 1]
        val_names = set(list(filter(None, val_names)))
        if split_list.shape[1] == 3:
            test_names = split_list[:, 2]
            test_names = set(list(filter(None, test_names)))
        else:
            test_names = set([])
        train_rows_curr, val_rows_curr, test_rows_curr = manual_split(rows, train_names, val_names, test_names)
        train_rows_all.append(train_rows_curr)
        val_rows_all.append(val_rows_curr)
        test_rows_all.append(test_rows_curr)
    train_lenghts = [len(item) for item in train_rows_all]
    val_lenghts = [len(item) for item in val_rows_all]
    test_lenghts = [len(item) for item in test_rows_all]

    train_rows = np.concatenate(train_rows_all)
    val_rows = np.concatenate(val_rows_all)
    test_rows = np.concatenate(test_rows_all)
    
    log.info(
        f"[+] There are {len(train_rows)} training samples, {len(val_rows)} validation samples, and {len(test_rows)} test samples.")

    log.info(f"[+] Frame padding is set to {FRAMES_PER_SIDE} frame for each side. (If multiframe)")
    log.info(f"[+] Samples will therefore have {FRAMES_PER_SIDE * 2 + 1} frames each. (If multiframe)")
    log.info(f"[+] The batch size is set to: {BATCH_SIZE}.")
    log.info(f"[+] There will be {EPOCHS} training epochs.")
    log.info(f"[+] Data augmentation for the training set is set to: {DATA_AUGMENTATION}.")
    log.info(f"[+] User independed split is set to: {USER_INDEPENDENT}.")
    log.info(f"[+] WHICH_SPLIT is set to: {WHICH_SPLIT}.")
    # print("*")
    train_data_generator = MultiFrameDataGenerator(cam_id=cam_list, rows=train_rows, rows_lenght=train_lenghts,
                                                   batch_size=BATCH_SIZE,
                                                   n_classes=10,
                                                   shuffle=False,
                                                   source="video", coverage=COVERAGE, reduced_size=REDUCED_SIZE,
                                                   augment=DATA_AUGMENTATION,
                                                   frames_per_side=FRAMES_PER_SIDE,
                                                   subsampling=SUBSAMPLING,
                                                   compute_timestamps=COMPUTE_TIMESTAMPS)

    validation_data_generator = MultiFrameDataGenerator(cam_id=cam_list, rows=val_rows, rows_lenght=val_lenghts,
                                                        batch_size=BATCH_SIZE,
                                                        n_classes=10,
                                                        shuffle=True,
                                                        source="video", coverage=COVERAGE,
                                                        reduced_size=REDUCED_SIZE, augment=False,
                                                        frames_per_side=FRAMES_PER_SIDE,
                                                        subsampling=SUBSAMPLING,
                                                        compute_timestamps=COMPUTE_TIMESTAMPS)

    # shuffle and batch_size have been changing just for testing purposes
    test_data_generator = MultiFrameDataGenerator(cam_id=cam_list, rows=test_rows, rows_lenght=test_lenghts,
                                                  batch_size=1,
                                                  n_classes=10,
                                                  shuffle=False,
                                                  source="video", coverage=COVERAGE, reduced_size=REDUCED_SIZE,
                                                  augment=False,
                                                  frames_per_side=FRAMES_PER_SIDE,
                                                  subsampling=SUBSAMPLING,
                                                  compute_timestamps=COMPUTE_TIMESTAMPS)

    # print("**")
    if TESTING:
        # print("***")
        exp_folder_name = split_filename.split("/")[0]
        log.info(f"[+] Loading best model for testing purposes.")
        testing_model = tf.keras.models.load_model(
            filepath=f"/media/cecco/Data/VIDEOPIN/ModelsFinal/{exp_folder_name}/{SAVE_MODEL_NAME}")
        log.info(f"[+] Model loaded.")
        log.info(testing_model.summary())

        X = []
        y = []

        for i in range(len(test_rows)):
            X.append(test_data_generator[i][0])
            y.append(int(np.argmax(test_data_generator[i][1][0], axis=-1)))

        log.info("[+] X and y populated")
        evaluator = AccEvaluator(PIN_LENGTH)
        predictions = evaluator.get_predictions(testing_model, X, y, exp_folder_name, PIN_LENGTH)
        key_top_n_acc, pin_top_n_acc, _, _ = evaluator.print_acc_s(predictions, y, True)
        log.info(f"[+] Key testing results: {key_top_n_acc}")
        log.info(f"[+] Pin testing results: {pin_top_n_acc}")

    else:
        # print("***")
        # print(split_filename.split("/")[0])
        # print("***")
        exp_folder_name = split_filename.split("/")[0]
        exp_folder_name_full = f"/media/cecco/Data/VIDEOPIN/max_acc_models/{exp_folder_name}"
        if not os.path.isdir(exp_folder_name_full):
            os.makedirs(exp_folder_name_full)
        model_filename = f"{exp_folder_name_full}/{SAVE_MODEL_NAME}"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_filename,
            save_weights_only=False,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )
        print("0")
        log.info(f"[+] Shape of input data is: {train_data_generator[0][0][0][0].shape}")
        print("1")
        cust_cnn = build_multiframe_model(input_shape=train_data_generator[0][0][0][0].shape,
                                          internal_model_name="center")
        # print("***")
        log.info(f"[+] Compiling network.")
        log.info("[+] Optimizer: SGD(), Loss: CategoricalCrossentropy(), Metrics: [accuracy, top-2, top-3]")

        cust_cnn.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy",
                     tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
                     ]
        )
        log.info(f"[+] Network has been compiled.")

        log.info(cust_cnn.summary())

        log.info("[+] Training is starting.")
        # removing validation from split 
        print("########")
        # print(len(train_data_generator))
        tmplen=len(train_data_generator)
        print("AAAAAAA")
        
        data_list = []
        for i in range(BATCH_SIZE):
            print("kk"+str(i))
            batch_data = train_data_generator[i]
            data_list.append(batch_data)
        train_data = np.concatenate(data_list, axis=0, dtype=object)
        train_data_generator = train_data
        # data_list = []
        # for data in train_data_generator:
        #     print("k")
        #     data_list.append(data)
        #     # print(data_list)
        # print("1111")
        # train_data_generator = np.array(data_list)

        # train_data_generator=np.array(train_data_generator)
        # train_data_generator=train_data_generator.to_numpy()
        print("*******")
        print(type(train_data))
        print(type(train_data_generator))
        print("////////////////")
        # type(validation_data_generator)
        print("afafdsaf")
        cust_cnn.fit(train_data_generator, epochs=EPOCHS, steps_per_epoch=len(train_data_generator),
                     verbose=2,  callbacks=[model_checkpoint_callback])

    log.info("[+] Over.")

