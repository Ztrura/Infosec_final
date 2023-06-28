import gc
import const
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import data_utils
import utils


class MultiFrameDataGenerator(data_utils.Sequence):

    def __init__(self, cam_id, rows, rows_lenght, batch_size=32, n_classes=10, shuffle=True, source="disk", coverage=0,
                 reduced_size=250, augment=False, frames_per_side: int = 5, compute_timestamps: bool = False,
                 subsampling: bool = False):
        self.coverage = coverage
        self.reduced_size = reduced_size
        self.frame_cache = {}
        self.cam_id = cam_id
        self.batch_size = batch_size
        self.compute_timestamps = compute_timestamps
        self.rows = rows
        self.rows_lenght = rows_lenght
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.source = source
        self.frames_per_side = frames_per_side
        self.max_seq_len = frames_per_side * 2 + 1
        self.augment = augment
        self.subsampling = subsampling
        self.on_epoch_end()

    def __len__(self):
        # batches per epoch
        return int(np.floor(len(self.rows) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # extract the videos
        rows_batch = [self.rows[k] for k in indexes]

        X, y = self.__data_generation(rows_batch, indexes)
        # print("asdf")
        # print(type(X))
        # print(X)
        # X=np.array(X, dtype=object)
        # print("asdf")

        return X, y

    # called once at the beginning, and at the end of each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.rows))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        gc.collect()

    def load_frames(self, video, frame_indexes, username, directory, cam_id):
        frames = []

        # these variables can be used to simulate errors in the key logger (e.g., using sigma 3.33, 1.66, 1, ecc)
        mu, sigma = 0, 0

        frame_errors = np.random.normal(mu, sigma, len(frame_indexes))
        for frame_no, frame_error in zip(frame_indexes, frame_errors):
            frame_no = frame_no + int(np.round(frame_error))
            # [CASE MULTI-FRAME FROM DISK]
            if self.source == "disk":
                frame = cv2.imread(f"{const.OUT_FOLDER}/{cam_id}/{directory}/{frame_no}.jpg")
            else:
                cap = cv2.VideoCapture(
                    f"{const.IN_FOLDER}/{username}/{directory}/webcam/webcam_{cam_id}.avi")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                _, frame = cap.read()

            if username[0] == "2":
                pin_pad = 2
            else:
                pin_pad = 1
            # this is basic pre-processing, it holds for all frames
            frame = utils.basic_preproc_pipeline(input_frame=frame, cam_id=cam_id, reduced_size=self.reduced_size,
                                                 coverage=self.coverage, pin_pad=pin_pad)
            frames.append(frame)

        # add to cache
        if video not in self.frame_cache:
            self.frame_cache[video] = {}
        self.frame_cache[video][frame_indexes] = frames
        return frames

    def __data_generation(self, rows_batch, indexes):

        # warn: (250, 250) is hardcoded inside basic_preproc_pipeline
        #   remember to change that too if you want to use a different shape
        image_dims = (self.max_seq_len, self.reduced_size, self.reduced_size)
        X = np.empty((self.batch_size, *image_dims, 1))

        Z = np.empty((self.batch_size, *(2,)), dtype=int)

        y = np.empty(self.batch_size, dtype=int)

        for i, (row, index_row) in enumerate(zip(rows_batch, indexes)):
            target_frame_no = int(row[0])
            starting_frame_no = int(row[1])
            ending_frame_no = int(row[2])
            timing_1 = float(row[3]) if self.compute_timestamps else 0
            timing_2 = float(row[4]) if self.compute_timestamps else 0
            edge_case = int(row[5])
            directory = row[6]
            username = row[7]
            key = row[-1]

            if not self.subsampling:
                # i only want self.frames_per_side frames before and after the target key.
                # if we are in one of the edge cases, only one side will be filled by this function
                #   the other side will be filled by the function extend_short_sequences() called later
                if edge_case == 1:
                    # first key of a pin, starting_frame_no == target_frame_no
                    assert starting_frame_no == target_frame_no, \
                        f"Edge case {edge_case} detected, but starting_frame_no is {starting_frame_no}," \
                        f"and target_frame_no is {target_frame_no}."
                    # i only want to take the next self.frames_per_side frames, if available
                    # if not, then ending_frame_no is already set correctly
                    if ending_frame_no - target_frame_no > self.frames_per_side:
                        ending_frame_no = target_frame_no + self.frames_per_side

                elif edge_case == 2:
                    # last key of a pin, ending_frame_no == target_frame_no
                    assert ending_frame_no == target_frame_no

                    # I only want to take the previous max_seq_len frames, if available
                    # if not, then starting_frame_no is already set correctly
                    if target_frame_no - starting_frame_no > self.frames_per_side:
                        starting_frame_no = target_frame_no - self.frames_per_side
                else:
                    # no edge case, i only want to crop the number of frames
                    # however i need to check if there are at least max_seq_len before and after the target frame
                    if ending_frame_no - target_frame_no > self.frames_per_side:
                        ending_frame_no = target_frame_no + self.frames_per_side

                    if target_frame_no - starting_frame_no > self.frames_per_side:
                        starting_frame_no = target_frame_no - self.frames_per_side

                frame_indexes = range(starting_frame_no, ending_frame_no + 1)

            else:
                # we are subsampling
                pre = np.round(np.linspace(starting_frame_no, target_frame_no, self.frames_per_side + 1)).astype(int)
                post = np.round(np.linspace(target_frame_no, ending_frame_no, self.frames_per_side + 1)).astype(int)
                frame_indexes = list(np.unique(np.concatenate((pre, post))))

            if index_row < self.rows_lenght[0]:
                cam_id = self.cam_id[0]
            else:
                if index_row < self.rows_lenght[0] + self.rows_lenght[1]:
                    cam_id = self.cam_id[1]
                else:
                    cam_id = self.cam_id[2]
            video = f"{const.IN_FOLDER}/{username}/{directory}/webcam/webcam_{cam_id}.avi"
            if video not in self.frame_cache:
                frames = self.load_frames(video, frame_indexes, username, directory, cam_id)
            else:
                if frame_indexes not in self.frame_cache[video]:
                    frames = self.load_frames(video, frame_indexes, username, directory, cam_id)
                else:
                    frames = self.frame_cache[video][frame_indexes]

            if self.augment:
                # i want to apply the same transformation to all the frames in this sequence
                augmentation_seed = np.random.randint(1, 999999)
                frames = self.__data_augmentation(sequence=frames, seed=augmentation_seed)

            if len(frames) < self.max_seq_len:
                frames = utils.extend_short_sequence(sequence=frames, edge_case=edge_case,
                                                     max_seq_length=self.max_seq_len)

            # print(len(range(starting_frame_no, ending_frame_no + 1)), self.max_seq_len, len(frames))
            # print(frames)
            assert len(frames) == self.max_seq_len

            X[i] = frames
            Z[i] = (timing_1, timing_2)
            y[i] = key
            print("generating...")
            # print(type(X[i]))
            # print(type(y[i]))
            # print(type(Z[i]))
        return [X, Z], to_categorical(y, num_classes=self.n_classes)

    @staticmethod
    def __data_augmentation(sequence, seed):
        augmented_frames = []

        augmented_datagen = ImageDataGenerator(
            rotation_range=7,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1)

        for frame in sequence:
            frame = np.expand_dims(frame, 0)
            it = augmented_datagen.flow(x=frame, batch_size=1, seed=seed)
            augmented_frame = it.next()[0]
            augmented_frames.append(augmented_frame)

        return augmented_frames
