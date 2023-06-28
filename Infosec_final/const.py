import sys

LEFT_CAM_ID = "0F259C0F"
CENTER_CAM_ID = "1AE49C0F"
RIGHT_CAM_ID = "A01B64BF"
THERMAL_CAM_ID = None

HUMAN_STR = {
    LEFT_CAM_ID: "left",
    CENTER_CAM_ID: "center",
    RIGHT_CAM_ID: "right",
    THERMAL_CAM_ID: "thermal"
}

CAM_ID = {
    "left": LEFT_CAM_ID,
    "center": CENTER_CAM_ID,
    "right": RIGHT_CAM_ID,
    "thermal": THERMAL_CAM_ID
}

CROP_ROI = {

    CENTER_CAM_ID: {

        "x1": 360,
        "x2": 960,
        "y1": 60,
        "y2": 560
    },

    RIGHT_CAM_ID: {

        "x1": 600,
        "x2": 1050,
        "y1": 200,
        "y2": 800
    },

    LEFT_CAM_ID: {

        "x1": 250,
        "x2": 750,
        "y1": 100,
        "y2": 650
    }
}

# These are folders names corresponding to anonymized participants (first dataset)
# BLACKLIST1 = {
#     "EXP1_2",
#     "EXP1_8",
#     "EXP1_35",
#     "EXP1_34",
#     "EXP1_11",
#     "EXP1_6",
#     "EXP1_40",
#     "EXP1_15",
#     "EXP1_29",
#     "EXP1_5",
#     "EXP1_19",
#     "EXP1_25",
#     "EXP1_39",
#     "EXP1_14"
# }

BLACKLIST1 = [
    "EXP1_" + str(i) for i in range(1, 33)
]

# These are folders names corresponding to anonymized participants (second dataset)
# BLACKLIST2 = {
#     "EXP2_12",
#     "EXP2_8"
# }

BLACKLIST2 = {
    "EXP2_" + str(i) for i in range(1, 13)
}

# See the README
project_folder = ""
IN_FOLDER = "dataset2"
OUT_FOLDER = "feedback_sound_recognition/dataset2"
EXP_SPLIT_FOLDER = "exp_split"

