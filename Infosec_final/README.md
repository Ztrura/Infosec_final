Launch this program passing as parameter the relative path of a split file. The first folder is "exp_split", the second folder name corresponds to an arbitrary experiment name.
The split file is a CSV file (with extension .csv) that contains three columns (containing the participants to be included in the training, validation, and testing, respectively).
Our dataset contains N participants. For each participant, the dataset has a folder, so there must be a list of folders names in each CSV file column. See an example file in exp_split.

The file const.py contains many variables needed to run the code

    BLACKLIST1 -> list of strings containing the folder name of the people to exclude from the validation and testing sets (from the first data collection)
    
    BLACKLIST2 -> list of strings containing the folder name of the people to exclude from the validation and testing sets (from the second data collection)
    
    project_folder -> absolute path containing the folder "experiments"
    
    IN_FOLDER -> absolute path containing the dataset
    
    OUT_FOLDER -> absolute path from where the program can read the keypress details. The feedback_sound_recognition folder contains the keypress details extracted using the feedback sound. The path of the folder dataset1 or dataset2 (only one at a time) must be inserted into the variable OUT_FOLDER.
    
    EXP_SPLIT_FOLDER -> absolute path of the "exp_split" folder (included the name "exp_split" at the end of the path)
