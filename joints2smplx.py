import numpy as np
import os
import joblib
from tqdm import tqdm
import glob

data_dir = "./joints2smplx"
save_dir = "./predicts_hml"
samples_list = os.listdir(data_dir)

for sample_idx in tqdm(range(len(samples_list))):
    sample = samples_list[sample_idx]
    sample_path = os.path.join(data_dir, sample)
    files_params = sorted(glob.glob(f"{sample_path}/0*.pkl"))

    frame_info = []

    for frame_idx in range(len(files_params)):
        raw_data = joblib.load(files_params[frame_idx])
        frame_cat = np.concatenate([raw_data['body_pose'].reshape(-1)[3:], raw_data['left_hand_pose'].reshape(-1), raw_data['right_hand_pose'].reshape(-1)])


        frame_info.append(np.expand_dims(frame_cat, 0))

    frame_info = np.concatenate(frame_info, axis = 0)
    np.save(os.path.join(save_dir, sample[:-4]), frame_info)
    