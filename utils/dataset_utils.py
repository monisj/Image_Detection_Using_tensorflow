import os

import pandas as pd
from tqdm import tqdm

from models.sign_model import SignModel
from models.face_model import FaceModel
from utils.landmark_utils import save_landmarks_from_video,save_landmarks_from_video_2,load_array


def load_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")

        for idx in range(n):
            save_landmarks_from_video_2(videos_not_in_dataset[idx])

    return videos


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", sign_name, video_name)

        sign_list = load_array(os.path.join(path, f"{video_name}.pickle"))
        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(sign_list))
        reference_signs["distance"].append(0)
    
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs


def load_reference_face(videos):
    reference_face = {"name": [], "face_model": [], "distance": []}
    for video_name in videos:
        face_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", face_name, video_name)

        face_list = load_array(os.path.join(path, f"{video_name}.pickle"))
        

        reference_face["name"].append(face_name)
        reference_face["face_model"].append(FaceModel(face_list))
        reference_face["distance"].append(0)
    
    reference_face = pd.DataFrame(reference_face, dtype=object)
    print(
        f'Dictionary count: {reference_face[["name", "face_model"]].groupby(["name"]).count()}'
    )
    return reference_face