import pandas as pd
from fastdtw import fastdtw
import numpy as np
from models.sign_model import SignModel
from models.face_model import FaceModel

def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    for idx, row in reference_signs.iterrows():
        # Initialize the row variables
        ref_sign_name, ref_sign_model, _ = row

        # If the reference sign has the same number of hands compute fastdtw
        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding

            if recorded_sign.has_left_hand:
                row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
            if recorded_sign.has_right_hand:
                row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])

def dtw_distances_2(recorded_face: FaceModel, reference_face: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded Face & the reference Faces

    :param recorded_face: a FaceModel object containing the data gathered during record
    :param reference_Face: pd.DataFrame
                            columns : name, dtype: str
                                      face_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a Face dictionary sorted by the distances from the recorded Face
    """
    # Embeddings of the recorded Face
    rec_face = recorded_face.face_embedding
    

    for idx, row in reference_face.iterrows():
        # Initialize the row variables
        ref_face_name, ref_face_model, _ = row

        # If the reference sign has the same number of hands compute fastdtw
        if (recorded_face.has_face == ref_face_model.has_face):
            ref_face = ref_face_model.face_embedding
            

            if recorded_face.has_face:
                row["distance"] += list(fastdtw(rec_face))[0]
            
        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_face.sort_values(by=["distance"])