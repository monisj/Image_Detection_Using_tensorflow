from typing import List

import numpy as np

from models.face_detect_model import Face_Detect_Model


class FaceModel(object):
    def __init__(
        self, face_list: List[List[float]]
    ):
        """
        Params
            face_list: List of all landmarks for each frame of a video
        Args
            has_x_hand: bool; True if x hand is detected in the video, otherwise False
            xh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections)
        """
        self.has_face= np.sum(face_list) != 0

        self.face_embedding = self._get_embedding_from_landmark_list(face_list)

    @staticmethod
    def _get_embedding_from_landmark_list(
        face_list: List[List[float]],
    ) -> List[List[float]]:
        """
        Params
            face_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        embedding = []
        for frame_idx in range(len(face_list)):
            if np.sum(face_list[frame_idx]) == 0:
                continue

            face = FaceModel(face_list[frame_idx])
            embedding.append(face.feature_vector)
        return embedding
