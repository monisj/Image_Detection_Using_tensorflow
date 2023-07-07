import pandas as pd
import numpy as np
from collections import Counter

from utils.dtw import dtw_distances,dtw_distances_2
from models.sign_model import SignModel
from models.face_model import FaceModel

from utils.landmark_utils import extract_landmarks_face


class FaceRecorder(object):
    def __init__(self, reference_face: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_face = reference_face

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_face["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_face)

        if np.sum(self.reference_face["distance"].values) == 0:
            return "", self.is_recording
        return self._get_face_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        face_list = []
        for results in self.recorded_results:
            face = extract_landmarks_face(results)
            face_list.append(face)
            

        # Create a FacaeModel object with the landmarks gathered during recording
        recorded_face = FaceModel(face_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_face = dtw_distances_2(recorded_face, self.reference_face)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_face_predicted(self, batch_size=5, threshold=0.5):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        face_names = self.reference_face.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        face_counter = Counter(face_names).most_common()

        predicted_face, count = face_counter[0]
        if count / batch_size < threshold:
            return "Face Not Found"
        return predicted_face
