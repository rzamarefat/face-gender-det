import torch
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, extract_face
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np

class MTCNNFaceDetector:
    def __init__(self):
        self.device = torch.device('cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device, selection_method='largest')
        self._crop_size = (112, 112)
        self._refrence = get_reference_facial_points(default_square=self._crop_size[0] == self._crop_size[1])


    def _crop_and_align(self, source_image, landmarks):
        faces = []
        for l in landmarks:
            warped_face = warp_and_crop_face(np.array(source_image), l, self._refrence, crop_size=self._crop_size)
            faces.append(warped_face)

        return faces


    def _detect_pose(self, landmarks):
        """
        ref: https://github.com/fisakhan/Face_Pose/blob/master/pose_detection_mtcnn.py
        points : TYPE - Array of float32, Size = (10,)
        """
        rolls = []
        yaws = []
        pitchs = []
        for landmark in landmarks:
            roll = landmark[6] - landmark[5]
            
            le2n = landmark[2] - landmark[0]
            re2n = landmark[1] - landmark[2]
            yaw = le2n - re2n
        
            eye_y = (landmark[5] + landmark[6]) / 2
            mou_y = (landmark[8] + landmark[9]) / 2
            e2n = eye_y - landmark[7]
            n2m = landmark[7] - mou_y
            pitch = e2n / n2m

            rolls.append(roll)
            yaws.append(yaw)
            pitchs.append(pitch)
        
        return rolls, yaws, pitchs


    def get_face(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        
        if len(boxes) == 0:
            return [], []
        
        faces = self._crop_and_align(image, landmarks)
        
        landmarks_for_pose = []
        for landmark in landmarks:
            reorganized_landmark = []
            for l in landmark:
                reorganized_landmark.append(l[0])
            for l in landmark:
                reorganized_landmark.append(l[1])
            landmarks_for_pose.append(reorganized_landmark)

        rolls, yaws, pitchs = self._detect_pose(landmarks_for_pose)

        

        return faces, boxes , landmarks, rolls, pitchs, yaws


if __name__ == "__main__":
    from uuid import uuid1
    image_path = "/home/rmarefat/projects/github/face_gender_det/Dataset/Test/Female/160001.jpg"
    
    img = cv2.imread(image_path)

    mtcnn = MTCNNFaceDetector()
    

    faces, boxes , landmarks, rolls, pitchs, yaws = mtcnn.get_face(img)

    