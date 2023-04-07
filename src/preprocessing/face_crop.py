from glob import glob
import cv2
from MTCNNFaceDetector import MTCNNFaceDetector
from tqdm import tqdm
from uuid import uuid1
import os

path_to_data = "/home/rmarefat/projects/github/face_gender_det/Dataset/Validation/*/*"
path_to_save = "/home/rmarefat/projects/github/face_gender_det/cropped_faces"

dones = [f.split("/")[-1] for f in sorted(glob(os.path.join(path_to_save, "*", "*", "*")))]



def preprocess():
    mtcnn = MTCNNFaceDetector()
    

    for img_p in tqdm(sorted(glob(path_to_data))):
        img_name = img_p.split("/")[-1]
        img_class = img_p.split("/")[-2]
        img_category = img_p.split("/")[-3]

        if img_name in dones:
            print("Skipping...")
            continue
        
        

        if not(os.path.isdir(os.path.join(path_to_save, img_category))):
            os.mkdir(os.path.join(path_to_save, img_category))

        if not(os.path.isdir(os.path.join(path_to_save, img_category, img_class))):
            os.mkdir(os.path.join(path_to_save, img_category, img_class))


        try:
            img = cv2.imread(img_p)
            faces, boxes , landmarks, rolls, pitchs, yaws = mtcnn.get_face(img)

            # for face in faces:
            cv2.imwrite(os.path.join(path_to_save, img_category, img_class, f"{img_name}"), faces[0])

        except Exception as e:
            print(e)
            continue
        



if __name__ == "__main__":
    preprocess()