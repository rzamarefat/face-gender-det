import torch 
import torchvision
from Dataset import GenderDataset
from torch.utils.data import DataLoader
from glob import glob
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from train import get_model
import matplotlib.pyplot as plt
import numpy as np


# =====> config
BATCH_SIZE = 120
MODEL_NAME = "efficientnet_b7" #"resnet101" #"mobilenet_v2"  
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
INITIAL_LEARNING_RATE = 3e-4
ROOT_PATH_TO_TEST_IMAGES = "/home/rmarefat/projects/github/face_gender_det/cropped_faces/Test"

PATH_TO_SAVE_CKPT = f"/home/rmarefat/projects/github/face_gender_det/src/weights/model__{MODEL_NAME}.pt"



def run_evaluation():
    model = get_model(model_name=MODEL_NAME)

    model.load_state_dict(torch.load("/home/rmarefat/projects/github/face_gender_det/src/weights/model__efficientnet_b7.pt"))

    model.to(DEVICE)

    test_images = [f for f in sorted(glob(os.path.join(ROOT_PATH_TO_TEST_IMAGES, "*", "*")))]

    test_ds = GenderDataset(test_images)
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs, labels = imgs.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            preds = preds.detach().to('cpu').numpy()
            labels = labels.detach().to("cpu").numpy()
            all_labels.extend([l for l in labels])
            all_preds.extend([p for p in preds])
            break


        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        print("all_labels.shape",all_labels.shape)
        print("all_preds.shape", all_preds.shape)

        # tn, fp, fn, tp = confusion_matrix(all_preds, all_labels).ravel()
        # print(f"TN {tn} | FP {fp} | FN {fn} | TP {tp}")
        
        cm = confusion_matrix(all_labels, all_preds, labels=[1., 0.])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1., 0.])
        disp.figure_.savefig('eff.png')
        
        





if __name__ == "__main__":
    run_evaluation()

