import torch 
from torchvision import models
from Dataset import GenderDataset
from torch.utils.data import DataLoader

# =====> config
BATCH_SIZE = 32
MODEL_NAME = "mobilenet_v2"
EPOCHS = 100
INITIAL_LEARNING_RATE = 3e-4
ROOT_PATH_TO_TRAIN_IMAGES = "/home/rmarefat/projects/github/face_gender_det/cropped_faces/Train"
ROOT_PATH_TO_VAL_IMAGES = "/home/rmarefat/projects/github/face_gender_det/cropped_faces/Validation"


def train_step():
    pass

def val_step():
    pass


def get_loaders():
    train_images = [f for f in sorted(glob(os.path.join(ROOT_PATH_TO_TRAIN_IMAGES, "*", "*")))]
    val_images = [f for f in sorted(glob(os.path.join(ROOT_PATH_TO_VAL_IMAGES, "*", "*")))]

    print(f"Number of train images: {len(train_images)}")
    print(f"Number of val images: {len(val_images)}")

    train_ds = GenderDataset(train_images)
    val_ds = GenderDataset(val_images)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


    
def get_model():
    pass

    


def run_engine():
    train_loader, val_loader = get_loaders()
    model = get_model()
    optimizer = torch.optim.Adam(model)

    for epoch in range(1, EPOCHS + 1):

        
    pass


if __name__ == "__main__":
    run_engine()