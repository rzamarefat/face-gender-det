import torch 
import torchvision
from Dataset import GenderDataset
from torch.utils.data import DataLoader
from glob import glob
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# =====> config
BATCH_SIZE = 32
MODEL_NAME = "efficientnet_b7" #"mobilenet_v2"
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INITIAL_LEARNING_RATE = 3e-4
ROOT_PATH_TO_TRAIN_IMAGES = "/home/rmarefat/projects/github/face_gender_det/cropped_faces/Train"
ROOT_PATH_TO_VAL_IMAGES = "/home/rmarefat/projects/github/face_gender_det/cropped_faces/Validation"
PATH_TO_SAVE_CKPT = f"/home/rmarefat/projects/github/face_gender_det/src/weights/model__{MODEL_NAME}.pt"


def train_step(model, optimizer, criterion, imgs, labels):
    out = model(imgs)
    loss = criterion(out, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = torch.argmax(out, dim=1)
    preds = preds.detach().to('cpu').numpy()
    
    y = y.detach().to("cpu").numpy()
    train_acc = accuracy_score(y, preds)
    
    return train_acc, loss.item()

def val_step():
    out = model(imgs)
    loss = criterion(out, y)

    preds = torch.argmax(out, dim=1)
    preds = preds.detach().to('cpu').numpy()
    
    y = y.detach().to("cpu").numpy()
    validation_acc = accuracy_score(y, preds)

    return validation_acc, loss.item()


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


    
def get_model(model_name):
    if model_name == "efficientnet_b7":
        model = torchvision.models.efficientnet_b7(pretrained=True)

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=2560, out_features=1)
        )

        for n,p in model.named_parameters():
            p.requires_grad = False
            if n.__contains__("features.8") or n.__contains__("features.7") or n.__contains__("classifier"):
                p.requires_grad = True

        return model
    


def run_engine():
    train_loader, val_loader = get_loaders()
    model = get_model(model_name=MODEL_NAME)
    model.to(DEVICE)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    min_val_loss = 10 ** 10

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch: {epoch}")

        running_train_acc = []
        running_val_acc = []
        running_train_loss = []
        running_val_loss = []

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            model.train()
            train_acc, train_loss.item() = train_step(model, optimizer, criterion, imgs, labels)

            model.eval()
            val_acc, val_loss.item() = val_step(model, optimizer, criterion, imgs, labels)

            running_train_acc.append(train_acc)
            running_train_loss.append(train_loss)
            running_val_acc.append(val_acc)
            running_val_loss.append(val_loss)

        epoch_train_acc = round(sum(train_acc) / len(train_acc), 2)
        epoch_train_loss = round(sum(train_loss) / len(train_loss), 2)
        epoch_val_acc = round(sum(val_acc) / len(val_acc), 2)
        epoch_val_loss = round(sum(val_loss) / len(val_loss), 2)


        report_statement = f"Epoch {epoch} | train_acc {epoch_train_acc} | train_loss {epoch_train_loss} | val_acc {epoch_val_acc} | val_loss {epoch_val_loss}"

        with open("TRAIN_REPORT.txt", "a+") as h:
            h.seek(0)
            h.writelines(report_statement)
            h.writelines("\n")


        if epoch_val_loss < min_val_loss:
            torch.save(model.state_dict(), PATH_TO_SAVE_CKPT)
            




if __name__ == "__main__":
    run_engine()