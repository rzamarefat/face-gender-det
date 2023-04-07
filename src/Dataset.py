import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class GenderDataset(Dataset):
    def __init__(self, images_path, image_size=(224, 224)):
        self.images_path = images_path
        self.image_size = image_size

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(self.image_size),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.label_encoder = {
            "Male": 0.,
            "Female": 1.,
        }

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        target_img_path = self.images_path[index]
        target_img_label = self.images_path[index].split("/")[-2]

        label = self.label_encoder[target_img_label]
        
        img = Image.open(target_img_path)

        img = self.transforms(img)

        return img, label

if __name__ == "__main__":
    from glob import glob
    images_path = [f for f in sorted(glob("/home/rmarefat/projects/github/face_gender_det/cropped_faces/Train/*/*"))]
    gen_ds = GenderDataset(images_path)
    
    for imgs, labels in gen_ds:
        print(imgs.shape)
        print(labels)