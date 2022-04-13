# Load Dataset 
import collections
import os
import cv2 as cv
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

class PhoneDataset(Dataset):
    def __init__ (self, root, transform = None):
        self.root = root
        self.transform = transform
        self.images = list(sorted([file for file in os.listdir(root) if file.endswith('.jpg')]))
        self.annotations = self.generateAnnotFiles(self.root)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images[idx])
        img = Image.open(image_path).convert("RGB")
        img = T.ToTensor()(img)

        boxes = self.annotations[self.images[idx]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.unsqueeze(0)
        # print("check: ", boxes.shape)
        
        labels = torch.ones((len(boxes),), dtype= torch.int64)
        area = (boxes[:,3] - boxes [:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((boxes.shape[0], ), dtype= torch.int64)
        image_id = int(self.images[idx][:-4])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['image_id'] = torch.as_tensor(image_id, dtype = torch.int64)
        # print("verify: ",target)
        return img, target

    def __len__(self):
        return len(self.images)

    def getDatasetInfo(self, foldername):
        file = open(foldername + '/labels.txt', 'r')
        lines = file.read().splitlines()
        obj_coordinates = {}
        for line in lines:
            label = line.split(' ')
            obj_coordinates[label[0]] = {'coordinates' : (float(label[1]), float(label[2]))}
        for file in os.listdir(foldername):
            if file.endswith('.jpg'):
                # print(file)
                img = cv.imread(foldername + "/" +file)
                h, w, c = img.shape
                obj_coordinates[file]['size'] = (h,w)
        return obj_coordinates

    def getPixelfromCoordsinX(self, img_w, x):
        pix_x = img_w * x
        return round(pix_x)

    def getPixelfromCoordsinY(self, img_h, y):
        pix_y = img_h * y
        return round(pix_y)

    def generateAnnotFiles(self, foldername):
        dataset_info = self.getDatasetInfo(foldername)
        dict = {}
        for file in os.listdir(foldername):
            if file.endswith('.jpg'):
                h, w = dataset_info[file]['size']
                c1, c2 = dataset_info[file]['coordinates']
                # print("check:  ", file, ": ", (c1, c2),": ",(c1*w, c2*h))
                pix_x_min = self.getPixelfromCoordsinX(w, c1-0.05)
                pix_y_min = self.getPixelfromCoordsinY(h, c2-0.05)
                pix_x_max = self.getPixelfromCoordsinX(w, c1+0.05)
                pix_y_max = self.getPixelfromCoordsinY(h, c2+0.05)
                dict[file] = [pix_x_min, pix_y_min, pix_x_max, pix_y_max]
        
        return dict