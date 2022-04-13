########################################################################################
### Code by Mayank Kumar 
### Training code for phone detection using Faster RCNN
########################################################################################
### Import department 

import os
import cv2 as cv
import torch
import torchvision
import utils
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from CustomDataset import *

########################################################################################
### train function declaration 
def finalTrainingModel():
    """
    downloads the model definition and modifies the classifer for the use case.
    
    output: modified model definition
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) ###load a model pre-trained on COCO
    num_classes = 2  ### 1 class (phone) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features  ### get number of input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) ### input feature mapped to desired classifier
    return model

def train(model, full_dataset, num_epochs = 10):
    """ 
    model: instance of model with customised classifier 
    full_dataset: it takes all images and splits the data into 80:20 ratio before start of training. 
    num_epochs: default - 10, can be changes accordingly 
    """
    ### select device 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ### train and test split on full dataset
    dataset, dataset_test = train_test_split(full_dataset, test_size=0.2, shuffle=True, random_state=43)
    
    ### define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, 
                                                num_workers=4,collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, 
                                                num_workers=4,collate_fn=utils.collate_fn)

    ### send model to available device
    model.to(device) 

    ### Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    ### Schedular
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    ### start of the epochs
    for epoch in tqdm(range(num_epochs)):
        ### train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step() ### update the learning rate
        evaluate(model, data_loader_test, device=device) ### evaluate on the validation dataset

    print("Training Completed!!")
########################################################################################
### driver code

if __name__ == '__main__':
    ### get arguments/foldername from the terminal
    foldername = '' 
    if len(sys.argv) > 1:
        foldername = sys.argv[1]    #absolute path to the folder
    else:
        print('No Dataset Folder provided')
        raise ValueError('Please Provide a Dataset Folder')
    
    ### loads all images as a custom dataset
    complete_dataset = PhoneDataset(foldername) 
    model = finalTrainingModel()
    
    ### start training 
    print("Training Started...")
    train(model, complete_dataset, num_epochs= 10)
    print("Training Completed...")
    ### save trained model to local folder for testing purpose
    print("Saving Model...")
    SavePATH = os.path.join(os.getcwd(), "TrainedModel/FasterRCNN_10EPOCHS_1.pth")
    torch.save(model.state_dict(), SavePATH)
    print("Model Saved!!")

########################################################################################

