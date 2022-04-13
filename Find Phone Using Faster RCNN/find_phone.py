########################################################################################
### Code by Mayank Kumar 
### Locate phone in the image and return normalised center 
### trained Faster RCNN model is loaded from local folder 

########################################################################################
### Import Department

import os
import cv2 as cv 
from PIL import Image
import torch 
import torchvision 
import sys
from torchvision.transforms import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import warnings
warnings.filterwarnings("ignore")

########################################################################################
### Function definition department 

def ImageToTensor(imgPATH): 
    """
    Input: image path 
    Output: image as a tensor 
    """
    img = Image.open(imgPATH).convert("RGB")
    img = T.ToTensor()(img)
    return img

def getCenter(prediction, XLen, YLen):
    """
    input 1: gets direct prediction from the model, 
    input 2: image dimension in X direction
    input 3: image dimension in Y direction

    output: center of the phone (X, Y)
    """
    prediction = prediction[0]['boxes'].tolist()[0]
    centerX = ( prediction[0] + prediction[2] )/(2 * XLen)
    centerY = ( prediction[1] + prediction[3] )/(2 * YLen)
    return round(centerX,4), round(centerY,4)

def getPrediction(imgPATH, model):
    """
    input 1: takes image path and internally calls other relevant functions to generate center prediction
    input 2: model with modified classifier
    
    output: return center of the phone
    """
    img = ImageToTensor(imgPATH)
    with torch.no_grad():
        prediction = model([img.to(device)])
    return getCenter(prediction, img.shape[2], img.shape[1])

def finalModel():
    """
    downloads the model definition and modifies the classifer for the use case.
    
    output: modified model definition
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False) ###load a model pre-trained on COCO
    num_classes = 2  ### 1 class (phone) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features  ### get number of input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) ### input feature mapped to desired classifier
    return model
    
########################################################################################
###  Driver Code 
if __name__ == '__main__': 
    imageLocation = '' 
    if len(sys.argv) > 1:
        imageLocation = sys.argv[1]    
    else:
        print('No image found!!')
        raise ValueError('please provide a valid PATH')

    ### here's the path to the saved model 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    modelPATH = os.path.join(os.getcwd(), "TrainedModel/FasterRCNN_10EPOCHS.pth")
    ### Load instance of the model
    model = finalModel()
    model.load_state_dict(torch.load(modelPATH, map_location = device))   ### load model 
    model.eval()  ### model to eval mode

    print(getPrediction(imageLocation, model))

########################################################################################