## Code by Mayank Kumar: Find phone on small dataset
 
#### Phone detection task has been performed using Faster-RCNN with ResNet-50 as backbone. 
### Commands
- Open terminal at root directory or cd to root directory
- Install dependencies 
    ```
    pip install -r requirements.txt

- training: train_phone_finder.py
    ```
    > python train_phone_finder.py ~/find_phone

- testing: find_phone.py
    ```
    > python find_phone.py ~/find_phone_test_images/51.jpg 

in both the cases, path to folder (training script) or image (testing script) should be absolute path.

### Directory structure
root: Kumar-Mayank-FindPhone
.
├── CustomDataset.py
├── Find phone.pdf
├── FindPhoneAlgo.ipynb
├── Outcome_all_Images.csv
├── ReadMe.md
├── TrainedModel
├── __pycache__
├── coco_eval.py
├── coco_utils.py
├── engine.py
├── find_phone
├── find_phone.py
├── requirements.txt
├── train_phone_finder.py
├── transforms.py
└── utils.py

- `CustomDataset.py` contains declaration of PhoneDataset class. This is custom written class to load data in format needed before passing it to dataloader for training and validation. 
- TrainedModel: contains model trained on the provided dataset in folder find_phone. While testing script will load the model from this folder, move it in evaluation mode and then display the predicted co-ordinates. 
- `FindPhoneAlgo.ipynb` was used for quick prototyping in google colab environment. 
- `find_phone.py` and `train_phone_finder.py` are the scripts required for the submission. 
- other `.py` files are taken from torchvision repository and contains supporting methods for training and testing. 
- `Outcome_all_Images.csv`: contains prediction on complete dataset with difference of distance between actual annonation and predicted center of the phone. 


### Some considerations
- used pretrained Faster R-CNN with ResNet50 as backbone. Model is trained on coco dataset. 
- Classifier layer was modified to work with 2 classes (0- background, 1-phone). 
- As Faster R-CNN model expects bounding boxes instead of single point. I have created those bounding boxes under assumption that camera capturing the images is located at approximately same focal length and hence, area taken by a phone will remain same. This can be a good area of inmprovement if we are considering more generalised outcome. 
- `TrainedModel` folder contains the model trained with modified clasiifier. This is trained using GPU runtime in google colab. 
- `train_phone_finder.py` will automatically take care of device (cuda or cpu). But it is recommended to use cuda for faster training. 
- currently, above script consider 10 epochs by default. 

### Some improvement points. 
- Data augmentation techniques can be used to improve the model accuracy and to avoid overfitting. 
- While data augmentation, we need to take special care of augmenting the bounding boxes. One easy way to do it is using libraries like `Albumentations`. This automatically takes care of augmenting bounding boxes while applying any transformation. 
- In `Inference` phase, I am downloading a model instance from torchvision library with keyword `pretrained = False`. This adds to the overall time for inference. This can be optimised by having the model instance in local directory or using TorchScript to save a model instance in the local folder. 
- Currently backbone used is ResNet50, model can also be tested with other backbones with less numner of parameters. 


