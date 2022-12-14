# Towards-safe-visual-navigation-of-a-wheelchair-using-landmark-detection
Multi-label image classification using a fine-tuned vision transformer
# Indoors-Traversability-Estimation-with-Less-Labels for Mobile Robots


1. Open a terminal window
2. Clone the repo

## Install the requirements on your machine 

`pip install -r requirements.txt`

## Fine-tuning on your dataset 


For fine-tuning on your dataset using e.g. the fine tuned ViT:
1. Open ViT_fine_tuned.py
2. Specify the train and test dataset paths as '/home/../../set'
3. Run `python3 VIT_fine_tuned.py`

You can follow the exact same process for fine-tuning with pretrained ResNet50

## Using Ensemble GAN as described by Hirose et al. [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8594031)

#### For training on a large dataset
1. Arrange your folder structure as /../../data/class/
2. Set the dataroot path '/../../data/' 
3. Run `python3 GAN_ensemble_train.py`

The model saves thre .pth files each corresponding to the models of the ensemble

#### For testing

1. Open the script gan_classifier.py
2. Specify the saved models to load as `netD.load_state_dict(torch.load('model.pth'))`
3. Specify the train and test path as '/../../train/' and '/../../test/' 
4.Run `python3 gan_classifier.py`

### GoPro dataset

Link to the dataset [here](https://drive.google.com/file/d/1LrNHrJbAxsuKhw8TV0OvhKZk_E9SReLg/view?usp=share_link)


