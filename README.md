# Image-Classification-using-MobileNetV2-with-pytorch </br>
This project is written code based on mobileNet-v2.
</br>
</br>

## Prerequisite </br>
 * Python = 3.7</br>
 * Pytorch = 1.7.1</br>
 * Opencv</br>
 * Pillow</br>

## Project Structure </br>
 * train.py: Code to train the mobileNet model
 * evaluate.py: Code to test the mobileNet model
 * model.py: Build the architecture of MobileNet-v2
 * main.py: Execute the project
 * settting.py: Assign value of various parameters

## Dataset </br>
* image classification 50 classes
* Num of dataset: training(500), validation(50), test(50)
* Image size 320x320 (RGB 3 channels)
* Training images are given with corresponding lables
* Validation and Test images are not given with no label information
* Example
<img src = "https://github.com/s5unnyjjj/Image-Classification-using-MobileNetV2/assets/70457520/051e6342-0466-4362-9550-9f8cf92716b9" width="50%" height="50%">

## Traininig and Validation </br>
```
python main.py
 > start_training()
 > start_evaluation()
```
 * Training loss graph
<img src = "https://github.com/s5unnyjjj/Image-Classification-using-MobileNetV2/assets/70457520/593dd82b-04df-41dc-bb45-43ae0058e887" width="50%" height="50%">
</br>
