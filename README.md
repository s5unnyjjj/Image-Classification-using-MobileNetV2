# Image-Classification-using-MobileNetV2-with-pytorch </br>
This project is written code based on mobileNet-v2.
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
</br>

## Dataset </br>
* Info: image classification 50 classes
* Num of dataset: training(500), validation(50), test(50)

## Traininig </br>
```
python train.py
 > Session().training()
```
 * Training loss graph
<img src = "https://user-images.githubusercontent.com/70457520/184366315-34267b21-d9a1-4633-9649-dcd856121b61.png" width="50%" height="50%">
</br>

## Validation
```
python train.py
 > Session().validation()
```
 * Validataion results at some iterations </br>
