# Image-Classification-and-Serving-TensorFlow
This repo will show you the procedure of solving an image classification problem with tensorflow2.0 and serving the trained model.

## How to use

### Requirements
+ python3.6.8
+ pip install tensorflow-gpu==1.12.0
+ The file directory of the dataset should look like this: 
```
${dataset_root}
|——train
|   |——class_name_0
|   |——class_name_1
|   |——class_name_2
|   |——class_name_3
|——valid
|   |——class_name_0
|   |——class_name_1
|   |——class_name_2
|   |——class_name_3
|——test
    |——class_name_0
    |——class_name_1
    |——class_name_2
    |——class_name_3
```

### Train
Run the script
```
python train.py
```
to train the network on your image dataset, the final model will be stored. You can also change the corresponding training parameters in the `config.py`.<br/>

### Evaluate
To evaluate the model's performance on the test dataset, you can run `evaluate.py`.<br/>

The structure of the network is defined in `model_definition.py`, you can change the network structure to whatever you like.

### Serving
After you pull the tensorflow serving from docker hub, run tensorflow serving service:
```sudo docker run -it -p 8501:8501 -v "$(pwd)/saved_model/flower_photos_serving/:/models/flower_photos_serving" -e MODEL_NAME=flower_photos_serving tensorflow/serving
```
Then use ```python evaluate_serving.py``` to evaluate the model on tensorflow serving.
