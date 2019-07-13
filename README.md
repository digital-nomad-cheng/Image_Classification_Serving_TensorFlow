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
To evaluate the model's performance on the test dataset, you can run `python evaluate_*.py`.

The structure of the network is defined in `model.py`, you can change the network structure to whatever you like.

Here I provide three scripts to evaluate keras/saved_model/model through serving to gurantee there isn't any discrepancy in the model deployment procedure.

#### Keras
Evaluate the saved keras model:
```
python evaluate_keras.py
```
#### Saved Model
Evaluate using keras model loaded from saved_model:
```
python evaluate_serving.py
```
#### TensorFlow Serving
Evaluate saved model through tensorflow serving service:

After you pull the tensorflow serving from docker hub, run tensorflow serving service:
```
sudo docker run -it -p 8501:8501 -v "$(pwd)/saved_model/flower_photos_serving/:/models/flower_photos_serving" -e MODEL_NAME=flower_photos_serving tensorflow/serving
```
Then use ```python evaluate_serving.py``` to evaluate the model on tensorflow serving.

- [ ] using batching strategy on serving side to test latency

