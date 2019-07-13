# serving using Restful API
import os
import json
import time
import requests

import numpy as np
from PIL import Image

import config

def get_input(img_path, input_size):
    img = Image.open(img_path).convert('RGB').resize(input_size)
    img = np.array(img).astype('float32')
    img = img / 255.
    img = (img - np.mean(img)) / np.std(img)
    img = np.expand_dims(img, axis=0)

    return img

def main(data_path):
    count = 0
    total = 0
    for i, _ in enumerate(sorted(os.listdir(data_path))):
        sub_path = os.path.join(data_path, _)
        imgs = [f for f in os.listdir(sub_path) if not f.startswith('.')]
        for img in imgs:
            input_data = get_input(os.path.join(sub_path, img), (224, 224))
            data = {'inputs': {'input_1': input_data.tolist()}}
            r = requests.post('http://localhost:8501/v1/models/flower_photos_serving:predict', json=data)
            results = json.loads(r.content.decode('utf-8'))
            probs = np.asarray(results['outputs'][0])
            index = np.argmax(probs)
            if index == i:
                count += 1
            total += 1

    print("Serving Accuracy: {:6.3f}".format(count / total * 100))

if __name__ == "__main__":
    main(config.valid_dir)
