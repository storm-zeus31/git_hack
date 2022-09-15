import json
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
import tensorflow.lite as tflite
import numpy 
import random

class_index ={
    0:'sad',
    1:'happy'
}

def prediction(image: Image.Image):
    interpreter = tflite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image =image.resize((224, 224))
    input_shape = input_details[0]['shape']
    input_tensor= np.array(np.expand_dims(image,0),dtype=np.float32)
    
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    highest_pred_loc = np.argmax(pred)
    predictedValue =class_index[highest_pred_loc]
    n=random.randint(10,20)
    moodList = np.array([0] * n + [1] * (60-n))
    np.random.shuffle(moodList)
    nlist=moodList.tolist()
    json_str = json.dumps(nlist)
    response ={
        "mood":predictedValue,
        "rangeArray":json_str
    }

    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
   

  

    

