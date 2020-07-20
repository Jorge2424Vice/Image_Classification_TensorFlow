import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
from utils import process_image, get_classes

import argparse
import numpy as np
import json


def predict(image_path, model_path, top_k , class_names_json):
    '''
    Function takes inputs and prints top predicted class name, label and probability for flower image.
    It also prints the top k results for flower image.

    INPUT: image_path - (str) path to image.
           model_path - (str) path to tensorflow model (h5).
           top_k - (int) Top K results requested.
           class_names (json_file) Dict [class names : class ids]
    OUTPUT: NONE
    '''

    #Getting mapping file for class index and class names
    class_names = get_classes(class_names_json)

    #Reads Tensorflow model.
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})


    #open image
    img = Image.open(image_path)

    #put image into array
    image_numpy = np.asarray(img)

    #resize image for processing
    processed_image = process_image(image_numpy)


    #Predict image using tensorflow model
    prob_preds = model.predict(np.expand_dims(processed_image,axis=0))
    prob_preds = prob_preds[0]

    #Get top_k results as tensors.
    values, index= tf.math.top_k(prob_preds, k=top_k)

    #Conver tensors to numpy for use.
    probs = values.numpy().tolist()
    class_index = index.numpy().tolist()

    #Map class ids to class names.
    pred_label_names = []
    for i in class_index:
        pred_label_names.append(class_names[str(i)])



    #1 Result
    print(f"""\n\n Class most likely based on the following {image_path} with the highest probaility as listed: \n
          class_id: {class_index[0]} \n
          class_label: {pred_label_names[0]} \n
          probability: {str(round(float(probs[0]) *100, 2)) + '%'} \n\n\n
          """)

    if top_k >1:
        print(f"\n Top {top_k} probs", probs)
        print(f"\n Top {top_k} class names", pred_label_names)
        print(f"\n Top {top_k} class ids", class_index)
        print("\n\n")


if __name__ == "__main__":

    #Instantiate parser.
    parser = argparse.ArgumentParser(description = "Description for my parser")
    #Required image_path and saved_model
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")

    #Optional top_k value or 3 as a default.
    parser.add_argument("--top_k", help="Get top k predictions", required = False, type = int, default = 3)

    #Optional category json file name or 'label_map.json default.
    parser.add_argument("--category_names", help="Class map json file path/name", required = False, default = "label_map.json")

    #Takes all the parsed arguements.
    args = parser.parse_args()

    #calls predict function with all the parsed inputs.
    predict(args.image_path, args.saved_model, args.top_k, args.category_names)
