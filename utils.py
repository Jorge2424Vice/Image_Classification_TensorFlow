
import tensorflow as tf
import tensorflow_hub as hub
import json

IMAGE_SIZE = 224

def get_classes(class_path):
    '''
    Function takes a json path and returns dictionary for mapping
    class ids to class names.
    INPUT: class_path :(str) path to json file.
    OUTPUT: Dictionary with class_ids to class_names
    '''
    #Open json file
    with open(class_path, 'r') as f:
        class_names = json.load(f)
    #if json file is label_map.json then rename keys
    #Curren error in keys where the index is wrong for class name.
    if 'label_map.' in class_path:
        new_class_dict = dict()
        for key in class_names:
            new_class_dict[str(int(key)-1)] = class_names[key]
        return new_class_dict
    #If it isn't label_map then return dictionary as is.
    else:
        return class_names


def process_image(numpy_image):
    '''
    Function takes an image in numpy array format and processes for
    tensorflow ingestion.
    INPUT: numpy array
    OUTPUT: numpy for ingestion.

    '''
    print(numpy_image.shape)

    return  tf.image.resize(numpy_image,(IMAGE_SIZE, IMAGE_SIZE)).numpy()/255
