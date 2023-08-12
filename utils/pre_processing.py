from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
#from os.path import join 
#from viz.visualization import display_grid
import os


label_map={
  "Normal":1,
  "Tuberculosis":2,
}

data_root = "data/tuber_dataset"
index_to_label_dict= { index:label for label,index in label_map.items()}

def image_transforms(file_name, label) -> np.ndarray:
    file_path = os.path.join(data_root, label, file_name)
    array = read_image(file_path, "zoom", grayscale=True)
    flatten_image = array.flatten()
    return flatten_image


def label_transforms(label) -> int:
    # label_to_index
    return label_to_index(label)
  
def label_to_index( label:str):
  if label not in label_map:
    raise KeyError("label in not valid")
  return label_map[label]

def index_to_label( inx:int):
  if inx not in index_to_label_dict:
    raise KeyError("index is not valid")
  return index_to_label_dict[inx]
  

 

def read_image(image_path: str,mode:str,size:tuple=(256,256), grayscale:bool = False) ->np.ndarray:
    """ reads image from the given path and returns as a numpy array
    TODO: resize the image and implement the mode of zoom or paddding 
    args:
    -----
    image_path: the image which we want to read
    mode: either 'zoom' or 'pad'
    size:the size of the image we want to set to
    """

    image = Image.open(image_path)
    #image= image.resize(size)
    height, width= image.size
  
    if mode == "padding":
       if height== width:
         pass
       else:
        image=ImageOps.pad(image, (256, 256), color=None, centering=(0.5, 0.5))
    
    if mode== "zoom":
        diff= height-width
        if diff>0:
            right = width
            (left, upper, right, lower) = (0, diff//2, right, height-(diff//2))
            image= image.crop((left, upper, right, lower))
           
             
        else:
            lower= height
            diff = abs(diff)
            (left, upper, right, lower) = (diff//2, 0, width-(diff//2), lower)
            image = image.crop((left, upper, right, lower))
     
    image = image.resize((256,256))     
    img_array= np.asarray(image)
    return img_array

      
       
if __name__ == "__main__":
  index = label_to_index("Normal")
  print(index)
  
  value= index_to_label(1)
  print(value)
