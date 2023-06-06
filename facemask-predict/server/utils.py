import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import magic
import torch

def decode_image(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def build_input_tensor(image_bytes):
    transform = transforms.Compose([transforms.ToPILImage(), 
                        transforms.Resize((16, 16)), 
                        transforms.Grayscale(),
                        transforms.ToTensor()])
    return transform(decode_image(image_bytes))

def predict(tensor, model):
    inp = tensor.view(1, 1, 16, 16)
    out = model(inp)
    pred = out.argmax()
    return pred.item()

def is_valid_mime(file_bytes):
    mime = magic.from_buffer(file_bytes, mime=True)
    discrete_type = mime.split("/")[0]
    return discrete_type == "image"

def is_valid_image(file_bytes):
    return is_valid_mime(file_bytes) and len(file_bytes)


@torch.no_grad()
def get_features(model, img):
    fmaps = []
    convs = [layer
             for layer in list(model.children()) 
             if isinstance(layer, nn.Conv2d)]  
    
    processed = [img]
    for conv in convs:
        processed.append(conv(processed[-1]))

    for idx, features in enumerate(processed[1::]):
        layer_name = f"CONV{idx+1}"
        layer_features = [transform_feature(f) for f in features]
        fmaps.append([layer_name, layer_features])
        
    return fmaps

def transform_feature(tensor):
    width, height = tensor.shape
    dim = (width*3, height*3)
    image = cv2.normalize(tensor.numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.uint8).tolist()
    return image

