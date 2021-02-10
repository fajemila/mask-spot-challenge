import numpy as np
import os
import torch
from torch.functional import F
import torch.nn as nn
import torchvision
import io
import torchvision.transforms as transforms
from flask import Flask , render_template,request,redirect,Response,url_for
from PIL import Image
import cv2
app = Flask(__name__)
UPLOAD_FOLDER = ""
DEVICE = "cpu"

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

def transform_image2(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(image_bytes).convert('RGB')
    return my_transforms(image).unsqueeze(0)

class Model(nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()
        self.arch = arch
        if 'DenseNet' in str(arch.__class__):
            self.arch.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
        self.Linear = nn.Linear(1024,2)
    def forward(self, inputs):
        x = inputs
        x = self.arch(x)
        # x = self.Linear(x)
        x = F.sigmoid(x)
        return x

arch = torchvision.models.densenet121(pretrained=True)

modelmask = Model(arch)
modelmask.load_state_dict(torch.load("model_fold_3.bin",map_location=torch.device('cpu')))
modelmask.eval()

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = modelmask.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.detach().item()
def get_prediction2(image_bytes):
    tensor = transform_image2(image_bytes=image_bytes)
    outputs = modelmask.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.detach().item()

@app.route('/maskclass',methods=['POST',"GET"])
@app.route('/',methods=['POST',"GET"])
def maskclass():
    if request.method == "POST":
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image.save('static/took.jpg')
        # cv2.imwrite('static/took.jpg',image)
        class_name = get_prediction(image_bytes=img_bytes)
        class_dict = {0:"is no",1:"is a"}
        prediction = "From the image uploaded, there " + str(class_dict[class_name])  + ' face mask'
        return render_template('maskclass.html', prediction=prediction)
    return render_template('maskclass.html')

# UPLOAD_FOLDE = 'static/uploads'
@app.route('/cap',methods=['POST',"GET"])
def cap():
    if request.method == "POST":
        video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        result = True
        while result:
            ret,frame = video.read()
            cv2.imwrite('static/bush.jpg',frame)
            result = False
        video.release()
        cv2.destroyAllWindows()
        class_name = get_prediction2(image_bytes='static/bush.jpg')
        class_dict = {0:"is no",1:"is a"}
        prediction = "From the image uploaded, there " + str(class_dict[class_name])  + ' face mask'
        image_file = url_for('static',filename='bush.jpg')
        return render_template('cap.html', prediction=prediction,image_sc = image_file)
    return render_template('cap.html')

if __name__=="__main__":
    app.run(debug=True)
