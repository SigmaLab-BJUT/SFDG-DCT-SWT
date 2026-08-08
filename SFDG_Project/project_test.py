import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from models.MAT import MAT
import pickle
import dlib
from kornia.filters import gaussian_blur2d, blur_pool2d
from torchvision.utils import save_image
from datasets.augmentations import get_boundingbox
from skimage.feature import local_binary_pattern
from albumentations import CenterCrop,Compose,Resize,RandomCrop, Normalize
from albumentations.pytorch.transforms import ToTensor
from albumentations.pytorch import ToTensorV2


def data_prepare(path):
    resize=(380,380)
    trans=Compose([Resize(*resize), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2(p=1)])
    save_trans=Compose([Resize(*resize),ToTensor()])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_detector=dlib.get_frontal_face_detector()
    faces = face_detector(gray, 1)
    if len(faces):
        # For now only take biggest face
        face = faces[0]
        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]
    else:
        raise IOError('There are no faces in the current image')
        # cropped_face = image
    # cropped_face = image
    
    out_image = trans(image=cropped_face)['image']

    save = save_trans(image=cropped_face)['image']
    name = path.split('/')[-1]
    save_image(save, './test_img/crop_face/'+ name)
    
    return out_image
  

def load_model(name):
    with open('runs/%s/config.pkl'%name,'rb') as f:
        config=pickle.load(f)
    net= MAT(**config.net_config).cuda()
    return config,net

if __name__ == '__main__':
    name = 'Freq2_LBP_Graph_23'
    attention = False
    prepare_real = True
    ckpt = 29 # 29
    config, net=load_model(name)
    print('Starting load the SFDG Model: ')
    state_dict=torch.load('./checkpoints/%s/ckpt_%s.pth'%(name,ckpt))['state_dict']
    net.load_state_dict(state_dict,strict=False)
    print('Model Successful load!')
    net.eval()
    name_all = os.listdir('./test_img/full_image/')
    for pic in name_all:
        img = data_prepare('./test_img/full_image/'+pic)
        print('./test_img/full_image/'+pic)
        logit = net(x=img.unsqueeze(0).cuda())
        predict_index=torch.max(logit,dim=1)[1]
        predict = F.softmax(logit, dim=1)
        if predict_index == 0:
            print('The test image is predicted as Real and the logit is %.4f' % predict[0,0])
        else:
            print('The test image is predicted as Fake and the logit is %.4f' % predict[0,1])
        