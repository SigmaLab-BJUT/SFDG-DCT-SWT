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
from albumentations.pytorch import ToTensorV2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def data_prepare(path):
    resize=(380,380)
    trans=Compose([Resize(*resize), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2(p=1)])
    save_trans=Compose([Resize(*resize),ToTensorV2()])
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片文件: {path}")
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
    save_path = '../result/SFDG/'+ name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 修复 save_image 类型问题：确保tensor在正确的范围内
    save_normalized = save.float() / 255.0 if save.max() > 1 else save.float()
    save_image(save_normalized, save_path)

    return out_image
  

def load_model(name):
    with open('runs/%s/config.pkl'%name,'rb') as f:
        config=pickle.load(f)
    net= MAT(**config.net_config).cuda()
    return config,net


def SFDG_process_primary(imgpath):

    # iamge = cv2.imread(imgpath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    name = 'Freq2_LBP_Graph_23'
    attention = False
    prepare_real = True
    ckpt = 29  # 29
    config, net = load_model(name)
    # print('Starting load the SFDG Model: ')
    state_dict = torch.load('./checkpoints/%s/ckpt_%s.pth' % (name, ckpt))['state_dict']
    net.load_state_dict(state_dict, strict=False)
    # print('Model Successful load!')
    net.eval()
    print(imgpath)
    img = data_prepare(imgpath)
    logit = net(x=img.unsqueeze(0).cuda())
    predict_index = torch.max(logit, dim=1)[1]
    predict = F.softmax(logit, dim=1)
    if predict_index == 0:
        print('The test image is predicted as Real and the logit is %.4f' % predict[0, 0])
        label = 'real'
        # probability = predict[0, 1]    #输出是伪造概率

    else:
        print('The test image is predicted as Fake and the logit is %.4f' % predict[0, 1])
        label = 'fake'
        # probability = predict[0, 1]   #输出是伪造概率
    # probability = '%.4f' % predict[0, 1]  # 输出是伪造概率
    probability = float('%.4f' % predict[0, 1])

    name = imgpath.split('/')[-1]
    SFDG_path = '../result/SFDG/' + name

    # print('*'*10)
    # print(probability, label, SFDG_path)

    # font_face = cv2.FONT_HERSHEY_SIMPLEX
    # thickness = 2
    # font_scale = 1
    # cv2.putText(image, str("%.2f" % predicted) + '=>' + label, (x, y + h + 30),
    #             font_face, font_scale,
    #             color, thickness, 2)
    # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    # cv2.imshow(image)
    # cv2.waitKey(0)


    return probability, label, SFDG_path


def SFDG_process(imgpath):
    code = None
    probability = None
    label = None
    SFDG_path = None

    try:
        probability, label, SFDG_path = SFDG_process_primary(imgpath)
        code = 0
        return code, probability, label, SFDG_path
    except:
        code = 1
        return code, probability, label, SFDG_path


if __name__ == '__main__':

    # # 原始处理测试代码，文件夹处理方式，处理该文件夹下的所有图片
    # name = 'Freq2_LBP_Graph_23'
    # attention = False
    # prepare_real = True
    # ckpt = 29 # 29
    # config, net=load_model(name)
    # print('Starting load the SFDG Model: ')
    # state_dict=torch.load('./checkpoints/%s/ckpt_%s.pth'%(name,ckpt))['state_dict']
    # net.load_state_dict(state_dict,strict=False)
    # print('Model Successful load!')
    # net.eval()
    # name_all = os.listdir('./test_img/full_image/')
    # for pic in name_all:
    #     img = data_prepare('./test_img/full_image/'+pic)
    #     print('./test_img/full_image/'+pic)
    #     logit = net(x=img.unsqueeze(0).cuda())
    #     predict_index=torch.max(logit,dim=1)[1]
    #     predict = F.softmax(logit, dim=1)
    #     if predict_index == 0:
    #         print('The test image is predicted as Real and the logit is %.4f' % predict[0,0])
    #     else:
    #         print('The test image is predicted as Fake and the logit is %.4f' % predict[0,1])

    # # 单图测试
    # # imgpath = './test_img/full_image/634_1.png'
    # imgpath = './test_img/full_image/634_1.png'
    # imgpath = '/home/vis/NQI145/testdata/test_20240527091838.jpg'
    # code, probability, label, SFDG_path = SFDG_process(imgpath)
    # print(code, probability, label, SFDG_path)

    # 单图测试
    # imgpath = './test_img/full_image/000_1.png'
    imgpath = './test_img/full_image/001_0.png'
    probability, label, SFDG_path = SFDG_process_primary(imgpath)
    print(probability, label, SFDG_path)
        
