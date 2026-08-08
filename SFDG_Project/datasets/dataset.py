import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.augmentations import augmentations, get_boundingbox, salt_pepper_noise
from albumentations import CenterCrop, Compose, Resize, RandomCrop, Normalize
# from albumentations.pytorch.transforms import ToTensor
from albumentations.pytorch import ToTensorV2
from skimage.feature import local_binary_pattern
import json
import random
import cv2
from datasets.data import *
import dlib

index = ['012', '026', '630', '015', '169', '158', '423', '399','288','321','705','707','306','467','314','347','741','634','445','190','257','420','550','452','047','607','552','000','724','725','220', '219', '141', '429']
#index = ['012', '026', '630', '015', '169', '158', '423', '399','288','321','705','707']
index_neural = ['953_974', '227_169', '319_352', '233_995', '731_741', '706_479', '138_142', '868_949', '249_280', '452_550']
index_faceswap = ['233_995','953_974']
index_face2face = ['227_169','233_995','868_949']

class DeepfakeDataset(Dataset):

    def __init__(self, phase='train', datalabel='', resize=(320, 320), imgs_per_video=30, min_frames=0,
                 normalize=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), frame_interval=10, max_frames=300, augment='augment0'):
        assert phase in ['train', 'valid', 'test']
        self.datalabel = datalabel
        self.phase = phase
        self.imgs_per_video = imgs_per_video
        self.frame_interval = frame_interval
        self.num_classes = 2
        self.epoch = 0
        self.max_frames = max_frames
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/mnt/disk1/wy/shape_predictor_68_face_landmarks.dat")
        if min_frames:
            self.min_frames = min_frames
        else:
            self.min_frames = max_frames*0.3
        # self.min_frames = 40
        self.dataset = []
        self.aug = augmentations[augment]
        resize_ = (int(resize[0]/0.8), int(resize[1]/0.8))
        self.resize = resize_
        # self.trans=Compose([Resize(*resize_,interpolation=cv2.INTER_CUBIC),CenterCrop(*resize),ToTensor(normalize=normalize)])
        self.trans = Compose([Resize(*resize), ToTensorV2(p=1), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        ###############
        # doing resize and center crop in trans
        if type(datalabel) != str:
            self.dataset = datalabel
            return
        if 'ff-5' in self.datalabel:
            for i, j in enumerate(['original_sequences/youtube', 'manipulated_sequences/Deepfakes', 'manipulated_sequences/NeuralTextures',
                                   'manipulated_sequences/FaceSwap', 'manipulated_sequences/Face2Face']):
                temp = FF_dataset(j, self.datalabel.split('-')[2], phase)
                temp = [[k[0], i] for k in temp]
                self.dataset += temp
        elif 'ff-all' in self.datalabel:
            for i in ['original_sequences/youtube', 'manipulated_sequences/Deepfakes', 'manipulated_sequences/NeuralTextures',
                      'manipulated_sequences/FaceSwap', 'manipulated_sequences/Face2Face']:
                self.dataset += FF_dataset(i,
                                           self.datalabel.split('-')[2], phase)
            if phase != 'test':
                self.dataset = make_balance(self.dataset)
        elif 'ff' in self.datalabel:
            self.dataset += FF_dataset(self.datalabel.split('-')[1], self.datalabel.split(
                '-')[2], phase)+FF_dataset('original_sequences/youtube', self.datalabel.split('-')[2], phase)
        elif 'celeb' in self.datalabel:
            self.dataset = Celeb_test
        elif 'deeper' in self.datalabel:
            self.dataset = deeperforensics_dataset(
                phase)+FF_dataset('original_sequences/youtube', self.datalabel.split('-')[1], phase)
        elif 'dfdc' in self.datalabel:
            # if phase == 'train':
            #     self.dataset = DFDC_train(phase)
            # else:
            self.dataset = dfdc_dataset(phase)

        elif 'wild' in self.datalabel:
            if phase == 'test' or phase == 'valid':
                self.dataset = WildDeepfake_test
            else:
                self.dataset = WildDeepfake_train
                self.dataset = make_balance(self.dataset)
        else:
            raise(Exception('no such dateset'))

    def next_epoch(self):
        self.epoch += 1

    def __getitem__(self, item):
        try:
            vid = self.dataset[item//self.imgs_per_video]
            vd = sorted(os.listdir(vid[0]))

            if vid[0].split('/')[-1] in index and 'original_sequences' in vid[0]:
                vd = vd[:150]
            if vid[0].split('/')[-1] in index_neural and 'NeuralTextures' in vid[0]:
                vd = vd[:150]
            if vid[0].split('/')[-1] in index_faceswap and 'FaceSwap' in vid[0]:
               vd = vd[:150]
            if vid[0].split('/')[-1] in index_face2face and 'Face2Face' in vid[0]:
               vd = vd[:150]

            if len(vd) < self.min_frames:
                raise(Exception(str(vid)))
                # return self.__getitem__((item+self.imgs_per_video)%(self.__len__()))
            ind = (item % self.imgs_per_video*self.frame_interval +
                   self.epoch) % min(len(vd), self.max_frames)
            ind = vd[ind]
            image = cv2.imread(os.path.join(vid[0], ind))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector(gray, 1)
            if len(faces):
              # For now only take biggest face
              face = faces[0]
              # --- Prediction ---------------------------------------------------
              # Face crop with dlib and bounding box scale enlargement
              x, y, size = get_boundingbox(face, width, height)
              image = image[y:y + size, x:x + size]
            
              # face_shape = self.predictor(gray, face)
              # print(face_shape.parts())
            else:
               raise IOError('There are no faces in the current image')

            if self.phase == 'train':
                image = self.aug(image=image)['image']
                
            # image = gaussian_blur(image, 9)  # [7,9,13]
            # image = jpeg_compression(image, 6)  # [2, 3, 4, 5, 6]
            # image = gaussian_noise_color(image, param)  # [0.001, 0.002, 0.005, 0.01, 0.05]
            # image = color_saturation(img, param)  # [0.4, 0.3, 0.2, 0.1, 0.0]
            # image = color_contrast(img, param)  # [0.85, 0.725, 0.6, 0.475, 0.35]
            # image = block_wise(img, param)  # [16, 32, 48, 64, 80]
            # image = salt_pepper_noise(image, 0.01)

            return self.trans(image=image)['image'], vid[1]

        except IOError as e:
            # if self.phase!='test':
                # return self.__getitem__((item+self.imgs_per_video)%(self.__len__()))
            return self.__getitem__((item+self.imgs_per_video)%(self.__len__()))
            # else:
            #     return torch.zeros(3, self.resize[0], self.resize[1]), -1

        except Exception as e:
            # print(e)
            # raise(e)
            # if self.phase!='test':
            return self.__getitem__((item+self.imgs_per_video) % (self.__len__()))
            # else:
            #     return torch.zeros(3,self.resize[0],self.resize[1]),-1

    def __len__(self):
        return len(self.dataset)*self.imgs_per_video
