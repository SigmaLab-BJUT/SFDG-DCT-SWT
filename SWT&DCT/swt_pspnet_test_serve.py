import warnings

import os
import cv2
import numpy as np
from PIL import Image
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from torchvision import transforms
import torch
from model.PSPEnet_swt import PSPENet_b4
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
from pywt import swt2
from faceUtil.face_utils import FaceDetector, norm_crop
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getSWT(img):
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb_image)
    y = y.astype(np.float64)

    (cA4, (cH4, cV4, cD4)), (cA3, (cH3, cV3, cD3)), (cA2, (cH2, cV2, cD2)), (cA1, (cH1, cV1, cD1)) = swt2(y, 'bior2.4',
                                                                                                          level=4)

    # 小波变换之后，低频分量对应的图像：
    # a = np.uint8(cA / np.max(cA) * 255)
    # 小波变换之后，水平方向高频分量对应的图像：
    b1 = np.uint8(cH1 / np.max(cH1) * 255)
    b2 = np.uint8(cH2 / np.max(cH2) * 255)
    b3 = np.uint8(cH3 / np.max(cH3) * 255)
    b4 = np.uint8(cH4 / np.max(cH4) * 255)
    # 小波变换之后，垂直平方向高频分量对应的图像：
    c1 = np.uint8(cV1 / np.max(cV1) * 255)
    c2 = np.uint8(cV2 / np.max(cV2) * 255)
    c3 = np.uint8(cV3 / np.max(cV3) * 255)
    c4 = np.uint8(cV4 / np.max(cV4) * 255)
    # 小波变换之后，对角线方向高频分量对应的图像：
    d1 = np.uint8(cD1 / np.max(cD1) * 255)
    d2 = np.uint8(cD2 / np.max(cD2) * 255)
    d3 = np.uint8(cD3 / np.max(cD3) * 255)
    d4 = np.uint8(cD4 / np.max(cD4) * 255)

    b1=transforms_(b1)
    b2=transforms_(b2)
    b3=transforms_(b3)
    b4=transforms_(b4)
    c1=transforms_(c1)
    c2=transforms_(c2)
    c3=transforms_(c3)
    c4=transforms_(c4)
    d1=transforms_(d1)
    d2=transforms_(d2)
    d3=transforms_(d3)
    d4=transforms_(d4)

    swttensor = torch.cat(
        (b1, c1, d1, b2, c2, d2, b3, c3, d3, b4, c4, d4), 0);

    return swttensor


def swt_process_primary(imgpath):
    modelpath = "model_weight/swt_pspnet.pt"


    img = cv2.imread(imgpath)

    result_img_name = os.path.basename(imgpath).split('.')[0]

    face_detector = FaceDetector()
    face_detector.load_checkpoint("faceUtil/pre_model_weight/RetinaFace-Resnet50-fixed.pth")

    boxes, landms = face_detector.detect(img)
    if boxes.shape[0] == 0:
        print('未检测出人脸，任务结束。')
        sys.exit()
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    max_face_idx = areas.argmax()
    landm = landms[max_face_idx]

    landmarks = landm.detach().numpy().reshape(5, 2).astype(np.int32)
    img = norm_crop(img, landmarks, image_size=320)
    # plt.figure("original_face")  # 图像窗口名称
    # plt.imshow(img)
    # plt.axis('off')  # 关掉坐标轴为 off
    # plt.title('face')  # 图像题目
    # plt.show()

    original_path = os.path.join('../result/swt', result_img_name + '_original.jpg')
    os.makedirs(os.path.dirname(original_path), exist_ok=True)
    image_save = Image.fromarray(np.uint8(img))
    image_save.save(original_path)

    # 加载的模型的地址

    # print('swt-pspnet Job Start')

    model = PSPENet_b4(num_classes=2)

    model = nn.DataParallel(model)  # 在加载时需要把网络也转成DataParallel的
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(modelpath), strict=False)

    model.eval()

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_tensor = transforms_(img).unsqueeze(0)
    swt_tensor = getSWT(img).unsqueeze(0)

    with torch.no_grad():
        inputimg = img_tensor.to(device)
        inputswt = swt_tensor.to(device)
        classout, maskout = model(inputimg, inputswt)

        softmask = torch.softmax(maskout, 1)
        softclass = torch.softmax(classout, 1)

        _, maskpred = torch.max(maskout.data, 1)
        _, classpred = torch.max(classout.data, 1)

        maskpred = maskpred.squeeze(0)
        # # 输出2：伪造区域的预测
        # plt.figure("mask_prediction")  # 图像窗口名称
        # plt.imshow(maskpred.cpu().numpy())
        # plt.axis('off')  # 关掉坐标轴为 off
        # plt.title('mask_prediction')  # 图像题目
        # plt.show()

        predval = softclass.cpu().numpy()
        # # 输出1：图像为真的概率值
        # print('图像为真的概率为%.4f' % predval[0][1])
        # # 最终预测结果打印输出：真伪预测值，0为假，1为真
        # print('图像最终预测为：', classpred.cpu().item())

        if classpred.cpu().item() == 0:
            label = 'fake'
        else:
            label = 'real'

        probability = 1 - predval[0][1]
        swt_path = os.path.join('../result/swt', result_img_name + '.jpg')
        os.makedirs(os.path.dirname(swt_path), exist_ok=True)
        image_save = Image.fromarray(np.uint8(maskpred.cpu().numpy() * 255))
        image_save.save(swt_path)


    return probability, label, swt_path


def swt_process(imgpath):
    code = None
    probability = None
    label = None
    swt_path = None

    try:
        probability, label,swt_path = swt_process_primary(imgpath)
        code = 0
        torch.cuda.empty_cache()
        return code, probability, label, swt_path
    except Exception as e:
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        code = 1
        torch.cuda.empty_cache()
        return code, probability, label, swt_path



if __name__ == "__main__":
    # 输入图片的地址
    # 测试图片的路径:test_image/swt_pspnet,里面真的9张，假的9张


    # modelpath = "model_weight/swt_pspnet.pt"

    # imgpath = 'test_image/swt/000_003-36-fake.jpg'


    # img = cv2.imread(imgpath)
    #
    # face_detector = FaceDetector()
    # face_detector.load_checkpoint("faceUtil/pre_model_weight/RetinaFace-Resnet50-fixed.pth")
    #
    # boxes, landms = face_detector.detect(img)
    # if boxes.shape[0] == 0:
    #     print('未检测出人脸，任务结束。')
    #     sys.exit()
    # areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # max_face_idx = areas.argmax()
    # landm = landms[max_face_idx]
    #
    # landmarks = landm.detach().numpy().reshape(5, 2).astype(np.int)
    # img = norm_crop(img, landmarks, image_size=320)
    # plt.figure("original_face")  # 图像窗口名称
    # plt.imshow(img)
    # plt.axis('off')  # 关掉坐标轴为 off
    # plt.title('face')  # 图像题目
    # plt.show()
    #
    # # 加载的模型的地址
    #
    # print('swt-pspnet Job Start')
    #
    # model = PSPENet_b4(num_classes=2)
    #
    # model = nn.DataParallel(model)  # 在加载时需要把网络也转成DataParallel的
    # device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    # model.to(device)
    #
    # model.load_state_dict(torch.load(modelpath), strict=False)
    #
    # model.eval()
    #
    # transforms_ = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    #
    #
    # img_tensor = transforms_(img).unsqueeze(0)
    # swt_tensor = getSWT(img).unsqueeze(0)
    #
    #
    # with torch.no_grad():
    #     inputimg = img_tensor.to(device)
    #     inputswt = swt_tensor.to(device)
    #     classout, maskout = model(inputimg,inputswt)
    #
    #     softmask = torch.softmax(maskout, 1)
    #     softclass = torch.softmax(classout, 1)
    #
    #
    #     _, maskpred = torch.max(maskout.data, 1)
    #     _, classpred = torch.max(classout.data, 1)
    #
    #     maskpred =maskpred.squeeze(0)
    #     # 输出2：伪造区域的预测
    #     plt.figure("mask_prediction")  # 图像窗口名称
    #     plt.imshow(maskpred.cpu().numpy())
    #     plt.axis('off')  # 关掉坐标轴为 off
    #     plt.title('mask_prediction')  # 图像题目
    #     plt.show()
    #
    #     predval = softclass.cpu().numpy()
    #     # 输出1：图像为真的概率值
    #     print('图像为真的概率为%.4f' %predval[0][1])
    #     # 最终预测结果打印输出：真伪预测值，0为假，1为真
    #     print('图像最终预测为：', classpred.cpu().item())
    #
    #
    #
    # print('swt-pspnet Job End')
    # sys.exit()
    #输出的图片的地址

    # imgpath = 'test_image/swt/000_003-36-fake.jpg'
    # probability, label, swt_path = swt_process_primary(imgpath)
    # print(probability, label, swt_path)

    # imgpath = 'test_image/swt/000_003-36-fake.jpg'
    imgpath = './test_image/asiaFace/swt-pspnet-DCT-Unet/real/real2.jpg'
    code, probability, label, swt_path = swt_process(imgpath)
    print(code, probability, label, swt_path)

