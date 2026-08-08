# 从 108 下载的代码，这里修改代码， 根据预测概率 改变 概率图输出
import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from torchvision import transforms
import torch
from model.efficientnet_twostream import efficientnet_b4
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from faceUtil.face_utils import FaceDetector, norm_crop
import cv2
import os
import numpy as np
# import tqdm
from skimage import data, exposure, img_as_float
# from sklearn import preprocessing


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

face_detector = FaceDetector()
face_detector.load_checkpoint("faceUtil/pre_model_weight/RetinaFace-Resnet50-fixed.pth")


def getDCT(imgpath):
    dctmask = np.ones(shape=(320, 320)) * 255
    for i in range(320):
        for j in range(320):
            if i + j <= 106:
                dctmask[i][j] = 0


    img = cv2.imread(imgpath)
    img = cv2.resize(img,(320,320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb_image)
    y = y.astype(np.float64)
    y_dct = cv2.dct(y)
    y_idct = cv2.idct(y_dct * dctmask)
    idctimge = Image.fromarray(y_idct)
    return idctimge



def getDCT_face(img):
    dctmask = np.ones(shape=(320, 320)) * 255
    for i in range(320):
        for j in range(320):
            if i + j <= 106:
                dctmask[i][j] = 0


    # img = cv2.imread(imgpath)
    # img = cv2.resize(img,(320,320))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb_image)
    y = y.astype(np.float64)
    y_dct = cv2.dct(y)
    y_idct = cv2.idct(y_dct * dctmask)
    idctimge = Image.fromarray(y_idct)

    # image_array = np.array(idctimge)
    # brightness_factor = 1 # 亮度调整因子，大于1增加亮度，小于1降低亮度  1.5  4.172325134277344e-06
    # adjusted_image_array = image_array * brightness_factor
    # adjusted_image = Image.fromarray(adjusted_image_array)
    # if adjusted_image.mode == "F":
    #     adjusted_image = adjusted_image.convert('RGB')
    # adjusted_image.save('./result/dct/output_fake1.jpg')

    return idctimge


def getDCT_face_new_old(img):
    for ratio in range(40):
        dctmask = np.ones(shape=(320, 320)) * 255
        for i in range(320):
            for j in range(320):
                if i + j <= 106 * ratio * 0.05:
                    dctmask[i][j] = 0


        # img = cv2.imread(imgpath)
        # img = cv2.resize(img,(320,320))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv2.split(ycrcb_image)
        y = y.astype(np.float64)
        y_dct = cv2.dct(y)
        y_idct = cv2.idct(y_dct * dctmask)
        idctimge = Image.fromarray(y_idct)

        image_array = np.array(idctimge)
        brightness_factor = 1 # 亮度调整因子，大于1增加亮度，小于1降低亮度  1.5  4.172325134277344e-06
        adjusted_image_array = image_array * brightness_factor
        adjusted_image = Image.fromarray(adjusted_image_array)
        if adjusted_image.mode == "F":
            adjusted_image = adjusted_image.convert('RGB')
        print(ratio * 0.05)
        # adjusted_image.save('./result/dct/output_fake1_{}.jpg'.format(ratio * 0.05))
        adjusted_image.save('./result/dct/{}.jpg'.format(ratio))

    return idctimge


def getDCT_face_new(img):
    # for ratio in range(40):
    #     dctmask = np.ones(shape=(320, 320)) * 255
    #     for i in range(320):
    #         for j in range(320):
    #             if i + j <= 106 * ratio * 0.05:   #真的 39.jpg 纹理特征少，假的 19 纹理特征多；
    #                 dctmask[i][j] = 0
    #
    #
    #     # img = cv2.imread(imgpath)
    #     # img = cv2.resize(img,(320,320))
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #     (y, cr, cb) = cv2.split(ycrcb_image)
    #     y = y.astype(np.float64)
    #     y_dct = cv2.dct(y)
    #     y_idct = cv2.idct(y_dct * dctmask)
    #     idctimge = Image.fromarray(y_idct)
    #
    #     image_array = np.array(idctimge)
    #     brightness_factor = 1 # 亮度调整因子，大于1增加亮度，小于1降低亮度  1.5  4.172325134277344e-06
    #     adjusted_image_array = image_array * brightness_factor
    #
    #     # # 创建归一化器
    #     # normalizer = preprocessing.MinMaxScaler()
    #     # # 对矩阵进行归一化
    #     # normalized_matrix = normalizer.fit_transform(adjusted_image_array)
    #     # normalized_matrix *= 255
    #     # adjusted_image = Image.fromarray(normalized_matrix)
    #
    #     # # 计算每列的最小值和最大值   按照固定维度 axis=0归一化
    #     # min_vals = np.min(adjusted_image_array, axis=0)
    #     # max_vals = np.max(adjusted_image_array, axis=0)
    #     # # 归一化矩阵
    #     # normalized_matrix = (adjusted_image_array - min_vals) / (max_vals - min_vals)
    #     # normalized_matrix *= 255
    #     # adjusted_image = Image.fromarray(normalized_matrix)
    #
    #     # 计算矩阵的最大最小值，全局归一化
    #     min_vals = np.min(adjusted_image_array)
    #     max_vals = np.max(adjusted_image_array)
    #     # 归一化矩阵
    #     normalized_matrix = (adjusted_image_array - min_vals) / (max_vals - min_vals)
    #     normalized_matrix *= 255
    #     adjusted_image = Image.fromarray(normalized_matrix)
    #
    #     # adjusted_image = Image.fromarray(adjusted_image_array)
    #     if adjusted_image.mode == "F":
    #         adjusted_image = adjusted_image.convert('RGB')
    #     print(ratio * 0.05)
    #     # adjusted_image.save('./result/dct/output_fake1_{}.jpg'.format(ratio * 0.05))
    #     adjusted_image.save('./result/dct/{}.jpg'.format(ratio))

    dctmask = np.ones(shape=(320, 320)) * 255
    for i in range(320):
        for j in range(320):
            if i + j <= 106:  # 真的 39.jpg 纹理特征少，假的 19 纹理特征多；
                dctmask[i][j] = 0

    # img = cv2.imread(imgpath)
    # img = cv2.resize(img,(320,320))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb_image)
    y = y.astype(np.float64)
    y_dct = cv2.dct(y)
    y_idct = cv2.idct(y_dct * dctmask)
    idctimge = Image.fromarray(y_idct)


    return idctimge


def getDCT_face_new_toshow(img, prob, savepath):

    if prob >= 0.5:
        prob = 1 + prob/10
    else:
        prob = 39 * 0.05 + prob / 10   # 35
    dctmask = np.ones(shape=(320, 320)) * 255
    for i in range(320):
        for j in range(320):
            if i + j <= 106 * prob:   #真的 39.jpg 纹理特征少，假的 19 纹理特征多；
                dctmask[i][j] = 0


    # img = cv2.imread(imgpath)
    # img = cv2.resize(img,(320,320))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb_image)
    y = y.astype(np.float64)
    y_dct = cv2.dct(y)
    y_idct = cv2.idct(y_dct * dctmask)
    idctimge = Image.fromarray(y_idct)

    image_array = np.array(idctimge)
    brightness_factor = 1 # 亮度调整因子，大于1增加亮度，小于1降低亮度  1.5  4.172325134277344e-06
    adjusted_image_array = image_array * brightness_factor

    # # 创建归一化器
    # normalizer = preprocessing.MinMaxScaler()
    # # 对矩阵进行归一化
    # normalized_matrix = normalizer.fit_transform(adjusted_image_array)
    # normalized_matrix *= 255
    # adjusted_image = Image.fromarray(normalized_matrix)

    # # 计算每列的最小值和最大值   按照固定维度 axis=0归一化
    # min_vals = np.min(adjusted_image_array, axis=0)
    # max_vals = np.max(adjusted_image_array, axis=0)
    # # 归一化矩阵
    # normalized_matrix = (adjusted_image_array - min_vals) / (max_vals - min_vals)
    # normalized_matrix *= 255
    # adjusted_image = Image.fromarray(normalized_matrix)

    # 计算矩阵的最大最小值，全局归一化
    min_vals = np.min(adjusted_image_array)
    max_vals = np.max(adjusted_image_array)
    # 归一化矩阵
    normalized_matrix = (adjusted_image_array - min_vals) / (max_vals - min_vals)
    normalized_matrix *= 255
    adjusted_image = Image.fromarray(normalized_matrix)

    # adjusted_image = Image.fromarray(adjusted_image_array)
    if adjusted_image.mode == "F":
        adjusted_image = adjusted_image.convert('RGB')
    # adjusted_image.save('./result/dct/output_fake1_{}.jpg'.format(ratio * 0.05))
    # adjusted_image.save('../result/dct/{}.jpg'.format(prob))
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    adjusted_image.save(savepath)



def dct_process_primary(imgpath):
    # 加载的模型的地址
    modelpath = "model_weight/dct.pt"


    # print('DCT Job Start')

    model = efficientnet_b4(num_classes=2)
    bestmodel = torch.load(modelpath)
    model.load_state_dict(bestmodel)

    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 切割人脸
    img = cv2.imread(imgpath)
    if img is None:
        raise FileNotFoundError(f"无法读取图片文件: {imgpath}")
    # img = Image.open(imgpath)
    # print(img.shape)

    result_img_name = os.path.basename(imgpath).split('.')[0]

    face_detector = FaceDetector()
    face_detector.load_checkpoint("faceUtil/pre_model_weight/RetinaFace-Resnet50-fixed.pth")

    boxes, landms = face_detector.detect(img)
    if boxes.shape[0] == 0:
        print('未检测出人脸，任务结束。')
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    max_face_idx = areas.argmax()
    landm = landms[max_face_idx]

    landmarks = landm.detach().numpy().reshape(5, 2).astype(np.int32)
    img = norm_crop(img, landmarks, image_size=320)
    # print(img.shape)
    # print(type(img))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # plt.figure("img")  # 图像窗口名称
    # plt.imshow(img)
    # plt.axis('off')  # 关掉坐标轴为 off
    # plt.title('img')  # 图像题目
    # plt.show()

    # dct_path = os.path.join('../result/dct', imgpath.split('/')[-1].split('.')[0] + '_dct.png')
    dct_path = os.path.join('../result/dct', imgpath.split('/')[-1].split('.')[0] + '_dct.png')
    # print(dct_path)
    # plt.savefig(dct_path)

    # img = Image.open(imgpath)
    # img = img.resize((320, 320))



    #输出DCT特征图
    # dctimg = getDCT(imgpath)
    # dctimg = getDCT_face(img)
    dctimg = getDCT_face_new(img)


    img_tensor = transforms_(img).unsqueeze(0)
    dct_tensor = transforms_(dctimg).unsqueeze(0)

    input = img_tensor.to(device)
    idct = dct_tensor.to(device)

    with torch.no_grad():
        classout = model(input, idct)

        softclass = torch.softmax(classout, 1)

        _, classpred = torch.max(classout.data, 1)

        # # 输出2：展示DCT的特征图像
        # plt.figure("DCT_Image")  # 图像窗口名称
        # plt.imshow(dctimg)
        # plt.axis('off')  # 关掉坐标轴为 off

        # plt.title('DCT')  # 图像题目
        # plt.show()

        # dctimg_gam = img_as_float(dctimg)
        # gam1 = exposure.adjust_gamma(dctimg_gam, 2)  # 调暗
        # gam2 = exposure.adjust_gamma(dctimg_gam, 0.5)  # 调亮
        #
        # plt.subplot(131)
        # plt.imshow(dctimg)
        # plt.axis('off')
        #
        # plt.subplot(132)
        # plt.imshow(gam1, plt.cm.gray)
        # plt.axis('off')
        #
        # plt.subplot(133)
        # plt.imshow(gam2, plt.cm.gray)
        # plt.axis('off')


        predval = softclass.cpu().numpy()
        # 输出1：图像为真的概率值
        # print('图像最终预测为：', classpred.cpu().item())
        # print('图像为真的概率为%.4f' % predval[0][1])
        # dct_path = os.path.join('./result/dct', imgpath.split('/')[-1].split('.')[0] + '_dct.png')
        # print(dct_path)
        # plt.savefig(dct_path)

        # 最终预测结果打印输出：真伪预测值，0为假，1为真
        probability = 1 - predval[0][1]  # 为假的概率
        if classpred.cpu().item() == 0:
            label = 'fake'
        else:
            label = 'real'

        # print('Unet Job End')
        # sys.exit()
        # 输出的图片的地址
        getDCT_face_new_toshow(img, probability, dct_path)
        return probability, label, dct_path


def dct_process(imgpath):
    code = None
    probability = None
    label = None
    dct_path = None

    try:
        probability, label, dct_path = dct_process_primary(imgpath)
        code = 0
        torch.cuda.empty_cache()
        return code, probability, label, dct_path
    except:
        code = 1
        torch.cuda.empty_cache()
        return code, probability, label, dct_path




if __name__ == "__main__":
    # 输入图片的地址
    # 真实图片
    # imgpath = "test_image/real.jpg"

    # 伪造图片
    # imgpath = "test_image/unet/fake7.jpg"
    # imgpath = "test_image/test/3.jpg"
    # imgpath = 'test_image/hu/image318.jpg'
    # imgpath = '/hu/frames/test_liu/image174.jpg'
    # imgpath = '/hu/frames/test_liu/image473.jpg'
    # imgpath = r'G:\pyproject\national_pro\Deepfake_new\hu\frames\test_liu\image473.jpg'


    # # 加载的模型的地址
    # modelpath = "model_weight/dct.pt"
    #
    #
    # print('DCT Job Start')
    #
    # model = efficientnet_b4(num_classes=2)
    # bestmodel = torch.load(modelpath)
    # model.load_state_dict(bestmodel)
    #
    # device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    # model.to(device)
    #
    # model.eval()
    #
    # transforms_ = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    #
    # # 切割人脸
    # img = cv2.imread(imgpath)
    # # img = Image.open(imgpath)
    # print(img.shape)
    #
    # result_img_name = os.path.basename(imgpath).split('.')[0]
    #
    # face_detector = FaceDetector()
    # face_detector.load_checkpoint("faceUtil/pre_model_weight/RetinaFace-Resnet50-fixed.pth")
    #
    # boxes, landms = face_detector.detect(img)
    # if boxes.shape[0] == 0:
    #     print('未检测出人脸，任务结束。')
    # areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # max_face_idx = areas.argmax()
    # landm = landms[max_face_idx]
    #
    # landmarks = landm.detach().numpy().reshape(5, 2).astype(np.int)
    # img = norm_crop(img, landmarks, image_size=320)
    # print(img.shape)
    # print(type(img))
    #
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # plt.figure("img")  # 图像窗口名称
    # plt.imshow(img)
    # plt.axis('off')  # 关掉坐标轴为 off
    # plt.title('img')  # 图像题目
    # plt.show()
    #
    # # img = Image.open(imgpath)
    # # img = img.resize((320, 320))
    #
    #
    #
    # #输出DCT特征图
    # # dctimg = getDCT(imgpath)
    # dctimg = getDCT_face(img)
    #
    # img_tensor = transforms_(img).unsqueeze(0)
    # dct_tensor = transforms_(dctimg).unsqueeze(0)
    #
    # input = img_tensor.to(device)
    # idct = dct_tensor.to(device)
    #
    # with torch.no_grad():
    #     classout = model(input, idct)
    #
    #     softclass = torch.softmax(classout, 1)
    #
    #     _, classpred = torch.max(classout.data, 1)
    #
    #     # 输出2：展示DCT的特征图像
    #     plt.figure("DCT_Image")  # 图像窗口名称
    #     plt.imshow(dctimg)
    #     plt.axis('off')  # 关掉坐标轴为 off
    #     plt.title('DCT')  # 图像题目
    #     plt.show()
    #
    #     predval = softclass.cpu().numpy()
    #     # 输出1：图像为真的概率值
    #     print('图像最终预测为：', classpred.cpu().item())
    #     print('图像为真的概率为%.4f' % predval[0][1])
    #
    #     # 最终预测结果打印输出：真伪预测值，0为假，1为真
    # print('DCT Job End')
    # sys.exit()

    # # imgpath = "test_image/unet/fake7.jpg"
    # imgpath = "./test_image/asiaFace/swt-pspnet-DCT-Unet/fake/fake2.jpg"
    # imgpath = "./test_image/asiaFace/swt-pspnet-DCT-Unet/real/real2.jpg"

    imgpath = './test_image/IMG_0003.jpg'

    probability, label, dct_path = dct_process_primary(imgpath)
    print(probability, label, dct_path)

    #单个文件处理
    # imgpath = "test_image/unet/fake7.jpg"
    # imgpath = "./test_image/asiaFace/swt-pspnet-DCT-Unet/real/real1.jpg"
    # code, probability, label, dct_path = dct_process(imgpath)
    # print(code, probability, label, dct_path)

    # # 文件夹处理
    # real_num = 0
    # fake_num = 0
    # folder_path = './hu_test_0817/t2'
    # imgList = os.listdir(folder_path)
    # for img in imgList:
    #     imgpath = os.path.join(folder_path, img)
    #     # print(imgpath)
    #     code, probability, label, dct_path = dct_process(imgpath)
    #     print(code, probability, label, dct_path)
    #     if label == 'real':
    #         real_num += 1
    #     elif label == 'fake':
    #         fake_num += 1
    #     else:
    #         pass
    #
    # print('It is the end!')
    # print('real_num is :{}, fake_num is :{}, the total is :{}.'.format(real_num, fake_num, real_num + fake_num))
