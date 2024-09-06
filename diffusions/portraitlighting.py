# coding: utf-8
import os
import os.path as osp
import sys

current_dir = osp.dirname(os.path.abspath(__file__))
parent_dir = osp.join(current_dir, "..")
portrait_lighting_dir = osp.join(parent_dir, "PortraitLighting")
trained_model_dir = os.path.join(portrait_lighting_dir, "trained_model")
sys.path.insert(0, parent_dir)

import numpy as np
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
from PortraitLighting.utils.utils_SH import *
from PortraitLighting.defineHourglass_512_gray_skip import *


def main():
    # /home/bo/workspace/diffusions/assets/liveportrait/source/s5.jpg
    img_name = "s5"
    img_path = "/home/bo/workspace/diffusions/assets/liveportrait/source/han.jpg"
    lightFolder = "/home/bo/workspace/diffusions/PortraitLighting/data/example_light"
    output_dir = "/home/bo/workspace/diffusions/assets/portraitlighting/results"
    num_lightings = len(os.listdir(lightFolder))

    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    # -----------------------------------------------------------------

    # load model
    modelFolder = trained_model_dir
    my_network = HourglassNet()
    my_network.load_state_dict(
        torch.load(os.path.join(modelFolder, "trained_model_03.t7"))
    )
    my_network.cuda()
    my_network.train(False)

    saveFolder = output_dir
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    img = cv2.imread(img_path)
    row, col, _ = img.shape
    img = cv2.resize(img, (512, 512))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:, :, 0]
    inputL = inputL.astype(np.float32) / 255.0
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    for i in range(num_lightings):
        print(f"rendering {i}...")
        sh = np.loadtxt(os.path.join(lightFolder, "rotate_light_{:02d}.txt".format(i)))
        sh = sh[0:9]
        sh = sh * 0.7

        # --------------------------------------------------
        # rendering half-sphere
        sh = np.squeeze(sh)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        shading = (shading *255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid
        cv2.imwrite(os.path.join(saveFolder, \
                'light_{:02d}.png'.format(i)), shading)
        # --------------------------------------------------

        # ----------------------------------------------
        #  rendering images using the network
        sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())
        outputImg, outputSH = my_network(inputL, sh, 0)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1, 2, 0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg * 255.0).astype(np.uint8)
        Lab[:, :, 0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        cv2.imwrite(
            os.path.join(saveFolder, "{}_{:02d}.jpg".format(img_name, i)), resultLab
        )
        # ----------------------------------------------


if __name__ == "__main__":
    main()
