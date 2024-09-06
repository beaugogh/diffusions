# coding: utf-8
import os
import os.path as osp
import sys

current_dir = osp.dirname(os.path.abspath(__file__))
parent_dir = osp.join(current_dir, "..")
portrait_lighting_dir = osp.join(parent_dir, "PortraitLighting")
trained_model_dir = osp.join(portrait_lighting_dir, "trained_model")
asset_dir = osp.join(parent_dir, "assets")
sys.path.insert(0, parent_dir)

import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch
import time
import cv2
from PortraitLighting.utils.utils_SH import *
import PortraitLighting.defineHourglass_512_gray_skip as pl_hourglass_512
import PortraitLighting.defineHourglass_1024_gray_skip_matchFeature as pl_hourglass_1024
from utils import *


def generate_2d_normals(begin_angle, end_angle, clockwise=True, num=10):
    """
             (b)
              +
              |
              |
    ---------- ----------+ (a) (ref angle)
              |
              |
    """
    arr = np.linspace(begin_angle, end_angle, num=num)
    sign = -1 if clockwise else 1
    pairs = []
    for i in arr:
        a = np.cos(i)
        b = np.sin(i) * sign
        pairs.append((a, b))

    return pairs


def generate_3d_normals(begin_angle, end_angle, clockwise=True, num=10, null_dim="y"):
    pairs = generate_2d_normals(begin_angle, end_angle, clockwise, num)
    result = []
    for a, b in pairs:
        if null_dim == "y":
            result.append((a, 0, b))
        elif null_dim == "x":
            result.append((0, a, b))
        elif null_dim == "z":
            result.append((a, b, 0))
        else:
            raise Exception("null_dim is ill defined")

    return np.reshape(result, (-1, 3))


def generate_lighting():
    output_dir = osp.join(asset_dir, "portraitlighting", "lighting_4")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #############################################################
    # lightings_1
    # num_points = 240
    # normals_xy = generate_3d_normals(0.5*np.pi, -1.5*np.pi, clockwise=True, num=num_points, null_dim='z')
    # normals_yz = generate_3d_normals(-1*np.pi, 0.5*np.pi, clockwise=True, num=num_points, null_dim='x')
    # normals_xz = generate_3d_normals(0.5*np.pi, 2*np.pi, clockwise=True, num=num_points, null_dim='y')
    # normals = np.concatenate((normals_xy, normals_yz, normals_xz))

    # lightings_2
    # num_points = 120
    # normals = generate_3d_normals(np.pi, -1*np.pi, clockwise=True, num=num_points, null_dim='z')

    # lightings_3
    # num_points = 120
    # normals = generate_3d_normals(-1*np.pi, np.pi, clockwise=True, num=num_points, null_dim='x')

    # lightings_4
    num_points = 120
    normals = generate_3d_normals(
        -0.5 * np.pi, 1.5 * np.pi, clockwise=True, num=num_points, null_dim="y"
    )
    ############################################################
    shs = SH_basis(normals)
    for i, sh in tqdm(enumerate(shs)):
        out_txt = osp.join(output_dir, "rotate_light_{:02d}.txt".format(i))
        np.savetxt(out_txt, sh, delimiter="\n")


def load_model_512():
    my_network = pl_hourglass_512.HourglassNet()
    my_network.load_state_dict(
        torch.load(osp.join(trained_model_dir, "trained_model_03.t7"))
    )
    my_network.cuda()
    my_network.train(False)
    return my_network


def load_model_1024():
    my_network_512 = pl_hourglass_1024.HourglassNet(16)
    my_network = pl_hourglass_1024.HourglassNet_1024(my_network_512, 16)
    my_network.load_state_dict(
        torch.load(osp.join(trained_model_dir, "trained_model_1024_03.t7"))
    )
    my_network.cuda()
    my_network.train(False)
    return my_network


def _generate(model, Lab, inputL, col, row, sh, model_size="S"):
    sh = sh[0:9]
    sh = sh * 0.7
    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    if model_size == "S":
        outputImg, outputSH = model(inputL, sh, 0)
    else:
        outputImg, _, outputSH, _ = model(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg * 255.0).astype(np.uint8)
    Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))
    return resultLab


def generate_portrait_lighting(img_path, light_dir_or_path, output_dir, model_size="S"):
    img_name = osp.basename(img_path).split(".")[0]
    num_lightings = len(os.listdir(light_dir_or_path)) if osp.isdir(light_dir_or_path) else 1

    # load model
    if model_size == "S":
        my_network = load_model_512()
        img_size = 512
    elif model_size == "L":
        my_network = load_model_1024()
        img_size = 1024
    else:
        raise Exception("unsupported model size")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(img_path)
    row, col, _ = img.shape
    img = cv2.resize(img, (img_size, img_size))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:, :, 0]
    inputL = inputL.astype(np.float32) / 255.0
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]
    inputL = Variable(torch.from_numpy(inputL).cuda())
    if osp.isdir(light_dir_or_path):
        light_dir = light_dir_or_path
        for i in tqdm(range(num_lightings)):
            sh = np.loadtxt(osp.join(light_dir, "rotate_light_{:02d}.txt".format(i)))
            resultLab = _generate(my_network, Lab, inputL, col, row, sh, model_size)
            cv2.imwrite(
                osp.join(output_dir, "{}_{:02d}.png".format(img_name, i)), resultLab
            )
    else:
        sh = np.loadtxt(light_dir_or_path)
        resultLab = _generate(my_network, Lab, inputL, col, row, sh, model_size)
        cv2.imwrite(osp.join(output_dir, f"{img_name}.png", resultLab))


if __name__ == "__main__":
    # step 1: generate lighting
    # generate_lighting()

    # step 2: generate images according to the lighting
    img_path = "/home/bo/workspace/diffusions/assets/liveportrait/source/han.jpg"
    light_dir = "/home/bo/workspace/diffusions/PortraitLighting/data/example_light"
    # light_dir = "/home/bo/workspace/diffusions/assets/portraitlighting/lightings_2"
    output_dir = "/home/bo/workspace/diffusions/assets/portraitlighting/results1024"
    generate_portrait_lighting(img_path, light_dir, output_dir, model_size="L")

    # step 3: compose images back to video
    # images_to_video(input_images_dir="/home/bo/workspace/diffusions/assets/portraitlighting/results1024")
    print("finished")
