import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

import importlib
data_loader = importlib.import_module(".data_loader", "U-2-Net")
RescaleT = data_loader.RescaleT
ToTensorLab = data_loader.ToTensorLab
SalObjDataset = data_loader.SalObjDataset

model = importlib.import_module(".model", "U-2-Net")
U2NET = model.U2NET # full size version 173.6 MB
U2NETP = model.U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map

from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

import time
import datetime

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def masked_output(image_name,pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)[:, :, :3]
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    masked_imo = Image.fromarray(image)
    masked_imo.putalpha(imo.convert('L'))
    '''
    mask = np.where(np.array(imo.convert('RGB')) > 0, True, False)
    masked_image = mask * image
    masked_imo = Image.fromarray(masked_image.astype(np.uint8)) 
    '''
    return masked_imo


def driver_with_scroll(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')     
    driver = webdriver.Chrome('./chromedriver', chrome_options=options)
    driver.get(url)

    height = driver.execute_script("return document.body.scrollHeight")
    n_times = 1000
    total_height = 0
  
    for i in range(2):
        for j in range(1, n_times):
            total_height += height
            driver.execute_script("window.scrollTo(0, " + str(total_height) + ");")
        if i == 1: continue
        time.sleep(1) # ボタンが表示される間
        driver.find_elements_by_xpath('//input')[-1].click()    
    return driver


def main(q, min_width, max_n_img=100):
    url = f'https://www.google.com/search?q={q}&tbm=isch'
    driver = driver_with_scroll(url)

    img_url_list = []
    for tag in driver.find_elements_by_tag_name('img'):
        if tag.rect['width'] < min_width: continue
        src = tag.get_attribute('src')
        if src is None or 'https' not in src: continue
        img_url_list.append(src)
        if len(img_url_list) == max_n_img: break

    model_name='u2net'#u2netp
    dt_now = datetime.datetime.now().isoformat()
    model_dir = f'./{model_name}.pth'

    img_name_list = img_url_list

    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    if(model_name=='u2net'): net = U2NET(3,1)
    elif(model_name=='u2netp'): net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    images = []
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        try:
            imo = masked_output(img_name_list[i_test],pred)
            images.append(imo)
        except:
            continue
    
        del d1,d2,d3,d4,d5,d6,d7
    return images

