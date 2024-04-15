#!/usr/bin/env python

import cv2
from PIL import Image


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.functional as metrics
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder



trans = transforms.Compose([
                    transforms.Resize((64,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ])

### 모듈 로딩
import cgi, sys, codecs, os
import datetime

# 웹 페이지의 form 태그 내의 input 태그 입력값 가져와서 저장하고 있는 인스턴스
form = cgi.FieldStorage()

device = 'cpu'



class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (32, 36, 64)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (64, 18, 32)

        self.fc1 = nn.Linear(64 * 16 * 8, 128)  #After pool2: torch.Size([9, 64, 16, 8]) 64 * 16* 8 꼴 하기
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print("After pool1:", x.shape)  # Debug: print the shape


        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print("After pool2:", x.shape)  # Debug: print the shape

        x = x.view(x.size(0), -1)
        # print("Before FC:", x.shape)  # Debug: print the shape

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


    
model = CNN(num_classes=5).to(device) 




# 이미지 파일이 저장되는 절대 경로
image_dir = os.path.abspath('./Image')
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 클라이언트의 요청 데이터 추출
if 'img_file' in form:
    fileitem = form['img_file']
    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    save_path = os.path.join(image_dir, f'{suffix}_{fileitem.filename}')
    with open(save_path, 'wb') as f:
        f.write(fileitem.file.read())

    # 모델 로드 및 예측
    model_path = os.path.join(os.path.dirname(__file__), '../dancer cnn.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 이미지 파일 로드 및 전처리
    image = cv2.imread(save_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 포맷으로 변경 (cv2를 사용할 때 필요)
    image = Image.fromarray(image)
    image_tensor = trans(image).unsqueeze(0)  # 배치 차원 추가


    # 예측 수행
    output = model(image_tensor)
    prediction = output.argmax().item()
    if prediction ==0:
        prediction = '라치카'
    elif prediction ==1:
        prediction = '바다'
    elif prediction ==2:
        prediction = '배윤정'
    elif prediction ==3:
        prediction = '위뎀보이즈'
    else:
        prediction = '최영준'

    # 결과 출력
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    print("Content-Type: text/html; charset=utf-8")
    print()
    print(f"<TITLE>CGI script output</TITLE>")
    print(f'<H1>이 안무는...</H1>')
    print(f"<H1>Prediction: {prediction} 안무가입니다.</H1>")

    print(f"<img src='../image/{suffix}_{fileitem.filename}' alt='Uploaded Image' style='width: 500px; height: 500px;'>")
else:
    # 이미지 업로드 폼 출력
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    print("Content-Type: text/html; charset=utf-8")
    print()
    print('망함')