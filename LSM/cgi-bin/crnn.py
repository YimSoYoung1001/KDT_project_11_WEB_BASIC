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

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        # CNN Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # LSTM Layers
        self.lstm = nn.LSTM(input_size=64 * 16 * 8, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        c_in = x.view(batch_size * seq_len, C, H, W)  # Combine batch and seq_len for CNN processing
        x = self.conv1(c_in)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Prepare for LSTM
        x = x.view(batch_size, seq_len, -1)  # Combine batch and seq_len for LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        # We use the last hidden state to classify
        x = self.fc(lstm_out[:, -1, :])
        return x

    
model = CRNN(num_classes=5).to(device) 




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
    model_path = os.path.join(os.path.dirname(__file__), '../dancer crnn.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 이미지 파일 로드 및 전처리
    image = cv2.imread(save_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 포맷으로 변경 (cv2를 사용할 때 필요)
    image = Image.fromarray(image)
    image_tensor = trans(image).unsqueeze(0)  # 배치 차원 추가

    # 시퀀스 길이 차원 추가 (seq_len=1)
    image_tensor = image_tensor.unsqueeze(1)  # 이 줄을 추가


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