### 모듈 로딩
import cgi, sys, codecs, datetime
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



### 예측해보기
new_path = '../img/240414_182045_aaa.jpg'

img = Image.open(new_path)


### 모델 돌릴 클래스
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        # 채널 수          커널수         커널 사이즈,
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)  # class는 총 8개

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 8 * 8 * 16)  # 차원을 변경함
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x


### 모델 로딩
model = my_model()
print(f"<h6>모델 인스턴스 생성 완료</h6>")

model_file = 'mood+face_100.pth'
model.load_state_dict(torch.load(model_file))
print(f"<h6>모델 로딩 완료</h6>")

### 모델이 학습된 형태의 이미지로 변환
preprocessing = transforms.Compose([
    transforms.Resize(size = (32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# img = img.unsqueeze(0)  # 차원을 추가
p_img = preprocessing(img)
print(p_img.shape)


### 모델 시연
model.eval()

with torch.no_grad():
    output = model(p_img)
    result = torch.argmax(output, dim = 1).item()


### 분석된 결과에 따라 어떤 분위기/감정인지 알려주기
mood_dict = {'0':'angry', '1':'anxiety', '2':'depressed', '3':'dynamic', '4':'happy', '5':'peaceful', '6':'tired', '7':'withered'}
mood = mood_dict[f"{result}"].upper()
print(f"<h2>your image has {mood} mood</h2>")


### 노래 추천
import IPython.display as ipd
import librosa

song_mood = {'angry':'1', 'anxiety':'2', 'depressed':'3', 'dynamic':'4', 'happy':'5', 'peaceful':'6', 'tired':'7', 'withered':'8'}
song_opposite = {'angry':'6', 'anxiety':'5', 'depressed':'5', 'dynamic':'6', 'happy':'6', 'peaceful':'6', 'tired':'5', 'withered':'5'}

print(f'your mood is {mood}')

# explain mood
song_num = song_mood[f'{mood}']
mp3_path = f'./song_list/song_0{song_num}.mp3'
print(f'<h2>then, this song will EXPLAIN your mood.</h2>')
print(f"<h2>(song number is {song_num})</h2>")

# MP3 파일 재생
y, sr = librosa.load(mp3_path)
audio1 = ipd.Audio(y, rate=sr, autoplay=False)
print(f"<h2>ipd.display({audio1})</h2>")

# change mood
song_num = song_opposite[f'{mood}']
mp3_path = f'./song_list/song_0{song_num}.mp3'
print(f'<h2>then, this song will CHANGE your mood.</h2>')
print(f"<h2>(song number is {song_num})</h2>")

# MP3 파일 재생
y, sr = librosa.load(mp3_path)
audio2 = ipd.Audio(y, rate=sr, autoplay=False)
print(f"<h2>ipd.display({audio2})</h2>")


# python -m http.server 8080 --bind 127.0.0.1 --cgi