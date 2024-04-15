# 모델 및 엔코더 로드
import joblib
model = joblib.load('xgb_model.pkl')
encoder = joblib.load('label_encoder.pkl')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import pandas as pd
import librosa
import numpy as np
import os
import sklearn

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# DataFrame을 저장할 빈 리스트 생성
df_list = []

# 소영언니의 분석 결과 -> 나중에 수정
so_result = './soyoung_songs/song_05.mp3'

# WAV 파일 로드
y, sr = librosa.load(so_result)

# MFCCs 계산
mfccs = librosa.feature.mfcc(y=y, sr=sr)
mfccs = normalize(mfccs, axis=1)
mfcc_mean_list = []
mfcc_var_list = []
for i, mfcc_line in enumerate(mfccs[::-1]):
    mfcc_mean_list.append(np.mean(mfcc_line))
    mfcc_var_list.append(np.var(mfcc_line))
    if i == 9:
        break

# Spectral Centroids 계산
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

# Spectral Rolloff 계산
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

# Chromagram 계산
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

# Zero-Crossing Rate (제로 크로싱 비율)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

# Root Mean Square (RMS) Energy (평균 제곱근 에너지)
rms_energy = librosa.feature.rms(y=y)[0]

# Spectral Bandwidth (주파수 대역폭)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

# Spectral Contrast (주파수 대비)
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# Spectral Flatness (주파수 평평도)
spectral_flatness = librosa.feature.spectral_flatness(y=y)

# BPM 계산
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

# 하모니(음색) -> 코러스 좀... / 충격파(둥둥)
harmonic, percussive = librosa.effects.hpss(y=y)

# 각 특징들의 평균과 분산 계산
data = {
    'Filename': so_result,
    'MFCC_Mean': [np.mean(mfccs)],
    'MFCC_Var': [np.var(mfccs)],
    'Chromagram_Mean': [np.mean(chromagram)],
    'Chromagram_Var': [np.var(chromagram)],
    'Spectral_Centroids_Mean': [np.mean(spectral_centroids)],
    'Spectral_Centroids_Var': [np.var(spectral_centroids)],
    'Spectral_Rolloff_Mean': [np.mean(spectral_rolloff)],
    'Spectral_Rolloff_Var': [np.var(spectral_rolloff)],
    'Zero_Crossing_Rate': [np.mean(zero_crossing_rate)],
    'RMS_Energy_Mean': [np.mean(rms_energy)],
    'RMS_Energy_Var': [np.var(rms_energy)],
    'Spectral_Bandwidth_Mean': [np.mean(spectral_bandwidth)],
    'Spectral_Bandwidth_Var': [np.var(spectral_bandwidth)],
    'Spectral_Contrast_Mean': [np.mean(spectral_contrast)],
    'Spectral_Contrast_Var': [np.var(spectral_contrast)],
    'Spectral_Flatness_Mean': [np.mean(spectral_flatness)],
    'Spectral_Flatness_Var': [np.var(spectral_flatness)],
    'Tempo': [tempo],

    "harmony_mean": [harmonic.mean()],
    "harmony_var": [harmonic.var()],
    "precussive_mean": [percussive.mean()],
    "percussive_var": [percussive.var()],
    'label': 0 # 장르는 임시로
}

# 데이터를 DataFrame으로 변환하여 리스트에 추가
df_list.append(pd.DataFrame(data))

# 모든 DataFrame을 하나의 큰 DataFrame으로 결합
new_df = pd.concat(df_list, ignore_index=True)

# 예측
featureDF = new_df.drop(columns=['label', "Filename"])
print(encoder.inverse_transform(model.predict(featureDF)))

## 곡추천
genreDF = pd.read_csv('./csv_data/genre.csv')
totalDF = pd.concat([new_df, genreDF])
totalDF = totalDF.set_index("Filename", drop=True).drop(columns="label")
# print(totalDF)

# 코사인 유사도 분석
data_scaled=preprocessing.scale(totalDF) # 데이터 전처리
similarity = cosine_similarity(data_scaled)
# Convert into a dataframe and then set the row index and column names as labels
simDF = pd.DataFrame(similarity)
simDF = simDF.set_index(totalDF.index)
simDF.columns = totalDF.index

# 유사도한 곡을 뽑아줄 함수
def find_similar_songs(name, sim_df_names):
    indices = sim_df_names[sim_df_names.index.str.endswith(name)].index
    # print(list(indices)) # ['./sounds/dance/dance3.wav']

    file_path = list(indices)[0]
    # Find songs most similar to another song
    series = sim_df_names[file_path].sort_values(ascending=False)

    # 자기자신 제외
    series = series.drop(file_path)

    # Display the 5 top matches
    print("\n*******\nSimilar songs to ")
    print(list(series.head(5).index))

find_similar_songs(so_result, simDF)



