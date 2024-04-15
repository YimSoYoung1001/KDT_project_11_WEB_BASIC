#!/usr/bin/env python3
import cgi
import os
import librosa
import numpy as np
import sys



def extract_pitch(file_path):
    y, sr = librosa.load(file_path, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    valid_f0 = f0[voiced_flag]
    min_pitch = np.min(valid_f0) if np.any(valid_f0) else None
    max_pitch = np.max(valid_f0) if np.any(valid_f0) else None
    return min_pitch, max_pitch

def calculate_singing_probability(min_pitch, max_pitch, other_min, other_max):
    # MIDI 변환 후 비교
    artist_range = (librosa.hz_to_midi(min_pitch), librosa.hz_to_midi(max_pitch))
    song_range = (librosa.hz_to_midi(other_min), librosa.hz_to_midi(other_max))
    overlap = max(0, min(artist_range[1], song_range[1]) - max(artist_range[0], song_range[0]))
    total_range = max(song_range[1] - song_range[0], 1)  # 0으로 나누는 것 방지
    return (overlap / total_range) * 100

form = cgi.FieldStorage()

response_html = "<html><body><h1>Uploaded Audio Analysis</h1>"


# 샘플 파일 처리
samples_pitches = []
for i in range(1, 11):
    file_field = f'file{i}'
    response_html += f"====>file_field {file_field}<br>"
    if file_field in form:
        fileitem = form[file_field]
        response_html += f"<p>file_field {file_field} : fileitem.filename :  {fileitem.filename}</p>"
        if fileitem.filename:
            filepath = 'C:\\temp\\' + os.path.basename(fileitem.filename)
            with open(filepath, 'wb') as f:
                f.write(fileitem.file.read())
            
            min_pitch, max_pitch = extract_pitch(filepath)
            samples_pitches.append((min_pitch, max_pitch))
            response_html += f"<p>Sample {i}의 음역대 - Min Pitch : {min_pitch}, Max Pitch : {max_pitch}</p>"

# # 전체 노래 파일 처리
# if 'full_song' in form:
#     fileitem = form['full_song']
#     if fileitem.filename:
#         filepath = 'C:\\temp\\' + os.path.basename(fileitem.filename)
#         with open(filepath, 'wb') as f:
#             f.write(fileitem.file.read())
#         full_min_pitch, full_max_pitch = extract_pitch(filepath)
#         response_html += f"<p>완곡의 음역대 - Min Pitch : {full_min_pitch}, Max Pitch : {full_max_pitch}</p>"

# # 다른 가수의 노래 파일 처리
# if 'other_song' in form:
#     fileitem = form['other_song']
#     if fileitem.filename:
#         filepath = 'C:\\temp\\' + os.path.basename(fileitem.filename)
#         with open(filepath, 'wb') as f:
#             f.write(fileitem.file.read())
#         other_min_pitch, other_max_pitch = extract_pitch(filepath)
#         probability = calculate_singing_probability(full_min_pitch, full_max_pitch, other_min_pitch, other_max_pitch)
#         response_html += f"<p>최애가 당신이 사랑하는 노래를 부를 수 있는 가능성은? : {probability:.2f}%</p>"


print("Content-Type: text/html")
print("")

response_html += "</body></html>"
print(response_html)
