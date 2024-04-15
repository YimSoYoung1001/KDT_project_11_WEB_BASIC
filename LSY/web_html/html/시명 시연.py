import cv2
import os

# 라치카 폴더 경로 설정
folder_path = "라치카"

# 입력 영상 파일 목록 가져오기
input_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".mp4")]

# 배경 제거 알고리즘 객체 생성
fgbg = cv2.createBackgroundSubtractorMOG2()

# 각 영상 처리
for index, input_file in enumerate(input_files):
  # 출력 파일 이름 설정 (라치카01.mp4, 라치카02.mp4 식)
  output_file = os.path.join(folder_path, f"라치카{index+1:02}.mp4")

  # 비디오 정보 가져오기
  cap = cv2.VideoCapture(input_file)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  # 출력 비디오 설정
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

  # 프레임 번호 카운터
  frame_count = 0

  while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()

    if not ret:
      break

    # 배경 제거 알고리즘 적용
    fgmask = fgbg.apply(frame)

    # 결과 프레임을 BGR 색 공간으로 변환 후 출력 비디오에 저장
    out.write(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))

    # 결과 표시 (선택사항)
    cv2.imshow('Original', frame)
    cv2.imshow("Background Removed", fgmask)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    frame_count += 1

  print(f"[{index+1}/{len(input_files)}] {input_file} processed. Total frames: {frame_count}")

  # 자원 해제
  cap.release()
  out.release()

print("All videos processed.")
