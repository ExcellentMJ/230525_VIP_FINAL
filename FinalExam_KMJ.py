# import cv2
# import numpy as np
#
# # 동영상 파일 경로
# video_path = "final.mp4"
#
# # 배경 차분을 위한 배경 모델 초기화
# background_subtractor = cv2.createBackgroundSubtractorMOG2()
#
# # 히스토그램 차이 임계값
# alpha = 0.12
#
# # 샷 변화 표시용 함수
# def draw_shot_change(frame, text):
#     cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
# # 배경 저장용 함수
# def save_background(frame, shot_index):
#     cv2.imwrite(f"shot_bg{shot_index}.jpg", frame)
#
# # 동영상 열기
# video = cv2.VideoCapture(video_path)
#
# # 초기 프레임 읽기
# _, prev_frame = video.read()
# prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (5, 5), 0)
# prev_hist = cv2.calcHist([prev_frame_gray], [0], None, [256], [0, 256])
#
# shot_index = 1
#
# while True:
#     # 현재 프레임 읽기
#     ret, frame = video.read()
#     if not ret:
#         break
#
#     # 그레이스케일로 변환
#     curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 전처리: 가우시안 블러
#     curr_frame_gray = cv2.GaussianBlur(curr_frame_gray, (5, 5), 0)
#     curr_hist = cv2.calcHist([curr_frame_gray], [0], None, [256], [0, 256])
#
#     # 배경 차분 수행
#     mask = background_subtractor.apply(frame)
#
#     # 이진화
#     _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#
#     # 모폴로지 연산
#     kernel = np.ones((3, 3), np.uint8)
#     binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
#
#     # 컨투어 검출
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 컨투어를 기반으로 움직이는 객체 검출 및 표시
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 50:  # 임계값 조정 가능
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # 샷 변화 체크
#     hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
#     if hist_diff > alpha:
#         draw_shot_change(frame, "Shot Changed")
#         save_background(frame, shot_index)
#         shot_index += 1
#
#     # 현재 프레임을 이전 프레임으로 설정
#     prev_frame_gray = curr_frame_gray.copy()
#     prev_hist = curr_hist.copy()
#
#     # 화면에 표시
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(5) == ord("q"):
#         break
#
# # 동영상 종료 후 정리
# video.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 동영상 파일 경로
video_path = "final.mp4"

# 배경 차분을 위한 배경 모델 초기화
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# 히스토그램 차이 임계값
alpha = 0.12
# 윤곽선 검출을 위한 모션 임계값
motion_threshold = 500
#
shot_index = 1
# 샷 변화 표시용 함수
def draw_shot_change(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 배경 저장용 함수
def save_background(frame, shot_index):
    cv2.imwrite(f"shot_bg{shot_index}.jpg", frame)
# 윤곽선 검출을 위한 모션 임계값

# # 초기 프레임 읽기
# # 동영상 열기
video = cv2.VideoCapture(video_path)
_, prev_frame = video.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (5, 5), 0)
prev_hist = cv2.calcHist([prev_frame_gray], [0], None, [256], [0, 256])

# Dense Optical Flow 파라미터
dense_flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

# 동영상 열기
video = cv2.VideoCapture(video_path)

while True:
    # 현재 프레임 읽기
    ret, frame = video.read()
    if not ret:
        break
    # 그레이스케일로 변환
    curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 전처리: 가우시안 블러
    curr_frame_gray = cv2.GaussianBlur(curr_frame_gray, (5, 5), 0)
    curr_hist = cv2.calcHist([curr_frame_gray], [0], None, [256], [0, 256])
    # 배경 차분 수행
    mask = background_subtractor.apply(frame)

    # 이진화
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 모폴로지 연산
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # 샷 변화 체크
    hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    if hist_diff > alpha:
        draw_shot_change(frame, "Shot Changed")
        save_background(frame, shot_index)
        shot_index += 1
    # 컨투어 검출
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어를 기반으로 움직이는 객체 검출 및 표시
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > motion_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dense Optical Flow 계산
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_roi = prev_frame_gray[y:y + h, x:x + w]

            curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_roi = curr_frame_gray[y:y + h, x:x + w]

            flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, **dense_flow_params)

            # 화살표 그리기
            step = 10
            for i in range(0, h, step):
                for j in range(0, w, step):
                    dx, dy = flow[i, j]
                    cv2.arrowedLine(frame, (x + j, y + i), (x + j + int(dx), y + i + int(dy)), (255, 0, 0), 2)

    # 현재 프레임을 이전 프레임으로 설정
    prev_frame_gray = curr_frame_gray.copy()
    prev_hist = curr_hist.copy()

    # 화면에 표시

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# 동영상 종료 후 정리
video.release()
cv2.destroyAllWindows()