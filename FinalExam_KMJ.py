# 12214028 김민준
import cv2

# 비디오 파일 경로
video_path = "final.mp4"

# 동영상 파일 열기
video = cv2.VideoCapture(video_path)

# 이전 프레임의 히스토그램
previous_hist = None

# 재생 속도 조절을 위한 딜레이 계산
delay = 10  # 원본 속도로 재생

# 임계값 설정
a = 0.02

# 히스토그램 차이를 기반으로 샷 변경 여부를 확인할 임계값
threshold = a

while video.isOpened():
    # 프레임 읽기
    ret, frame = video.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 현재 프레임의 히스토그램 계산
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    if previous_hist is not None:
        # 이전 프레임과의 히스토그램 차이 계산
        diff = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        if diff > threshold:
            # 샷 변경이 감지된 경우
            cv2.putText(frame, "Shot Changed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 현재 프레임의 히스토그램을 이전 프레임의 히스토그램으로 설정
    previous_hist = hist

    # 화면에 프레임 출력
    cv2.imshow("Video", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 종료 시 리소스 해제
video.release()
cv2.destroyAllWindows()