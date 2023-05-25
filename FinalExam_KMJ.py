# 12214028 김민준
import cv2
import numpy as np

# 비디오 파일 경로
video_path = "final.mp4"

# 동영상 파일 열기
video = cv2.VideoCapture(video_path)

# 이전 프레임의 히스토그램
previous_hist = None
# 이전 프레임
previous_frame = None



# 재생 속도 조절을 위한 딜레이 계산
delay = 10  # 원본 속도로 재생


# 히스토그램 차이를 기반으로 샷 변경 여부를 확인할 임계값
threshold = 500

while video.isOpened():
    # 프레임 읽기
    ret, frame = video.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 현재 프레임의 히스토그램 계산
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    print(hist)


    # 여기서부터는 움직이는 사물의 윤곽선만을 따기 위한 코드
    # 전처리: 가우시안 블러링
    blurred = cv2.GaussianBlur(frame, (25, 25), 0)
    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    if previous_frame is not None:
        diff = cv2.absdiff(gray, previous_frame)
        # 이진화
        _, thresholded = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)

        # 윤곽선 검출
        contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # 윤곽선 그리기
        for contour in contours:
            # 윤곽선을 둘러싼 직사각형 그리기
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    previous_frame = gray.copy()

    # 여기서부터는 히스토그램 계산을 이용해서 샷 바뀔때 인지하는 코드
    if previous_hist is not None:
        # 이전 프레임과의 히스토그램 차이 계산. 바타차야 거리 -> 계산속도는 느리지만 가장 정확한 알고리즘
        # diff = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        diff = hist - previous_hist
        print(diff)

        if np.any(diff > threshold):
            # 샷 변경이 감지된 경우...(400,30)정도 해야 우측 상단에 뿌려줌..흰색
            cv2.putText(frame, "Shot Changed", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 현재 프레임의 히스토그램을 이전 프레임의 히스토그램으로 설정
    previous_hist = hist

    # 화면에 프레임 출력
    cv2.imshow("Video", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 종료 시 리소스 해제
video.release()
cv2.destroyAllWindows()