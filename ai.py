# Made by KDY of D.H.S. feat.LCH.T
import tensorflow.keras     # 티처블 머신이 Tensorflow기반이므로 import
import numpy as np          # Tensorflow와 함께 사용
import cv2                  # 파이썬에서 영상처리를 위한 Open-CV를 import
import random               # random묘둘을 통해 pc의 가위바위보 제시
import h5py                 # Keras모델이 .h5 확장자이므로 h5py묘둘을 사용
import time                 # 딜레이를 위한 Time 묘둘

computer = random.choice(['rock', 'scissor', 'paper'])     # 컴퓨터의 제시를 ramdom으로 선택

f = h5py.File('keras_model.h5')     # Teachable Machine에서 추출한 Keras모델을 불러옴

model = tensorflow.keras.models.load_model(f)       # Keras모델을 불러온후 model 변수로 지정

print('3초 이내 가위,바위,보 중에서 원하는 것을 카메라에 표현하세요.')
print('안 내면 진다')
print('가위')
time.sleep(1.5)
print('바위')
time.sleep(1.5)
print('보')

# ------------Open-cv의 영상 크기 지정--------------
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# --------------------------------------------------

# ----------------- 224x224 크기로 이미지 데이터를 수정 -----------------
def preprocessing(frame):       # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1       # 이미지 정규화
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))     # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다. (keras 모델에 공급할 올바른 모양의 배열 생성)
    return frame_reshaped
# -----------------------------------------------------------------------

# ---------------------------- 이미지 데이터 예측 -----------------------------
def predict(frame):
    prediction = model.predict(frame)
    return prediction


ret, frame = capture.read()

preprocessed = preprocessing(frame)
prediction = predict(preprocessed)  # Teachable Machine으로 받은 예측값을 prediction 변수로 지정

if (prediction[0,0] > prediction[0,1], prediction[0,2]):   # 예측값이 0,0 >> rock 묵
    player = 'rock'
    cv2.putText(frame, 'rock', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))


elif (prediction[0,1] > prediction[0,0], prediction[0,2]):   # 예측값이 0,1 >> paper 빠
    cv2.putText(frame, 'paper', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    player = 'paper'

else:
    cv2.putText(frame, 'scissor', (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))   # 예측값이 0,2 >> scissor 찌
    player = 'scissor'


if computer == 'scissor':
    print('컴퓨터는 가위를 냈습니다.')
    if player == 'scissor':
        print('플레이어는 가위를 냈습니다.')
        print('무승부')
    elif player == 'rock':
        print('플레이어는 바위를 냈습니다.')
        print('플레이어 승리')   
    elif player == 'paper':
        print('플레이어는 보를 냈습니다.')
        print('컴퓨터 승리')

elif computer == 'rock':
    print('컴퓨터는 바위를 냈습니다.')
    if player == 'scissor':
        print('플레이어는 가위를 냈습니다.')
        print('컴퓨터 승리')
    elif player == 'rock':
        print('플레이어는 바위를 냈습니다.')
        print('무승부')
    elif player == 'paper':
        print('플레이어는 보를 냈습니다.')
        print('플레이어 승리')
        
elif computer == 'paper':
    print('컴퓨터는 보를 냈습니다.')
    if player == 'scissor':
        print('플레이어는 가위를 냈습니다.')
        print('플레이어 승리')
    elif player == 'rock':
        print('플레이어는 바위를 냈습니다.')
        print('컴퓨터 승리') 
    elif player == 'paper':
        print('플레이어는 보를 냈습니다.')
        print('무승부')
