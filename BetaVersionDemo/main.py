'''
作者：刘洪达
名称：手势识别演示程序
功能：调用个人电脑摄像头检测用户做出的手势，输出识别结果和识别用时
'''


import copy
import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow.keras.backend as K
from tensorflow import keras


def lossf_3(y_true, y_pred):
    return K.mean((1 - y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(1 - y_pred, 0)))


def predict(test_of, hand_num):
    test_of = np.array(test_of)
    test_of = test_of.reshape(1, 60, 21)

    print("{} hand".format(hand_num))

    list_pro = []
    if hand_num == 1:
        p = model1.predict([test_of, dj_std])
        list_pro.append(p)
        p = model1.predict([test_of, py_std])
        list_pro.append(p)
        p = model1.predict([test_of, zq_std])
        list_pro.append(p)
    else:
        p = model2.predict([test_of, xz_std])
        list_pro.append(p)
        p = model2.predict([test_of, fs_std])
        list_pro.append(p)

    # lhd调参结果，准确率与动态性均满足，缺点是预测时间比较长
    # if hand_num == 1:
    #     if list_pro[1] > 0.9:
    #         print('平移')
    #         pass
    #     elif list_pro[0] > 0.99:
    #         print('点击')
    #         pass
    #     elif list_pro[2] > 0.995:
    #         print('抓取')
    #         pass
    #     pass
    # elif hand_num == 2:
    #     if list_pro[0] > 0.5:
    #         print('旋转')
    #         pass
    #     elif list_pro[1] > 0.95:
    #         print("放缩")

    cate = list_pro.index(max(list_pro))
    if list_pro[cate] > 0.95:
        if hand_num == 1:
            if cate == 0:
                print("点击")
            elif cate == 1:
                print("平移")
            elif cate == 2:
                print("抓取")
        else:
            if cate == 0:
                print("旋转")
            elif cate == 1:
                print("放缩")





# 识别参数
cap_device = 0
cap_width = 960
cap_height = 540
max_num_hands = 2
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

# 初始化视频捕获器
cap = cv2.VideoCapture(cap_device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
cap.set(cv2.CAP_PROP_FPS, 30)

# 初始化手部识别器
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    # static_image_mode=False,
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)
mp_drawing = mp.solutions.drawing_utils

# load model
model1 = keras.models.load_model("1hand_2lstm_3dense.h5", custom_objects={'lossf_3': lossf_3})
model2 = keras.models.load_model("2hand_2lstm_3dense.h5", custom_objects={'lossf_3': lossf_3})
dj_std = np.load('dj_std.npy')
py_std = np.load('py_std.npy')
zq_std = np.load('zq_std.npy')
xz_std = np.load('xz_std.npy')
fs_std = np.load('fs_std.npy')
print("all lib has been loaded")

# 记录最近60帧的optical_flow
frame_of = []
pre_ldmrk = []
frame_idx = 0

# 显示帧数
pre_t = time.time()
start_time = 0

# 双手帧数
two_hand_frame_num = 0

while True:

    ret, image = cap.read()
    if not ret:
        break

    # 输出实时帧率
    # time_diff = time.time() - pre_t
    # if time_diff != 0:
    #     print("fps:", 1.0 / (time.time() - pre_t))
    # pre_t = time.time()

    # 镜像翻转
    image = cv2.flip(image, 1)
    debug_image = copy.deepcopy(image)

    # mediapipe 处理
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 如没识别到手
    if not results.multi_hand_landmarks:
        key = cv2.waitKey(1)
        if key == 27:
            break
        cv2.imshow('MediaPipe Hand Demo', debug_image)
        continue

    which_hand = 0
    if len(results.multi_hand_landmarks) >= 2:
        two_hand_frame_num += 1
        if results.multi_handedness[0].classification[0].label == "Right":
            which_hand = 0
        else:
            which_hand = 1

    # 将landmark画在图上
    mp_drawing.draw_landmarks(debug_image, results.multi_hand_landmarks[which_hand], mp_hands.HAND_CONNECTIONS)

    # 显示图像，waitKey保证画面停留，ESC以关闭
    key = cv2.waitKey(1)
    if key == 27:
        break
    cv2.imshow('MediaPipe Hand Demo', debug_image)

    ldmrk = []
    cur_hand_landmark = results.multi_hand_landmarks[which_hand].landmark

    # 存储landmark
    for landmarks in cur_hand_landmark:
        ldmrk.append([landmarks.x, landmarks.y])

    if frame_idx == 0:
        pre_ldmrk = ldmrk
        frame_idx += 1
        continue
    frame_idx += 1

    cur_of = []
    for i in range(21):
        cur_of.append(np.sqrt(((pre_ldmrk[i][0] - ldmrk[i][0]) * 100) ** 2 + ((pre_ldmrk[i][1] - ldmrk[i][1]) * 100) ** 2))

    frame_of.append(cur_of)
    pre_ldmrk = ldmrk

    if len(frame_of) == 60:
        pre_t = time.time()
        if two_hand_frame_num > 40:
            predict(frame_of, 2)
        else:
            predict(frame_of, 1)
        two_hand_frame_num = 0
        frame_of.clear()
        # del frame_of[0:30]
        print("time cost:", time.time()-pre_t)
        print('--------------------')


cap.release()
cv2.destroyAllWindows()
