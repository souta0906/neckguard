import cv2
import mediapipe as mp

# MediaPipe Poseの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Webカメラの起動（0は通常メインカメラ）
cap = cv2.VideoCapture(0)

# 姿勢推定モデルの設定
with mp_pose.Pose(
    min_detection_confidence=0.5,  # 検出の信頼度閾値
    min_tracking_confidence=0.5    # トラッキングの信頼度閾値
) as pose:
    
    while cap.isOpened():
        # フレームの読み込み
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR画像をRGBに変換（MediaPipeはRGBを想定）
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿勢推定の実行
        results = pose.process(image)
        
        # 描画のためRGBをBGRに戻す
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 検出された姿勢を画像上に描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        # ウィンドウに表示
        cv2.imshow('MediaPipe Pose', image)
        
        # 'q'キーで終了
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# リソースの解放
cap.release()
cv2.destroyAllWindows()