import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time


class WomensHealthVideoAnalyzer:

    def __init__(self, model_path='yolov8n.pt'):
        self.model_path = model_path
        self.detected_events = []

        self.last_alert_time = -999
        self.alert_interval = 3

        self._load_models()

    def _load_models(self):

        # Carrega YOLOv8
        self.model = YOLO(self.model_path)

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Utilitário para desenhar landmarks
        self.mp_drawing = mp.solutions.drawing_utils

    def _detect_head_down(self, landmarks):

        nose = landmarks[
            self.mp_pose.PoseLandmark.NOSE.value
        ]

        left_shoulder = landmarks[
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ]

        right_shoulder = landmarks[
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ]

        shoulder_y = (
                             left_shoulder.y +
                             right_shoulder.y
                     ) / 2

        if nose.y > shoulder_y - 0.08:
            return True

        return False

    def analyze_video(self, video_path, output_path="output_cirurgia.avi"):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return []

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (width, height)
        )

        print(f"--- Iniciando Análise de Vídeo: {video_path} ---")

        cv2.namedWindow(
            "Monitoramento Feminino - YOLOv8",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Monitoramento Feminino - YOLOv8",
            900,
            600
        )

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            results = self.model(frame, verbose=False)

            annotated_frame = results[0].plot()

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pose_results = self.pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                head_down = self._detect_head_down(
                    landmarks
                )

                if head_down:
                    timestamp = frame_count / fps

                    time_since_last_alert = (
                            timestamp -
                            self.last_alert_time
                    )

                    if time_since_last_alert >= self.alert_interval:
                        event = {
                            "tempo_seg": round(timestamp, 2),
                            "alerta": (
                                "Possível desconforto "
                                "psicológico detectado"
                            )
                        }

                        self.detected_events.append(event)

                        self.last_alert_time = timestamp

                    cv2.putText(
                        annotated_frame,
                        "ALERTA: POSSIVEL DESCONFORTO",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            out.write(annotated_frame)

            cv2.imshow(
                "Monitoramento Feminino - YOLOv8",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return self.detected_events

    def generate_report(self):

        if not self.detected_events:
            return "Nenhum evento de desconforto detectado."

        report = "RELATÓRIO DE MONITORAMENTO DE SAÚDE DA MULHER\n"
        report += "=" * 50 + "\n"
        report += f"Total de eventos: {len(self.detected_events)}\n"

        for event in self.detected_events[:5]:
            report += (
                f"- [{event['tempo_seg']}s] "
                f"{event['alerta']}\n"
            )

        return report