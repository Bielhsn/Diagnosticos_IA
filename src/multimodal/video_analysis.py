import cv2
from ultralytics import YOLO
import time


class SurgicalVideoAnalyzer:
    def __init__(self, model_path='yolov8n.pt'):
        # Carrega o modelo YOLO
        self.model = YOLO(model_path)
        # Classes de interesse (No modelo padrão: 76=scissors, 43=knife)
        # Em um modelo médico real, seriam: 'bisturi', 'pinca', 'afastador'
        self.target_classes = [43, 76]
        self.detected_events = []

    def analyze_video(self, video_path, output_path="output_cirurgia.avi"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return []

        # Configuração para salvar o vídeo processado
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        print(f"--- Iniciando Análise de Vídeo: {video_path} ---")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Realiza a predição
            results = self.model(frame, verbose=False)

            annotated_frame = results[0].plot()  # Desenha as caixas

            # Lógica de Detecção de Anomalias/Instrumentos
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Se detectou um "instrumento" (tesoura/faca simulando bisturi)
                    if cls in self.target_classes and conf > 0.5:
                        timestamp = frame_count / fps
                        event = {
                            "tempo_seg": round(timestamp, 2),
                            "objeto": self.model.names[cls],
                            "confianca": round(conf, 2),
                            "alerta": "Uso de instrumento cortante detectado"
                        }
                        self.detected_events.append(event)

                        # Adiciona alerta visual no vídeo
                        cv2.putText(annotated_frame, "INSTRUMENTO ATIVO", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(annotated_frame)

            # Mostra preview (opcional, aperte 'q' para sair)
            cv2.imshow("Monitoramento Cirurgico - YOLOv8", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return self.detected_events

    def generate_report(self):
        if not self.detected_events:
            return "Nenhum instrumento crítico detectado."

        report = "RELATÓRIO DE MONITORAMENTO CIRÚRGICO (GINECOLÓGICO)\n"
        report += "=" * 50 + "\n"
        report += f"Total de detecções: {len(self.detected_events)}\n"
        report += "Eventos Críticos:\n"

        # Resume os primeiros 5 eventos para não poluir
        for event in self.detected_events[:5]:
            report += f"- [Time: {event['tempo_seg']}s] {event['objeto']} ({event['confianca']}) -> {event['alerta']}\n"

        return report