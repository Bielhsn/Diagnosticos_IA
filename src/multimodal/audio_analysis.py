import librosa
import numpy as np
import torch
from transformers import pipeline
import os


class WomensHealthAudioAnalyzer:
    def __init__(self):
        print("Carregando modelo de áudio (pode demorar na 1ª vez)...")
        # Modelo especializado em emoções (inglês, mas emoções vocais são universais)
        self.classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

    def analyze_audio(self, audio_path):
        if not os.path.exists(audio_path):
            return "Erro: Arquivo de áudio não encontrado."

        print(f"--- Processando Áudio: {audio_path} ---")

        # O pipeline do HuggingFace faz o trabalho pesado
        # Ele retorna: neutral, happy, sad, angry
        results = self.classifier(audio_path, top_k=5)

        analysis = self._interpret_results(results)
        return analysis

    def _interpret_results(self, results):
        # Mapeamento para contexto clínico
        translation = {
            "neutral": "Neutro",
            "happy": "Estável/Positivo",
            "sad": "Sinais de Tristeza/Depressão",
            "angry": "Sinais de Estresse/Irritabilidade",
            "fear": "Sinais de Medo/Ansiedade"
        }

        top_emotion = results[0]
        label = top_emotion['label']
        score = top_emotion['score']

        clinical_note = translation.get(label, label)

        report = "ANÁLISE DE ÁUDIO - SAÚDE MENTAL MATERNA\n"
        report += "=" * 50 + "\n"
        report += f"Emoção Predominante: {clinical_note}\n"
        report += f"Nível de Confiança: {round(score * 100, 2)}%\n\n"

        report += "INTERPRETAÇÃO CLÍNICA:\n"
        if label == 'sad':
            report += "⚠️ ALERTA: Indicadores vocais compatíveis com Depressão Pós-Parto. Recomenda-se aplicar escala de Edimburgo.\n"
        elif label == 'fear' or label == 'angry':
            report += "⚠️ ALERTA: Indicadores de alta ansiedade ou estresse. Avaliar histórico de violência doméstica ou ansiedade gestacional.\n"
        else:
            report += "✅ Padrão vocal dentro da normalidade esperada.\n"

        return report