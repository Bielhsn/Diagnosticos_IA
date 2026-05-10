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

        translation = {
            "neu": "Neutro",
            "hap": "Estável/Positivo",
            "sad": "Tristeza",
            "ang": "Estresse/Irritabilidade",
            "fea": "Medo/Ansiedade"
        }

        top_emotion = results[0]
        label = top_emotion['label']
        score = top_emotion['score']

        emotion_scores = {}

        for emotion in results:
            emotion_scores[
                emotion['label']
            ] = emotion['score']

        report = (
            "ANÁLISE DE ÁUDIO - "
            "SAÚDE DA MULHER\n"
        )

        report += "=" * 50 + "\n"

        report += (
            "PRINCIPAIS PADRÕES "
            "VOCAIS DETECTADOS:\n"
        )

        for emotion in results[:3]:
            translated = translation.get(
                emotion['label'],
                emotion['label']
            )

            confidence = round(
                emotion['score'] * 100,
                2
            )

            report += (
                f"- {translated}: "
                f"{confidence}%\n"
            )

        report += "\n"

        report += (
            "INTERPRETAÇÃO CLÍNICA:\n"
        )

        if label == "sad":

            report += (
                "⚠️ Possíveis sinais "
                "de sofrimento emocional.\n"
            )

            report += (
                "⚠️ Perfil vocal "
                "compatível com "
                "depressão pós-parto "
                "ou tristeza persistente.\n"
            )

            report += (
                "Recomendação: "
                "aplicar escala "
                "de Edimburgo.\n"
            )

        elif label == "fea":

            report += (
                "⚠️ Sinais vocais "
                "de ansiedade, medo "
                "ou trauma emocional.\n"
            )

            report += (
                "⚠️ Recomenda-se "
                "triagem para "
                "ansiedade gestacional "
                "e investigação "
                "de possível "
                "violência doméstica.\n"
            )

        elif label == "ang":

            report += (
                "⚠️ Indícios de "
                "estresse emocional "
                "ou sofrimento psíquico.\n"
            )

            report += (
                "Avaliar sobrecarga "
                "emocional e "
                "rede de suporte.\n"
            )

        elif label == "neu":

            sadness_level = (

                emotion_scores.get(

                    "sad",

                    0

                )

            )

            fear_level = (

                emotion_scores.get(

                    "fea",

                    0

                )

            )

            if sadness_level > 0.25:

                report += (

                    "⚠️ Apesar do "
            
                    "padrão vocal "
            
                    "majoritariamente "
            
                    "neutro, foram "
            
                    "detectados sinais "
            
                    "moderados de "
            
                    "tristeza vocal.\n"

                )

                report += (

                    "Recomenda-se "
            
                    "monitoramento "
            
                    "para sofrimento "
            
                    "emocional ou "
            
                    "depressão "
            
                    "pós-parto.\n"

                )


            elif fear_level > 0.25:

                report += (

                    "⚠️ Detectados "
            
                    "traços vocais "
            
                    "de ansiedade "
            
                    "ou medo.\n"

                )

                report += (

                    "Recomenda-se "
            
                    "avaliação "
            
                    "emocional "
            
                    "preventiva.\n"

                )


            else:

                report += (

                    "✅ Sem indicadores "
            
                    "vocais relevantes "
            
                    "de sofrimento "
            
                    "emocional.\n"

                )

        elif label == "hap":

            report += (
                "✅ Padrão vocal "
                "compatível com "
                "estabilidade emocional.\n"
            )

        return report