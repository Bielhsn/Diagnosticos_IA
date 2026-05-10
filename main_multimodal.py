import os
from src.multimodal.video_analysis import WomensHealthVideoAnalyzer
from src.multimodal.audio_analysis import WomensHealthAudioAnalyzer
video_analyzer = WomensHealthVideoAnalyzer()

def main():
    print("=== FASE 4: SISTEMA MULTIMODAL DE SAÚDE DA MULHER ===")

    # Caminhos
    video_path = "data/fase4/mulher_consulta.mp4"
    audio_path = "data/fase4/OAF_back_sad.wav"

    # 1. Análise de Vídeo (Cirurgia/Procedimentos)
    print("\n[1/2] Iniciando Análise de Vídeo...")
    if os.path.exists(video_path):
        video_analyzer = WomensHealthVideoAnalyzer()  # Usa modelo padrão (detecta tesouras/facas como teste)
        # Processa
        video_analyzer.analyze_video(video_path, output_path="data/fase4/resultado_video.avi")
        # Relatório
        print(video_analyzer.generate_report())
    else:
        print(f"⚠️ Arquivo de vídeo não encontrado em {video_path}. Adicione um arquivo para testar.")

    # 2. Análise de Áudio (Saúde Mental)
    print("\n[2/2] Iniciando Análise de Áudio...")
    if os.path.exists(audio_path):
        audio_analyzer = WomensHealthAudioAnalyzer()
        relatorio_audio = audio_analyzer.analyze_audio(audio_path)
        print(relatorio_audio)
    else:
        print(f"⚠️ Arquivo de áudio não encontrado em {audio_path}. Adicione um arquivo para testar.")


if __name__ == "__main__":
    main()