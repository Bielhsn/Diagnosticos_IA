import json
import os


def gerar_dados_hospitalares(output_dir="data/fase3"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Protocolos Clínicos (Para o RAG - Base de Conhecimento)
    protocolos = [
        {
            "id": "PROT-001",
            "titulo": "Protocolo de Sepse - Hospital Universitário",
            "conteudo": "Em caso de suspeita de sepse (qSOFA >= 2), iniciar pacote de 1 hora: Coletar lactato e hemoculturas antes dos antibióticos. Administrar antibióticos de amplo espectro. Iniciar reposição volêmica com cristaloides 30ml/kg em caso de hipotensão ou lactato >= 4mmol/L."
        },
        {
            "id": "PROT-002",
            "titulo": "Protocolo de Dor Torácica",
            "conteudo": "Paciente com dor torácica aguda: Realizar ECG em até 10 minutos. Se supra de ST, acionar hemodinâmica imediatamente. Se sem supra, dosar Troponina e estratificar risco (HEART Score). Administrar AAS 300mg mastigável se não houver contraindicação."
        },
        {
            "id": "PROT-003",
            "titulo": "Uso de Antibióticos Restritos",
            "conteudo": "O uso de Meropenem e Vancomicina exige aprovação da CCIH. Preencher ficha de solicitação justificando o uso com base em cultura ou gravidade clínica."
        }
    ]

    # 2. Dataset para Fine-Tuning (Instrução -> Resposta)
    fine_tuning_data = [
        {
            "instruction": "Qual a conduta inicial para suspeita de sepse segundo o protocolo interno?",
            "input": "",
            "output": "Segundo o protocolo do Hospital Universitário, deve-se iniciar o pacote de 1 hora: coletar lactato e hemoculturas, iniciar antibióticos de amplo espectro e hidratação com 30ml/kg se houver hipotensão."
        },
        {
            "instruction": "O paciente apresenta dor no peito. O que fazer?",
            "input": "Paciente homem, 55 anos, dor opressiva.",
            "output": "Realizar ECG em até 10 minutos. Se houver supra de ST, acionar hemodinâmica. Administrar AAS 300mg se não houver alergia. Monitorar sinais vitais."
        },
        {
            "instruction": "Posso prescrever Meropenem direto?",
            "input": "",
            "output": "Não. O Meropenem é um antibiótico restrito e exige aprovação da CCIH mediante preenchimento de ficha de solicitação."
        }
    ]

    # Salvar arquivos
    with open(f"{output_dir}/protocols.json", "w", encoding="utf-8") as f:
        json.dump(protocolos, f, indent=4, ensure_ascii=False)

    with open(f"{output_dir}/finetuning_data.json", "w", encoding="utf-8") as f:
        json.dump(fine_tuning_data, f, indent=4, ensure_ascii=False)

    print(f"Dados sintéticos gerados em {output_dir}")


if __name__ == "__main__":
    gerar_dados_hospitalares()