# Detecção de Queimadas com Deep Learning - Hackathon INPE 2025

Este repositório contém a solução desenvolvida para o desafio de detecção de queimadas do **WorCAP 2025**, um hackathon promovido pelo INPE. O objetivo é identificar áreas de cicatrizes de queimadas em imagens de satélite usando um modelo de Deep Learning.

## Metodologia

A abordagem utilizada se baseia na análise de mudança temporal de índices espectrais, processada por uma rede neural convolucional do tipo U-Net.

1.  **Pré-processamento e Engenharia de Atributos:**
    *   Para cada local, utilizamos imagens de dois momentos distintos: antes (T1) e depois (T2) da ocorrência da queimada.
    *   Em vez de usar as imagens brutas, calculamos dois índices espectrais chave para cada imagem:
        *   **NBR (Normalized Burn Ratio):** Altamente sensível à presença de vegetação queimada.
        *   **NDVI (Normalized Difference Vegetation Index):** Mede a saúde e a densidade da vegetação.
    *   Criamos uma "imagem de mudança" de 2 canais calculando a diferença desses índices entre os dois momentos: `dNBR = NBR_T1 - NBR_T2` e `dNDVI = NDVI_T1 - NDVI_T2`. Esta imagem de mudança é a entrada para o nosso modelo.

2.  **Arquitetura do Modelo:**
    *   Foi implementada uma arquitetura **U-Net**, que é um padrão de excelência para tarefas de segmentação de imagens, especialmente em contextos biomédicos e de sensoriamento remoto.
    *   O modelo foi treinado para receber a imagem de mudança de 2 canais (dNBR, dNDVI) e gerar uma máscara de segmentação binária, onde cada pixel é classificado como "Queimado" (1) ou "Não Queimado" (0).

3.  **Treinamento e Avaliação:**
    *   O dataset foi dividido em 80% para treino e 20% para validação.
    *   O modelo foi treinado utilizando a função de perda `Binary Cross-Entropy` (BCELoss) e o otimizador `Adam`.
    *   A avaliação do modelo foi realizada com base em métricas de classificação por pixel, com foco especial no **F1-Score** para a classe "Queimado", que oferece o melhor balanço entre `Precision` e `Recall`.

## Resultados

O modelo final ("Sem Peso") alcançou um excelente equilíbrio entre a capacidade de detecção e a confiabilidade dos alertas, obtendo os seguintes resultados no conjunto de validação:

| Classe | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Queimado** | **0.78** | **0.60** | **0.68** |
| Não Queimado | 0.99 | 1.00 | 1.00 |

O **AUC-ROC** geral foi de **0.9904**, indicando uma excelente capacidade de discriminação do modelo.

## Como Executar o Projeto

Este projeto foi desenvolvido em um ambiente Google Colab.

### 1. Configuração do Ambiente

Execute a primeira célula do notebook `worcap-2025.ipynb` para:
*   Instalar as dependências (`kaggle`, `torch`, `rasterio`, `spyndex`).
*   Fazer o upload da sua chave de API do Kaggle (`kaggle.json`).
*   Baixar e descompactar o dataset da competição.
*   Instalar a versão correta do PyTorch com suporte a CUDA.

### 2. Treinamento do Modelo

Execute a célula principal do notebook que contém as definições de `ChangeIndexDataset`, `UNet`, e as funções `train` e `evaluate_model`.
*   Isso irá carregar os dados, treinar o modelo por 51 épocas e salvar o modelo treinado como `unet_indices_model.pth`.
*   Ao final, serão exibidos o Relatório de Classificação, a curva ROC-AUC e exemplos visuais das predições.

### 3. Geração do Arquivo de Submissão

A última célula do notebook é responsável por gerar o arquivo `predicted_change_masks.csv` para submissão na competição.
*   Ela carrega o modelo treinado (`unet_indices_model.pth`).
*   Itera sobre as imagens da pasta de `avaliacao`.
*   Aplica o mesmo pré-processamento de cálculo de índices.
*   Gera a máscara de predição para cada imagem e a salva no formato CSV