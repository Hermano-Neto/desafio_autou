# Analisador de Produtividade de E-mails com IA

## 🎯 Descrição do Projeto

Este projeto é uma aplicação web desenvolvida para classificar e-mails como "Produtivos" ou "Improdutivos" e, adicionalmente, sugerir respostas automáticas adequadas ao teor de cada e-mail. O objetivo principal é otimizar o tempo de equipes que gerenciam um alto volume de correspondências eletrônicas, automatizando parte do processo de triagem e resposta inicial.

A solução utiliza a API do Google Gemini para realizar a análise inteligente e a classificação do conteúdo dos e-mails, e técnicas de Processamento de Linguagem Natural (com a biblioteca NLTK) para o pré-processamento dos textos.

## ✨ Funcionalidades

* **Entrada Flexível de E-mail:** Permite ao usuário colar o texto de um e-mail diretamente na interface ou fazer o upload de arquivos nos formatos `.txt` ou `.pdf`.
* **Classificação Automática por IA:** Categoriza os e-mails em:
    * **Produtivo:** E-mails que demandam uma ação ou resposta específica (ex: solicitações de suporte, atualizações de status, dúvidas técnicas, envio de documentos para revisão/ação).
    * **Improdutivo:** E-mails que não requerem uma ação imediata ou são de natureza social/informativa geral (ex: mensagens de felicitações, agradecimentos, spam, newsletters).
* **Sugestão de Resposta Inteligente:** Com base na classificação, o sistema propõe um rascunho de resposta em português.
* **Interface Web Intuitiva:** Uma página HTML única, estilizada com Tailwind CSS (utilizando o Play CDN), para interação do usuário.
* **Backend Eficiente:** Construído em Python utilizando o framework FastAPI, garantindo uma API rápida e robusta.

## 🛠️ Tecnologias Utilizadas

* **Backend:**
    * Python 3.9+
    * FastAPI: Framework web para construção da API.
    * Uvicorn: Servidor ASGI para rodar a aplicação FastAPI.
    * Google Generative AI SDK: Para interação com a API do Gemini (`gemini-1.5-flash-latest`).
    * NLTK: Para pré-processamento de texto (tokenização, remoção de stop words, stemming).
    * python-dotenv: Para gerenciamento de variáveis de ambiente em desenvolvimento local.
    * pdfplumber: Para extração de texto de arquivos PDF.
* **Frontend:**
    * HTML5
    * Tailwind CSS (via Play CDN): Para estilização da interface.
    * JavaScript (Vanilla): Para interatividade da página e comunicação com o backend.
* **Deployment (Tentativa):**
    * Vercel

## 🚀 Configuração e Execução Local

Siga os passos para configurar e rodar o projeto em seu ambiente local:

### Pré-requisitos Indispensáveis

* Python 3.9 ou superior.
* `pip` (gerenciador de pacotes Python).
* Uma chave de API válida do Google Gemini (obtenha no [Google AI Studio](https://aistudio.google.com/)).

### Passos para Configuração

1.  **Obtenha os Arquivos do Projeto:**
    Baixe ou clone os arquivos do projeto para um diretório em sua máquina.

2.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    Abra o terminal na pasta do projeto e execute:
    ```bash
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as Dependências:**
    Com o ambiente virtual ativo, instale as bibliotecas necessárias:
    ```bash
    pip install -r requirements.txt
    ```
    (Certifique-se de que o arquivo `requirements.txt` está presente e atualizado com todas as dependências, incluindo `fastapi`, `uvicorn`, `google-generativeai`, `nltk`, `python-dotenv`, `pdfplumber`).

4.  **Configure a Chave da API:**
    * Crie um arquivo chamado `.env` na raiz do projeto.
    * Dentro do arquivo `.env`, adicione sua chave da API:
        ```env
        GOOGLE_API_KEY="SUA_CHAVE_DA_API_DO_GEMINI_AQUI"
        ```
    * Substitua `"SUA_CHAVE_DA_API_DO_GEMINI_AQUI"` pela sua chave real.

5.  **Downloads de Recursos NLTK e Carregamento do .env:**
    A aplicação tentará baixar os recursos NLTK (punkt, stopwords, rslp) e carregar o `.env` automaticamente na inicialização do servidor FastAPI (através da função `download_nltk` chamada no evento de startup). Certifique-se de que tem conexão com a internet na primeira execução para os downloads do NLTK.

6.  **Execute o Servidor Backend:**
    No terminal (com ambiente virtual ativo e na pasta do projeto):
    ```bash
    uvicorn main:app --reload --port 5000
    ```
    Ou, alternativamente:
    ```bash
    python main.py
    ```
    O servidor estará disponível em `http://127.0.0.1:5000`.

7.  **Acesse a Aplicação:**
    Abra o arquivo `index.html` em seu navegador. A interface deve carregar e ser capaz de se comunicar com o backend local.

## ☁️ Deploy no Vercel

A aplicação foi configurada para tentativa de deploy no Vercel.

### Arquivos de Configuração para Vercel:

* **`vercel.json`**:
    ```json
    {
      "version": 2,
      "builds": [
        {
          "src": "main.py",
          "use": "@vercel/python",
          "config": {
            "maxLambdaSize": "50mb"
          }
        }
      ],
      "routes": [
        {
          "src": "/(.*)",
          "dest": "main.py"
        }
      ]
    }
    ```
* **`runtime.txt`** (para especificar a versão do Python, ex: 3.9):
    ```
    python-3.9
    ```

### Passos (via Vercel CLI):

1.  Instale o Node.js e o Vercel CLI (`npm install -g vercel`).
2.  Faça login: `vercel login`.
3.  Navegue até a pasta do projeto.
4.  Inicie o deploy: `vercel` (siga as instruções para criar/linkar o projeto).
5.  **Adicione a GOOGLE_API_KEY como variável de ambiente no Vercel:**
    ```bash
    vercel env add GOOGLE_API_KEY
    ```
    (Insira sua chave quando solicitado e aplique ao ambiente de Produção).
6.  Faça o deploy para produção: `vercel --prod`.

## 📖 Como Usar

1.  Com o backend rodando (localmente ou no Vercel, caso o problema de startup seja resolvido), abra o `index.html` no navegador.
2.  **Insira o Conteúdo do E-mail:**
    * Cole o texto diretamente na área designada.
    * OU clique em "Selecione o arquivo" para fazer upload de um arquivo `.txt` ou `.pdf`.
3.  **Análise:** Clique no botão "Analisar Email".
4.  **Resultados:** A classificação ("Produtivo" ou "Improdutivo") e uma sugestão de resposta serão exibidas abaixo do formulário.
5.  **Erros:** Mensagens de erro aparecerão caso ocorra algum problema durante o processamento.
