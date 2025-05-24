import os
import pdfplumber
from typing import Optional  # Mantido pois é usado por Form() e File()

import nltk
import uvicorn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # ADICIONADO
from pydantic import BaseModel
import io


def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')
    load_dotenv()


def start_api(api_key_google_param):
    modelo_gemini = None
    if api_key_google_param:
        try:
            genai.configure(api_key=api_key_google_param)
            modelo_gemini = genai.GenerativeModel('gemini-1.5-flash-latest')
            print(f"Modelo Gemini configurado com exito.")
        except Exception as e:
            print(f"Erro! Não foi possível configurar o modelo Gemini: {e}")
    else:
        print(f"Erro! Chave da API não encontrada:")
    return modelo_gemini


class AnaliseResponse(BaseModel):
    classificacao: str
    sugestao_resposta: str


def pre_processamento_nltk(texto):
    if not texto: return ""
    try:
        tokens = word_tokenize(texto.lower(), language='portuguese')
        stop_words_pt = set(stopwords.words('portuguese'))
        palavras_filtradas = [token for token in tokens if token.isalnum() and token not in stop_words_pt]
        stemmer = RSLPStemmer()
        palavras_stemmed = [stemmer.stem(palavra) for palavra in palavras_filtradas]
        return " ".join(palavras_stemmed)
    except LookupError as e:
        print(f"Erro! Problema no pro processamento do NLTK: {e}")
        return texto


def get_resposta_ia(modelo_gemini, prompt_completo):
    generation_config = genai.types.GenerationConfig(max_output_tokens=350, temperature=0.2)
    resposta_ia_obj = modelo_gemini.generate_content(prompt_completo, generation_config=generation_config)
    if resposta_ia_obj.parts:
        resposta_ia_texto = "".join(part.text for part in resposta_ia_obj.parts).strip()
    elif hasattr(resposta_ia_obj, 'text') and resposta_ia_obj.text:
        resposta_ia_texto = resposta_ia_obj.text.strip()
    else:
        if resposta_ia_obj.candidates and resposta_ia_obj.candidates[0].content.parts:
            resposta_ia_texto = "".join(
                part.text for part in resposta_ia_obj.candidates[0].content.parts).strip()
        else:
            raise AttributeError("Nenhuma parte de texto encontrada.")

    return resposta_ia_texto, resposta_ia_obj


def get_classificacao_e_sugestao(resposta_ia_texto, resposta_ia_obj):
    if not resposta_ia_texto:
        if (hasattr(resposta_ia_obj, 'prompt_feedback') and
                hasattr(resposta_ia_obj.prompt_feedback, 'block_reason') and
                resposta_ia_obj.prompt_feedback.block_reason):
            block_reason_message = getattr(resposta_ia_obj.prompt_feedback, 'block_reason_message',
                                           "Motivo não especificado")
            return {"classificacao": "Bloqueado pela API",
                    "sugestao_resposta": f"Conteúdo bloqueado: {block_reason_message}"}
        return {"classificacao": "Erro na API", "sugestao_resposta": "Resposta da API do Gemini vazia."}

    classificacao, sugestao = "Não classificado", "Não foi possível sugerir uma resposta."
    linhas, achou_classificacao = resposta_ia_texto.split('\n'), False
    for linha in linhas:
        linha_strip = linha.strip()
        if linha_strip.startswith("Classificação:"):
            classificacao = linha_strip.replace("Classificação:", "").strip()
            achou_classificacao = True

        elif achou_classificacao and linha_strip.startswith("Sugestão de Resposta:"):
            sugestao = linha_strip.replace("Sugestão de Resposta:", "").strip()
            break

        elif achou_classificacao and sugestao == "Não foi possível sugerir uma resposta." and linha_strip:
            sugestao = linha_strip

    return {"classificacao": classificacao, "sugestao_resposta": sugestao}


def classificar_e_sugerir_resposta_gemini(texto_email_original, texto_email_preprocessado, modelo_gemini):
    prompt_completo = f"""
    Sua tarefa é analisar um email e classificá-lo como 'Produtivo' ou 'Improdutivo', e então sugerir uma resposta 
    apropriada em português. O email fornecido para análise pode ter passado por um pré-processamento (remoção de stop words, radicalização).

    Definições Importantes:
    - Produtivo: Emails que requerem uma ação ou resposta específica. Exemplos: solicitações de suporte técnico, pedidos
     de atualização sobre casos em aberto, dúvidas sobre o sistema, pedidos de informação concreta, envio de documentos 
     que exigem revisão ou ação, marcar ou desmarcar reunião.
    - Improdutivo: Emails que não necessitam de uma ação imediata ou específica relacionada ao trabalho, ou são de
     natureza social/informativa geral. Exemplos: mensagens de felicitações (Feliz Natal, feliz aniversário), 
     agradecimentos genéricos, spams, newsletters não solicitadas, convites para eventos sociais não relacionados 
     diretamente ao trabalho, códigos promocionais, cupons promocionais.

    Sua resposta DEVE seguir EXATAMENTE o formato abaixo, sem textos ou explicações adicionais antes ou depois das
     seções 'Classificação' e 'Sugestão de Resposta':
    Classificação: [Produtivo/Improdutivo]
    Sugestão de Resposta: [Texto da resposta aqui]

    ---
    Exemplos de Classificação e Resposta (baseados em texto natural, não pré-processado):

    Email Exemplo 1:
    ---
    Olá Ana,
    Gostaria de saber se há alguma atualização sobre o ticket de suporte #12345 que abri ontem sobre o problema de login.
    Aguardo ansiosamente.
    Obrigado, João.
    ---
    Classificação: Produtivo
    Sugestão de Resposta: Olá João, obrigado por entrar em contato. Vou verificar o status do ticket #12345 e retorno com uma atualização sobre o problema de login em breve.

    Email Exemplo 2:
    ---
    Caros colegas,
    Desejo a todos um excelente final de semana prolongado! Aproveitem para descansar.
    Abraços, Maria
    ---
    Classificação: Improdutivo
    Sugestão de Resposta: Olá Maria, agradecemos os votos! Um ótimo final de semana prolongado para você também.

    Email Exemplo 3:
    ---
    Prezada equipe,
    Segue em anexo o relatório financeiro do mês de Abril para vossa análise e aprovação.
    Por favor, confirmem o recebimento e me informem caso necessitem de alguma clarificação.
    Atenciosamente, Pedro.
    ---
    Classificação: Produtivo
    Sugestão de Resposta: Olá Pedro, relatório financeiro de Abril recebido. Iremos analisar e confirmaremos a aprovação ou enviaremos quaisquer dúvidas em breve. Obrigado!

    Email Exemplo 4:
    ---
    Oi, tudo bem por aí? Só passando pra dar um oi!
    ---
    Classificação: Improdutivo
    Sugestão de Resposta: Olá! Tudo bem por aqui, obrigado por perguntar.

    Email Exemplo 5: 
    ---
    Prezados líderes de frente,
    Conforme nosso breve alinhamento de ontem, convoco a todos para uma reunião crucial de realinhamento estratégico do Projeto Ômega. A participação de todos é indispensável.
    Data: 23 de Maio de 2025 (Sexta-feira) Horário: 09:00 - 11:00 Local: Sala de Conferências Principal.
    Pauta Detalhada: Desvios de Cronograma, Bloqueios Técnicos, Plano de Ação Corretivo. Material de apoio em anexo.
    Confirmem presença até 22/05.
    Atenciosamente, Carlos Andrade, Diretor de Operações
    ---
    Classificação: Produtivo
    Sugestão de Resposta: Presença confirmada para a reunião do Projeto Ômega em 23/05. O material de apoio será revisado previamente.

    Email Exemplo 6:
    ---
    Equipe de Suporte Nível 3 e Desenvolvimento,
    Falha crítica generalizada no Módulo de Pagamentos (v2.3.1) em produção desde 08:15 de hoje. Transações com cartão intermitentemente recusadas, algumas com cobrança duplicada. PIX/Boleto OK.
    Ações já Tomadas (sem sucesso): Restart dos serviços, verificação de conectividade. Rollback para v2.3.0 impossível devido a dependência crítica.
    Logs detalhados enviados em e-mail separado. Impacto: Paralisação parcial do faturamento, alta insatisfação. Necessitamos de solução urgente.
    Por favor, acusem recebimento e informem ETA para diagnóstico.
    Grato pela urgência, Fernando Costa, Gerente de E-commerce
    ---
    Classificação: Produtivo
    Sugestão de Resposta: Fernando, e-mail e urgência recebidos. A equipe já está analisando os logs e iniciando o diagnóstico da falha no módulo de pagamentos. Manteremos você informado sobre o progresso e um ETA assim que possível.

    Email Exemplo 7:
    ---
    Prezados, Carla (RH) e Tiago (Suporte TI),
    Escrevo em nome do Sr. Roberto Almeida (Patrimônio, 68 anos, conosco há 25 anos), que tem enfrentado dificuldades consideráveis com o novo Sistema de Inventário (SISPAT v3), gerando angústia e impactando sua produtividade. Ele mencionou dificuldades com: geração de relatórios customizados, processo de baixa de ativos e utilização do leitor de código de barras.
    Gostaria de solicitar: treinamento individualizado para ele, avaliação de um guia rápido simplificado, e verificação de opções de acessibilidade no SISPAT v3.
    Agradeço a atenção. Beatriz Lima, Gerente Administrativa
    ---
    Classificação: Produtivo
    Sugestão de Resposta: Prezada Beatriz, agradecemos o contato e a preocupação com o Sr. Roberto. Vamos analisar internamente as sugestões de treinamento, material de apoio e opções de acessibilidade para o SISPAT v3 e retornaremos em breve com um plano de ação.

    Email Exemplo 8:
    ---
    Atenção, Super Time AutoU! É com imensa alegria que anunciamos os detalhes finais da nossa Festa de Fim de Ano: 'Noite Estrelada AutoU'!
    Data: 13 de Dezembro (Sábado), 20:00-02:00. Local: Espaço Blue Moon Eventos. Dress Code: Gala Estrelado!
    Buffet Premium, Open Bar VIP, Show com Banda 'Cosmic Hits', DJ, Cabine de Fotos e Sorteios! Amigo Secreto Estelar (R$70-R$100).
    RSVP Urgente até 01/12 via link: [link_do_rsvp_aqui].
    Comissão Organizadora 'Noite Estrelada AutoU'
    ---
    Classificação: Improdutivo
    Sugestão de Resposta: Que ótima notícia! Obrigado pelos detalhes da festa 'Noite Estrelada AutoU'. Presença confirmada!

    Email Exemplo 9:
    ---
    Querida Sofia,
    Palavras me faltam para expressar o quão grato estou pela sua colaboração no Projeto Solaris. Sua dedicação foi crucial. Lembro da sua perspicácia com os dados legados e sua calma sob pressão. Sua condução da reunião com o cliente foi fundamental. Muito obrigado!
    Com admiração, Ricardo Mendes
    ---
    Classificação: Improdutivo
    Sugestão de Resposta: Ricardo, suas palavras são muito gentis! Fico imensamente feliz por ter contribuído para o sucesso do Projeto Solaris. Muito obrigada pelo reconhecimento!

    Email Exemplo 10:
    ---
    Meu caro Leonardo,
    Sua participação na apresentação para a 'Gigante Global Corp' foi brilhante! A forma como traduziu conceitos complexos e seu insight de última hora foram magistrais. Sua tranquilidade nas respostas e domínio técnico encantaram a todos. Muito obrigado, Leo! Você é fera!
    Abraços, Patrícia Vasconcelos, Diretora Comercial
    ---
    Classificação: Improdutivo
    Sugestão de Resposta: Patrícia, que feedback incrível! Fico muito feliz e honrado com suas palavras. Foi um prazer contribuir para o sucesso da apresentação. Obrigado!
    ---
    Agora, analise o seguinte email (que pode estar pré-processado) e forneça a classificação e sugestão de resposta no formato especificado:

    Email para Análise:
    ---
    {texto_email_preprocessado}
    ---
    Texto Original (para referência ao gerar a sugestão de resposta, se necessário, mas sua sugestão deve ser concisa):
    ---
    {texto_email_original}
    ---
    """
    try:
        resposta_ia_texto, resposta_ia_obj = get_resposta_ia(modelo_gemini, prompt_completo)
        classifc_e_sugest = get_classificacao_e_sugestao(resposta_ia_texto, resposta_ia_obj)
        return classifc_e_sugest
    except Exception as e:
        print(f"Erro! Problema na chamada da API Gemini: {type(e).__name__} - {e}")
        error_details = str(e)
        if hasattr(e, 'grpc_status_code'):
            error_details = f"gRPC Error {e.grpc_status_code()}: {e.message}"
        elif hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_details = f"{str(e)} - Detalhes: {e.response.text}"
        return {"classificacao": "Erro na API",
                "sugestao_resposta": f"Ocorreu um erro ao processar o email com Gemini: {error_details}"}


app = FastAPI(
    title="Analisador de Produtividade de Emails API",
    version="1.0.3",
    description="Uma API para classificar emails (texto ou arquivo .txt/.pdf) como produtivos ou improdutivos e sugerir respostas."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def tarefas_de_inicializacao():
    download_nltk()
    api_key = os.getenv("GOOGLE_API_KEY")
    app.state.modelo_gemini = start_api(api_key)
    if not app.state.modelo_gemini and api_key:  # Se tinha chave mas o modelo não carregou
        print("ALERTA FastAPI: Modelo Gemini não foi carregado na inicialização (apesar da chave API estar presente).")
    # Se não tinha chave, start_api() já imprimiu "Erro! Chave da API não encontrada:"
    # Não precisamos de outro print para "Chave API não encontrada" aqui.


# NOVA ROTA PARA SERVIR O INDEX.HTML
@app.get("/", include_in_schema=False)
async def servir_pagina_principal():
    caminho_base = os.path.dirname(os.path.abspath(__file__))
    caminho_index_html = os.path.join(caminho_base, "index.html")

    if os.path.exists(caminho_index_html):
        return FileResponse(caminho_index_html)
    else:
        print(f"Tentativa de servir index.html de {caminho_index_html}, mas o arquivo não foi encontrado.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Arquivo index.html principal não encontrado no servidor.")


@app.post("/processar_email", response_model=AnaliseResponse, tags=["Análise de Email"])
async def processar_requisicao_email(
        request: Request,
        texto_email: Optional[str] = Form(None),
        arquivo_email: Optional[UploadFile] = File(None)
):
    instancia_modelo_gemini = request.app.state.modelo_gemini
    if not instancia_modelo_gemini:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Serviço de IA (Modelo Gemini) não inicializado.")

    texto_original_extraido = ""
    if arquivo_email:
        filename = arquivo_email.filename
        print(f"[FastAPI] Arquivo recebido: {filename}")

        if not (filename.lower().endswith(".txt") or filename.lower().endswith(".pdf")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Formato de arquivo não suportado. Use .txt ou .pdf.")
        contents = await arquivo_email.read()
        await arquivo_email.close()
        if filename.lower().endswith(".txt"):
            try:
                texto_original_extraido = contents.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    texto_original_extraido = contents.decode("latin-1")
                except Exception as e_decode:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                        detail=f"Erro ao decodificar arquivo .txt: {e_decode}")
        elif filename.lower().endswith(".pdf"):
            try:
                # import pdfplumber # Já importado no topo
                with pdfplumber.open(io.BytesIO(contents)) as pdf:
                    all_text_pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                    texto_original_extraido = "\n".join(all_text_pages)
                if not texto_original_extraido.strip():
                    print(f"Aviso: Não foi possível extrair texto do PDF '{filename}'.")
                    texto_original_extraido = ""
            except ImportError:
                print("Erro! Biblioteca pdfplumber não instalada.")
                raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                                    detail="Processamento de PDF não habilitado no servidor.")
            except Exception as e_pdf:
                print(f"Erro! Não foi possível processar PDF '{filename}': {e_pdf}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Erro ao processar arquivo PDF '{filename}': {str(e_pdf)}")
    elif texto_email:
        texto_original_extraido = texto_email
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Nenhum texto de email ou arquivo fornecido.")

    if not texto_original_extraido.strip() and arquivo_email and arquivo_email.filename.lower().endswith(".pdf"):
        print(f"[FastAPI] PDF '{arquivo_email.filename}' resultou em texto extraído vazio.")

    print(
        f"\n[FastAPI] Texto original extraído (primeiros 100 chars): \"{texto_original_extraido[:100].replace(os.linesep, ' ')}...\"")
    texto_preprocessado = pre_processamento_nltk(texto_original_extraido)
    print(f"[FastAPI] Texto pré-processado (primeiros 100 chars): \"{texto_preprocessado[:100]}...\"")

    try:
        resultado = classificar_e_sugerir_resposta_gemini(texto_original_extraido, texto_preprocessado,
                                                          instancia_modelo_gemini)
        print(f"[FastAPI] Resultado da IA: {resultado}")
        if resultado.get("classificacao") == "Erro na API" or resultado.get("classificacao") == "Bloqueado pela API":
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=resultado.get("sugestao_resposta"))
        return resultado
    except Exception as e:
        print(f"Erro inesperado no endpoint ao chamar Gemini: {type(e).__name__} - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Erro interno no servidor: {str(e)}")


if __name__ == "__main__":
    print("Servidor FastAPI iniciando...")
    print("Para interagir, abra o arquivo index.html no seu navegador.")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)