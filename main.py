# Execute no terminal: `uvicorn main:app --reload`
# A API estará disponível em http://127.0.0.1:8000
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Literal, Union
import json
import uuid
import google.generativeai as genai
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

# Variável global para o modelo Gemini.
gemini_model: genai.GenerativeModel = None

# --- Lifespan Event Handler do FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Função de lifespan executada quando a aplicação FastAPI é iniciada
    e desligada. Usada para inicializar o modelo Google Gemini.
    """
    global gemini_model # Declara que estamos usando a variável global

    print("Iniciando lifespan: Configurando Google Gemini...")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("A chave da API GEMINI_API_KEY não foi encontrada nas variáveis de ambiente.")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")
        print("Google Gemini configurado e modelo inicializado com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Falha ao inicializar o modelo Gemini: {e}")

    yield
    print("Desligando a aplicação FastAPI.")

# Inicialização da aplicação FastAPI
app = FastAPI (
    title="API do Chatbot de Triagem Médica",
    description="Backend para um sistema de triagem médica utilizando a API do Google Gemini. Trata-se de um teste de conceito do projeto de PIBIC da ETC/UNICAP do aluno Marcos Filipe G. Capella.",
    lifespan=lifespan
)

# Variáveis globais para simular o armazenamento da ficha de atendimento.
ficha_de_atendimento_db: Dict[str, Dict[str, Any]] = {}

# Modelos Pydantic 
# Para os dados pessoais iniciais
class DadosPessoaisIniciais(BaseModel):
    nome_completo: str
    endereco: str
    idade: int
    
# Para mensagens do chat (do lado user e do lado bot)
class ChatMessage(BaseModel):
    role: Literal["user", "model"] # 'model' = Gemini
    parts: List[Dict[str, str]] # Lista de partes, onde cada parte é um dicionário com a chave 'text'
    
# Para entrada do chat conversacional
class ChatInput(BaseModel):
    session_id: str
    user_message: str

# Para resposta do Gemini no modo conversacional
class ConversationalGeminiResponse(BaseModel):
    status: Literal["ongoing", "final"] # O Gemini retornará um status ongoing ou final, se for final, incluirá a especialidade e a orientação.
    bot_message: str
    especialidade_medica: Union[str, None] = None
    orientacao_ao_medico: Union[str, None] = None
    
# --- Modelos abaixo servem para o modo não-conversacional do Gemini ---
# Serão mantidos como opção, podendo ser excluídos no futuro.    
# Para a queixa do paciente
class QueixaPaciente(BaseModel):
    session_id: str
    queixa: str
    
# Para a resposta da API Gemini
class GeminiResponse(BaseModel):
    especialidade_medica: str
    orientacao_ao_medico: str
    
# --- Endpoints da API ---

@app.post("/api/v1/iniciar_atendimento")
async def iniciar_atendimento(dados: DadosPessoaisIniciais):
    """
    Endpoint para iniciar o atendimento, recebendo os dados pessoais iniciais do paciente.
    Retorna um ID de sessão para acompanhar o atendimento.
    """
    session_id = str(uuid.uuid4()) # Gera um ID de sessão único
    ficha_de_atendimento_db[session_id] = {
        "session_id": session_id,
        "nome_completo": dados.nome_completo,
        "endereco": dados.endereco,
        "idade": dados.idade,
        "dados_fisiologicos": {}, # Por ora, serão artificiais
        "queixa_paciente": "",
        "especialidade_medica": "",
        "orientacao_ao_medico": "",
        "chat_history": [], # Armazenar o histórico de mensagens do modo conversacional
        "conversation_status": "ongoing" # Inicia 'ongoing' e muda para 'final'.
    }
    print(f"Atendimento iniciado para {dados.nome_completo} com session_id: {session_id}")
    return {"message": "Atendimento iniciado com sucesso.", "session_id": session_id}

@app.get("/api/v1/obter_dados_smartwatch/{session_id}")
async def obter_dados_smartwatch(session_id: str):
    """
    Endpoint para simular a coleta de dados de um smartwatch.
    Em um sistema real, aqui haveria a integração com a API do app de gestão do Smartwatch.
    Para este projeto, geramos dados artificiais.
    """
    # Conferir se a sessão existe de fato
    if session_id not in ficha_de_atendimento_db:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    # Simula dados fisiológicos do smartwatch
    dados_fisiologicos_artificiais = {
        "altura_cm": 175,
        "peso_kg": 70,
        "pressao_arterial_sistolica": 120,
        "pressao_arterial_diastolica": 80,
        "oxigenacao_sangue_percentual": 98,
        "nivel_estresse": "Baixo"
    }

    # Salvar dados obtidos no "banco de dados"
    ficha_de_atendimento_db[session_id]["dados_fisiologicos"] = dados_fisiologicos_artificiais
    print(f"Dados do smartwatch adicionados para session_id: {session_id}")
    return {
        "message": "Dados do smartwatch obtidos com sucesso.",
        "dados_fisiologicos": dados_fisiologicos_artificiais
    }

@app.post("/api/v0/processar_queixa") # Modo não-conversacional
async def processar_queixa(queixa_data: QueixaPaciente):
    """
    Endpoint principal para processar a queixa do paciente,
    interagir com a API do Google Gemini e determinar a especialidade e orientação.
    """
    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Serviço Gemini não inicializado. Tente novamente em breve.")
    
    session_id = queixa_data.session_id
    queixa_paciente = queixa_data.queixa

    if session_id not in ficha_de_atendimento_db:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    ficha = ficha_de_atendimento_db[session_id]
    ficha["queixa_paciente"] = queixa_paciente

    # Prepara os dados pessoais e fisiológicos para o prompt do Gemini
    dados_para_gemini = f"Nome: {ficha['nome_completo']}, Idade: {ficha['idade']}, Endereço: {ficha['endereco']}. " \
                        f"Dados Fisiológicos: Altura: {ficha['dados_fisiologicos'].get('altura_cm')}cm, " \
                        f"Peso: {ficha['dados_fisiologicos'].get('peso_kg')}kg, " \
                        f"Pressão Arterial: {ficha['dados_fisiologicos'].get('pressao_arterial_sistolica')}/" \
                        f"{ficha['dados_fisiologicos'].get('pressao_arterial_diastolica')} mmHg, " \
                        f"Oxigenação: {ficha['dados_fisiologicos'].get('oxigenacao_sangue_percentual')}%, " \
                        f"Nível de Estresse: {ficha['dados_fisiologicos'].get('nivel_estresse')}."

    # --- Prompt base ---
    # Este é o prompt que será enviado para a API do Google Gemini para garantir o formato JSON estrito na saída.
    gemini_prompt = (
        "Você é um atendente de triagem médica para urgências ou clínicas médicas. "
        "Seu papel é ouvir as queixas e dúvidas de saúde do usuário e colher informações suficientes para apoiar o diagnóstico médico. "
        f"Você já recebeu os seguintes dados pessoais e fisiológicos: {dados_para_gemini}. "
        "Agora, o paciente irá descrever sua queixa principal. "
        "Com base na queixa e nos dados fornecidos, defina a especialidade médica mais adequada para atendê-lo e gere uma orientação concisa para o médico. "
        "O output DEVE estar no formato JSON: {\"especialidade_medica\": \"[especialidade_aqui]\", \"orientacao_ao_medico\": \"[orientacao_aqui]\"}. "
        "Não adicione nenhum texto antes ou depois do JSON. Não estenda a conversa. "
        "Assim que o JSON for gerado, sua tarefa está completa. Apenas o JSON deve ser retornado."
    )

    # Adiciona a queixa do paciente ao prompt
    full_prompt = f"{gemini_prompt}\nQueixa do paciente: {queixa_paciente}"

    # Chamada do Google Gemini
    try:
        response = gemini_model.generate_content(full_prompt)
        gemini_output_text = response.text
        
        print(f"Resposta bruta do Gemini: {gemini_output_text}") # Para debugar
        
        # --- Tratamento para remover o bloco de código Markdown ---
        # Verifica se a resposta começa com '```json' e termina com '```'
        if gemini_output_text.strip().startswith("```json") and gemini_output_text.strip().endswith("```"):
            # Remove '```json' do início e '```' do final, e então remove espaços em branco
            json_string = gemini_output_text.strip()[len("```json"):].strip()[:-len("```")].strip()
        else:
            # Se não estiver no formato de bloco de código, assume que é JSON puro (ou tenta)
            json_string = gemini_output_text.strip()
        
        # Parsear a resposta como JSON, considerando que eu tenha de fato recebido JSON
        gemini_parsed_response = json.loads(json_string)
        gemini_response_model = GeminiResponse(**gemini_parsed_response)
        
        # Completar a ficha com a resposta do Gemini
        ficha["especialidade_medica"] = gemini_response_model.especialidade_medica
        ficha["orientacao_ao_medico"] = gemini_response_model.orientacao_ao_medico
        
        # Confirmações para debugar
        print(f"Queixa processa para session_id: {session_id}")
        print(f"Especialização: {ficha['especialidade_medica']}")
        print(f"Orientação: {ficha['orientacao_ao_medico']}")
        
        # Retornar a ficha para o front
        return {
            "message": f"Ficha de atendimento completa para session_id: {session_id} .",
            "ficha_de_atendimento": ficha
        }
        
    except json.JSONDecodeError as err:
        # Se o Gemini não retornar um JSON válido
        print(f"Erro ao parsear JSON da resposta do Gemini: {err}")
        print(f"Resposta do Gemini que causou o erro: '{gemini_output_text}'")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a resposta do Gemini. Formato JSON inválido: {err}")
    except Exception as err:
        # Captura outros erros que possam ocorrer durante a chamada à API Gemini
        print(f"Erro na chamada à API Gemini: {err}")
        raise HTTPException(status_code=500, detail=f"Erro ao interagir com a API do Google Gemini: {str(err)}")

@app.post("/api/v1/chat_with_gemini") # Modo conversacional
async def chat_with_gemini(chat_input: ChatInput):
    """
    Endpoint para gerenciar a conversa com o paciente via Google Gemini.
    A IA conduzirá a conversa para obter informações e, quando suficiente,
    determinará a especialidade e orientação.
    """
    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Serviço Gemini não inicializado. Tente novamente em breve.")
    
    session_id = chat_input.session_id
    user_message = chat_input.user_message

    if session_id not in ficha_de_atendimento_db:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    # Buscar ficha no "banco"
    ficha = ficha_de_atendimento_db[session_id]

    # Verifica se a conversa já foi finalizada para esta sessão
    if ficha["conversation_status"] == "final":
        return {
            "status": "final",
            "bot_message": "A triagem para esta sessão já foi finalizada. Por favor, inicie um novo atendimento se precisar.",
            "especialidade_medica": ficha["especialidade_medica"],
            "orientacao_ao_medico": ficha["orientacao_ao_medico"]
        }

    # Adiciona a mensagem do usuário ao histórico do chat
    ficha["chat_history"].append({"role": "user", "parts": [{"text": user_message}]})

    # Prepara os dados pessoais e fisiológicos para o prompt do Gemini
    dados_para_gemini = f"Nome: {ficha['nome_completo']}, Idade: {ficha['idade']}, Endereço: {ficha['endereco']}. " \
                        f"Dados Fisiológicos: Altura: {ficha['dados_fisiologicos'].get('altura_cm')}cm, " \
                        f"Peso: {ficha['dados_fisiologicos'].get('peso_kg')}kg, " \
                        f"Pressão Arterial: {ficha['dados_fisiologicos'].get('pressao_arterial_sistolica')}/" \
                        f"{ficha['dados_fisiologicos'].get('pressao_arterial_diastolica')} mmHg, " \
                        f"Oxigenação: {ficha['dados_fisiologicos'].get('oxigenacao_sangue_percentual')}%, " \
                        f"Nível de Estresse: {ficha['dados_fisiologicos'].get('nivel_estresse')}."

    # --- Prompt base para a conversa com o Gemini ---
    gemini_system_prompt = (
        "Você é um atendente de triagem médica para urgências ou clínicas médicas. "
        "Seu papel é ouvir as queixas e dúvidas de saúde do usuário e colher informações suficientes para apoiar o diagnóstico médico. "
        f"Os dados pessoais e fisiológicos iniciais do paciente são: {dados_para_gemini}. "
        "Você deve conduzir a conversa fazendo perguntas relevantes para obter detalhes sobre a queixa principal (localização, intensidade, duração, sintomas associados, histórico médico relevante, medicamentos em uso, etc.). "
        "Quando você sentir que tem informações suficientes para determinar a especialidade médica e uma orientação concisa para o médico, você DEVE finalizar a triagem. "
        "Se você precisar de mais informações, responda com uma pergunta clara e concisa para o paciente. "
        "Se você tiver informações suficientes, responda com um JSON no formato: "
        "{\"status\": \"final\", \"bot_message\": \"Obrigado pelas informações. Avaliei seu caso.\", \"especialidade_medica\": \"[especialidade_aqui]\", \"orientacao_ao_medico\": \"[orientacao_aqui]\"}. "
        "Se precisar de mais informações, responda com um JSON no formato: "
        "{\"status\": \"ongoing\", \"bot_message\": \"[sua_pergunta_aqui]\"}. "
        "Não adicione nenhum texto antes ou depois do JSON. Apenas o JSON deve ser retornado."
    )

    # Prepara o histórico do chat para o Gemini, incluindo o prompt do sistema como a primeira mensagem do "modelo" e depois as mensagens reais da conversa.
    # O Gemini API espera o formato [{ 'role': 'user', 'parts': [{ 'text': '...' }] }, { 'role': 'model', 'parts': [{ 'text': '...' }] }]
    # Para o prompt do sistema, o role 'model' é usado para dar contexto inicial.
    conversation_for_gemini = [
        {"role": "model", "parts": [{"text": gemini_system_prompt}]}
    ] + ficha["chat_history"]

    # Chamada do Google Gemini
    try:
        # A chamada para gerar conteúdo. O 'conversation_for_gemini' é o que enviamos ao Gemini.
        # O Gemini API usa o histórico para manter o contexto.
        response = gemini_model.generate_content(conversation_for_gemini)
        gemini_output_text = response.text

        print(f"Resposta bruta do Gemini: {gemini_output_text}") # Para debugar

        # --- Tratamento para remover o bloco de código Markdown ---
        # Verifica se a resposta começa com '```json' e termina com '```'
        if gemini_output_text.strip().startswith("```json") and gemini_output_text.strip().endswith("```"):
            # Remove '```json' do início e '```' do final, e então remove espaços em branco
            json_string = gemini_output_text.strip()[len("```json"):].strip()[:-len("```")].strip()
        else:
            # Se não estiver no formato de bloco de código, assume que é JSON puro (ou tenta)
            json_string = gemini_output_text.strip()

        # Parsear a resposta como JSON
        gemini_parsed_response = json.loads(json_string)
        gemini_response_model = ConversationalGeminiResponse(**gemini_parsed_response)

        # Adiciona a resposta do Gemini ao histórico do chat
        ficha["chat_history"].append({"role": "model", "parts": [{"text": gemini_response_model.bot_message}]})

        if gemini_response_model.status == "final":
            # Se a triagem foi finalizada, atualiza a ficha e o status da conversa
            ficha["especialidade_medica"] = gemini_response_model.especialidade_medica
            ficha["orientacao_ao_medico"] = gemini_response_model.orientacao_ao_medico
            # Consolida a queixa final a partir do histórico de chat
            ficha["queixa_paciente"] = "\n".join([
                msg["parts"][0]["text"] for msg in ficha["chat_history"] if msg["role"] == "user"
            ])
            ficha["conversation_status"] = "final"
            print(f"Triagem finalizada para session_id: {session_id}")
            print(f"Especialidade: {ficha['especialidade_medica']}")
            print(f"Orientação: {ficha['orientacao_ao_medico']}")

            return {
                "status": "final",
                "bot_message": gemini_response_model.bot_message,
                "ficha_de_atendimento": ficha
            }
        else: # status == "ongoing"
            print(f"Conversa em andamento para session_id: {session_id}")
            return {
                "status": "ongoing",
                "bot_message": gemini_response_model.bot_message
            }

    except json.JSONDecodeError as err:
        print(f"Erro ao parsear JSON da resposta do Gemini: {err}")
        print(f"String JSON que causou o erro: '{json_string}'")
        print(f"Resposta bruta original do Gemini: '{gemini_output_text}'")
        raise HTTPException(status_code=500, detail=f"Erro ao processar a resposta do Gemini. Formato JSON inválido: {err}")
    except Exception as err:
        print(f"Erro na chamada à API Gemini: {err}")
        raise HTTPException(status_code=500, detail=f"Erro ao interagir com a API do Google Gemini: {str(err)}")

# Endpoint para obter a ficha de atendimento completa (para o frontend do médico)
@app.get("/api/v1/ficha_completa/{session_id}")
async def obter_ficha_completa(session_id: str):
    """
    Endpoint para obter a ficha de atendimento completa de uma sessão específica.
    Este endpoint seria usado pelo frontend do médico.
    """
    if session_id not in ficha_de_atendimento_db:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return {"ficha_de_atendimento": ficha_de_atendimento_db[session_id]}