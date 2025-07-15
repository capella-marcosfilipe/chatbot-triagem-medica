# Execute no terminal: `uvicorn main:app --reload`
# A API estará disponível em http://127.0.0.1:8000
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json
import uuid
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Inicialização da aplicação FastAPI
app = FastAPI (
    title="API do Chatbot de Triagem Médica",
    description="Backend para um sistema de triagem médica utilizando a API do Google Gemini. Trata-se de um teste de conceito do projeto de PIBIC da ETC/UNICAP do aluno Marcos Filipe G. Capella."
)

# Configurações do Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("A chave da API GEMINI_API_KEY não foi encontrada nas variáveis de ambiente.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Variáveis globais para simular o armazenamento da ficha de atendimento.
ficha_de_atendimento_db: Dict[str, Dict[str, Any]] = {}

# Modelos Pydantic 
# Para os dados pessoais iniciais
class DadosPessoaisIniciais(BaseModel):
    nome_completo: str
    endereco: str
    idade: int
    
# Para a queixa do paciente
class QueixaPaciente(BaseModel):
    sessions_id: str
    queixa: str
    
# Para a resposta da API Gemini
class GeminiResponse(BaseModel):
    especialidade_medica: str
    orientacao_ao_medico: str
    
# --- Endpoints da API ---

@app.post("/api/iniciar_atendimento")
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
        "dados_fisiologicos": {}, # Inicializa dados fisiológicos vazios
        "queixa_paciente": "",
        "especialidade_medica": "",
        "orientacao_ao_medico": ""
    }
    print(f"Atendimento iniciado para {dados.nome_completo} com session_id: {session_id}")
    return {"message": "Atendimento iniciado com sucesso.", "session_id": session_id}

@app.get("/api/obter_dados_smartwatch/{session_id}")
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

@app.post("/api/processar_queixa")
async def processar_queixa(queixa_data: QueixaPaciente):
    """
    Endpoint principal para processar a queixa do paciente,
    interagir com a API do Google Gemini e determinar a especialidade e orientação.
    """
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

    # --- Prompt base corrigido e inserido no backend ---
    # Este é o prompt que será enviado para a API do Google Gemini.
    # Ele foi ajustado para garantir o formato JSON estrito na saída.
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
        response = model.generate_content(full_prompt)
        gemini_output_text = response.text
        
        print(f"Resposta bruta do Gemini: {gemini_output_text}") # Para debugar
        
        # Parsear a resposta como JSON, considerando que eu tenha de fato recebido JSON
        gemini_parsed_response = json.loads(gemini_output_text)
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

# Endpoint para obter a ficha de atendimento completa (para o frontend do médico)
@app.get("/api/ficha_completa/{session_id}")
async def obter_ficha_completa(session_id: str):
    """
    Endpoint para obter a ficha de atendimento completa de uma sessão específica.
    Este endpoint seria usado pelo frontend do médico.
    """
    if session_id not in ficha_de_atendimento_db:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return {"ficha_de_atendimento": ficha_de_atendimento_db[session_id]}