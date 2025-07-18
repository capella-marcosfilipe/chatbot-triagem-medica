import os
from dotenv import load_dotenv
import google.generativeai as genai

print("Iniciando teste de conexão Gemini...")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERRO: A chave da API GEMINI_API_KEY não foi encontrada nas variáveis de ambiente.")
    exit()

try:
    print("Configurando a API Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    print("Configuração da API Gemini concluída com sucesso.")

    print("Inicializando o modelo Gemini...")
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    print("Modelo Gemini inicializado com sucesso.")

    print("Fazendo uma chamada de teste simples para o Gemini...")
    test_prompt = "Diga olá."
    response = model.generate_content(test_prompt)
    print("Chamada de teste concluída.")

    print("\nResposta do Gemini:")
    print(response.text)
    print("\nTeste de conexão Gemini finalizado com sucesso!")

except Exception as e:
    print(f"\nERRO: Ocorreu um problema ao se conectar ou usar a API Gemini: {e}")
    if "network" in str(e).lower() or "timeout" in str(e).lower():
        print("Verifique sua conexão com a internet ou configurações de proxy.")
    if "api key" in str(e).lower():
        print("Verifique se sua chave de API está correta e ativa.")