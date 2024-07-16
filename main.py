
import os

from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from twilio.request_validator import RequestValidator
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Cargar variables de entorno
load_dotenv()

# Configuración de Flask
app = Flask(__name__)

# Configuración de Google AI y Twilio
google_api_key = os.getenv('GOOGLE_API_KEY')
twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
#validator = RequestValidator(twilio_auth_token)

# Configuración del prompt
prompt_template = """ En el plan de gobierno peronista

Context:

{context}

Question:

{question}

Answer:

"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Configuración del modelo LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1.0, google_api_key=google_api_key)

# Configuración de la cadena QA
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", verbose=True, prompt=prompt)

def validate_twilio_request():
    url = request.url
    params = request.form
    signature = request.headers.get('X-Twilio-Signature', '')
    return validator.validate(url, params, signature)

@app.route("/webhook", methods=['POST'])
def webhook():
    # Validar que la solicitud viene de Twilio
    #if not validate_twilio_request():
    #    abort(403)  # Forbidden

    # Obtener el mensaje entrante de WhatsApp
    incoming_msg = request.values.get('Body', '').lower()

    # Configuración de embeddings y vectorstore
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

    # Realizar búsqueda de similitud
    resultados_similares = vectordb.similarity_search(incoming_msg, k=15)

    try:
        # Procesar la pregunta con el sistema RAG
        respuesta = qa_chain({"input_documents": resultados_similares, "question": incoming_msg})

        # Crear una respuesta de Twilio con la respuesta del RAG
        twiml_response = MessagingResponse()
        twiml_response.message(respuesta["output_text"])

        # Logging (opcional, pero útil para debugging)
        print(f"Processed message for account {twilio_account_sid}")

        # Devolver la respuesta de Twilio
        return str(twiml_response)

    except Exception as e:
        # En caso de error, enviar un mensaje de error genérico
        error_response = MessagingResponse()
        error_response.message("Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, inténtalo de nuevo más tarde.")
        print(f"Error for account {twilio_account_sid}: {str(e)}")
        return str(error_response)

if __name__ == "__main__":
    app.run(debug=True)

