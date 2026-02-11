import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import uuid
from flask import Flask, request, jsonify, render_template
from TTS.api import TTS

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)
last_audio_filepath = None # To track the last generated audio file

# --- RAG Chain Setup (from prototype.ipynb) ---
# Load the HTML as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path=os.path.join(os.path.dirname(__file__), 'data', 'mg-zs-warning-messages.html'))
car_docs = loader.load()

# Initialize RecursiveCharacterTextSplitter to make chunks of HTML text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(car_docs)

# Locally create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Local LLM setup
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# RAG prompt template
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, say you don't know.

Question: {question}
Context: {context}

Answer:"""
)

# Build offline RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# --- Coqui TTS Setup ---
# This will download the model the first time it runs.
# You can choose other models from TTS.list_models()
try:
    tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False, gpu=False)
except Exception as e:
    print(f"Error initializing TTS model: {e}")
    print("Attempting to initialize with a different model or ensure model is downloaded.")
    # Fallback or alternative model if the first one fails, or provide instructions
    tts = None # Ensure tts is None if initialization fails

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global last_audio_filepath
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Get answer from RAG chain
        response = rag_chain.invoke(user_question)
        answer_text = response.content if hasattr(response, "content") else str(response)
        # Ensure answer_text ends with a period or other sentence-ending punctuation
        if answer_text and not answer_text.strip().endswith(('.', '!', '?')):
            answer_text += '.'

        audio_url = None
        if tts:
            # Clean up the previous audio file if it exists
            if last_audio_filepath and os.path.exists(last_audio_filepath):
                try:
                    os.remove(last_audio_filepath)
                except Exception as e:
                    print(f"Error deleting old audio file: {e}")
                    
            # Generate unique filename for audio
            audio_filename = f"response_{uuid.uuid4()}.wav"
            audio_filepath = os.path.join(app.static_folder, "audio", audio_filename)
            
            # Ensure static/audio directory exists
            os.makedirs(os.path.join(app.static_folder, "audio"), exist_ok=True)

            # Generate audio
            tts.tts_to_file(text=answer_text, file_path=audio_filepath)
            audio_url = f"/static/audio/{audio_filename}"
            last_audio_filepath = audio_filepath # Save the path of the new file
        else:
            print("TTS model not initialized. Skipping audio generation.")

        return jsonify({"answer_text": answer_text, "audio_url": audio_url})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure static/audio directory exists on startup
    os.makedirs(os.path.join(app.static_folder, "audio"), exist_ok=True)
    app.run(debug=True)
