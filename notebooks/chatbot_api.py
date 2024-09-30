import warnings

# Suppress specific UserWarning related to Flask-Limiter in-memory storage
warnings.filterwarnings("ignore", category=UserWarning, module='flask_limiter')

# All other imports
import re
import openai
import numpy as np
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.objects import ObjectIndex
from typing import Sequence
from llama_index.core.tools import BaseTool, FunctionTool
import pickle
import os
import yaml
import json
import utils
from flask import Flask, request, jsonify
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize OpenAI and LlamaIndex settings
Settings.llm = OpenAI(api_key=utils.get_api_key(name='openai'), model="gpt-4o-mini", temperature=0.5, max_tokens=1024)
Settings.embed_model = OpenAIEmbedding(api_key=utils.get_api_key(name='openai'), model="text-embedding-3-large", embed_batch_size=10, dimensions=1024)
book = SimpleDirectoryReader("../data/books").load_data()

# Custom text cleaning component
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes

# Data ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1000, chunk_overlap=120),
        TextCleaner(),
    ],
)
nodes = pipeline.run(documents=book)

# Vector index and engines
vector_index = VectorStoreIndex(nodes=nodes, show_progress=True)
chat_engine = vector_index.as_chat_engine(chat_mode="openai", verbose=True)
query_engine = vector_index.as_query_engine()

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the rate limiter
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

@app.route("/ask", methods=["POST"])
@limiter.limit("100 per hour")
def ask():
    try:
        data = request.get_json()
        user_query = data.get('query')

        if not user_query:
            return jsonify({"error": "Please provide a query"}), 400

        # Input sanitization
        user_query = user_query.strip().lower()

        logging.info(f"Received query: {user_query}")

        # Step 1: Retrieve relevant documents using the query engine
        try:
            query_response = query_engine.query(user_query)

            # Debug: Log the structure of the Response object
            logging.info(f"Query Response Attributes: {dir(query_response)}")

            # Extract data from the Response object
            if hasattr(query_response, 'source_nodes') and isinstance(query_response.source_nodes, list):
                serialized_query_response = [
                    {"node_text": node.node.text, "score": node.score} for node in query_response.source_nodes
                ]
            elif hasattr(query_response, 'response') and isinstance(query_response.response, str):
                serialized_query_response = [{"response_text": query_response.response}]
            else:
                logging.error("Query response format is not recognized.")
                return jsonify({"error": "Query response has invalid format"}), 500

        except Exception as e:
            logging.error(f"Error in query engine: {e}", exc_info=True)
            return jsonify({"error": "Error processing query engine request"}), 500

        # Step 2: Generate a response using the chat engine
        try:
            chat_prompt = f"Based on the following information: {serialized_query_response}, answer the question: {user_query}"
            chat_response = chat_engine.chat(chat_prompt)

            # Debugging: Log the chat response
            logging.info(f"Chat Engine Response: {chat_response}")

        except Exception as e:
            logging.error(f"Error in chat engine: {e}", exc_info=True)
            return jsonify({"error": "Error processing chat engine request"}), 500

        # Step 3: Return the response as JSON
        return jsonify({
            "query_response": serialized_query_response,
            "chat_response": chat_response
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
