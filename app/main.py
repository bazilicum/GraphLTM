"""
app.py

Main Flask application, routes, and SocketIO setup.
"""

from config import Config
import os
import uuid
import json
import sys

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import nltk
from sentence_transformers import SentenceTransformer

# Import our modules
from services.pdf_processor import PDFProcessor
from clients.db_client import get_db_client
from services.memory_service import (
    MemoryService
)
from clients.llm_client import get_llm_client
from utils.embedding import generate_embedding
from utils.logging_utils import setup_logging
from utils.conversation_history import ConversationHistory

# -----------------------------------------------------------------------------
#                         NLTK and Logging Configuration
# -----------------------------------------------------------------------------
config = Config()
logger = setup_logging(config, __name__)

nltk.download('punkt_tab')
nltk.download("punkt")

# -----------------------------------------------------------------------------
#                           Flask & SocketIO Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

# -----------------------------------------------------------------------------
#                       Model and DB Configuration
# -----------------------------------------------------------------------------

# initialize embedding model
model = SentenceTransformer(config.get("embeddings", "model"))

# initialize llm client
aux_llm_model = get_llm_client(functionality="auxiliary")
main_llm_client = get_llm_client(functionality="main_chat_loop")

# Initialize database client
db_client = get_db_client(config,model.get_sentence_embedding_dimension())

# Initialize memory service
memory_service = MemoryService(
    db_client=db_client,
    main_llm_client=main_llm_client,
    aux_llm_client=aux_llm_model,
    config=config,
    logger=logger
)

# -----------------------------------------------------------------------------
#                           Upload Folder Config
# -----------------------------------------------------------------------------
upload_folder = config.get('web_server', 'upload_folder')
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# -----------------------------------------------------------------------------
#                                Routes
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    """
    Redirects to the chat interface as the default page.
    """
    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    """
    Renders the chat interface.
    """
    save_to_memory_default = config.get('memory', 'long_term', 'write', 'enabled')
    return render_template('chat.html', active_tab='chat', save_to_memory_default=save_to_memory_default)

@app.route('/chat', methods=['POST'])
def process_chat():
    """
    Process chat messages and return responses.
    """
    try:
        # Initialize conversation history with configurable max turns
        conversation_history = ConversationHistory.get_instance(
            max_turns=config.get('memory', 'short_term', 'write', 'saved_turns')
        )
        logger.info(f"[START OF TURN]")
        data = request.get_json()
        message = data.get('message')
        save_to_memory = data.get('save_to_memory', config.get('memory', 'long_term', 'write', 'enabled'))
        if isinstance(save_to_memory, str):
            save_to_memory = save_to_memory.lower() == 'true'
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
            
        # Process the message using the memory service. This can involve LLM calls for query generation (aux LLM).
        logger.info(f"Generating retrieval queries and context")
        message_list = []
        try:
            message_list = memory_service.process_context_retrieval(message)
        except Exception as retrieval_e:
            logger.error(f"Error during context retrieval: {retrieval_e}")
            return jsonify({"response": "Could not connect to the LLM service. Please check its status and configuration."}), 500

        # Get assistant response from the main LLM
        logger.info(f"Generating LLM response")
        try:
            response = memory_service.main_llm_client.chat_generate(
            prompt=message,
            message_list=message_list,
            temperature=0.7,
                max_tokens=config.get('llm', config.get('llm', 'functionality', 'main_chat_loop', 'provider'), 'assistant_response_max_tokens')
            )
        except Exception as llm_generate_e:
            logger.error(f"Error during main LLM generation: {llm_generate_e}")
            return jsonify({"response": "Could not connect to the LLM service. Please check its status and configuration."}), 500
        
        # Ensure the LLM returned valid content
        if isinstance(response, dict) and "content" in response:
            response = response["content"]
        else:
            # If LLM didn't return expected content, it's likely a failure.
            # Log the full response object for debugging, but return a generic error.
            logger.error(f"LLM chat_generate returned unexpected response format: {response}")
            raise ValueError("LLM failed to generate a valid response. Please try again.")
            
        # Store the interaction in long-term memory if enabled
        logger.info(f"Processing memory storage")
        memory_id = None
        if save_to_memory:
            memory_id = memory_service.process_context_storing(message, response)
            logger.info(f"Storing process completed")
        else:
            logger.info(f"LTM storage disabled")

        # Add turn to in-memory conversation history
        logger.info(f"Adding turn to in-memory conversation history")
        conversation_history.add_turn(message, response, memory_id)

        logger.info(f"[END OF TURN]")
        
        return jsonify({"response": response})
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        if "connection error" in str(e).lower() or "llm" in str(e).lower():
            error_message = "Failed to communicate with the LLM service. Please check its status and configuration."
        
        logger.error(f"Error processing chat message: {e}")
        return jsonify({"error": error_message}), 500

@app.route('/upload_pdf')
def view_upload_pdf():
    """
    Renders the PDF upload interface with list of sources.
    """
    try:
        sources = db_client.get_sources()
        return render_template('upload_pdf.html', active_tab='upload', sources=sources)
    except Exception as e:
        logger.error(f"Error fetching sources: {e}")
        return render_template('upload_pdf.html', active_tab='upload', sources=[])

@app.route('/delete_source', methods=['POST'])
def delete_source():
    """
    Delete a knowledge source from the DB.
    Removes all Knowledge nodes with the specified source name.
    """
    try:
        source = request.form.get('source')
        if not source:
            return jsonify({"error": "Source name is required."}), 400

        # Check if source exists
        existing_sources = db_client.get_sources()
        if source not in existing_sources:
            return jsonify({"error": f"Source name '{source}' does not exist."}), 404

        # Delete the source
        success = db_client.delete_source(source)
        if success:
            return jsonify({"message": f"Source '{source}' has been successfully deleted."}), 200
        else:
            return jsonify({"error": f"Failed to delete source '{source}'."}), 500
    except Exception as e:
        logger.error(f"Error deleting source: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/sources', methods=['GET'])
def api_get_sources():
    """
    Get a list of all sources from the DB.
    Returns:
        JSON response with the list of sources.
    """
    try:
        sources = db_client.get_sources()
        return jsonify({
            "success": True,
            "sources": sources
        }), 200
    except Exception as e:
        logger.error(f"Error fetching sources: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/add_source', methods=['POST'])
def add_source():
    try:
        # Generate new source name
        source = f"source-{uuid.uuid4().hex[:6]}"

        # TODO: Save source to your DB or Qdrant here

        # Redirect to the upload page with the source as a query param
        return redirect(url_for('upload_pdf', source=source))
    
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handle PDF upload and start background processing via SocketIO.
    """
    source = request.form.get('source')
    pdf_file = request.files.get('source_file')
    socket_id = request.form.get('socket_id')

    logger.info(f"Received upload request for source name: {source} from socket_id: {socket_id}")

    if not source:
        logger.warning("No source name provided.")
        return jsonify({"error": "Source name is required."}), 400
    if not pdf_file:
        logger.warning("No PDF file uploaded.")
        return jsonify({"error": "No PDF uploaded."}), 400
    if not socket_id:
        logger.warning("No Socket ID provided.")
        return jsonify({"error": "No socket ID provided for progress updates."}), 400
    if not pdf_file.filename.lower().endswith('.pdf'):
        logger.warning("Uploaded file is not a PDF.")
        return jsonify({"error": "Uploaded file is not a PDF."}), 400

    try:
        filename = secure_filename(pdf_file.filename)
        unique_id = str(uuid.uuid4())
        saved_filename = f"{unique_id}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        pdf_file.save(save_path)
        logger.info(f"Saved PDF file to {save_path}")

        # Start background task for processing
        socketio.start_background_task(
            process_pdf,
            source,
            pdf_file.filename,
            save_path,
            socket_id
        )
        logger.info(f"Started background task for processing PDF: {save_path}")

        return jsonify({"message": "PDF upload successful and processing started."}), 200
    except Exception as e:
        error_message = f"Error processing PDF: {str(e)}"
        logger.error(error_message)
        socketio.emit('processing_error', {'message': error_message}, room=socket_id)
        return jsonify({"error": error_message}), 500


def process_pdf(source: str, original_file_name: str, pdf_path: str, socket_id: str):
    """
    Background thread function to chunk, embed, and upload the PDF content to Qdrant.
    """
    try:
        socketio.emit('processing_progress', {'progress': 0}, room=socket_id)
        socketio.emit('status', {'message': 'Starting PDF processing...'}, room=socket_id)

        # Chunk PDF
        pdf_chunks = PDFProcessor.chunk_pdf_text(
            pdf_file_path=pdf_path,
            original_file_name=original_file_name,
            socketio_instance=socketio,
            socket_id=socket_id,
            max_words=400,
            overlap_sentences=1
        )

        if not pdf_chunks:
            socketio.emit('processing_error', {'message': 'No text could be extracted from PDF. Please ensure the PDF contains searchable text.'}, room=socket_id)
            return

        socketio.emit('status', {'message': 'Generating embeddings...'}, room=socket_id)

        # Process each chunk
        total_chunks = len(pdf_chunks)
        for idx, chunk in enumerate(pdf_chunks):
            # creating embeddings from the full chunk prooved to be the best option
            embedding = generate_embedding(chunk['text'])
            
            # Create knowledge node
            knowledge_id = db_client.create_knowledge_node(
                source_name=source,
                embedding=embedding,
                source_file=chunk['pdf_name'],
                page_num=chunk['page'],
                chunk_id=chunk['chunk_id'],
                text=chunk['text']
            )
            
            # Process concepts from the chunk
            try:
                memory_service.context_storing_service.process_concept_from_new_node(chunk['text'], knowledge_id, embedding)
            except Exception as concept_e:
                logger.error(f"Error processing concepts for knowledge node {knowledge_id}: {concept_e}. Concepts might be incomplete.")
                # Continue processing PDF chunks even if concept extraction fails.
            
            # Find similar knowledge nodes and create relationships
            similar_nodes = db_client.fetch_similar_nodes(
                query_embedding=embedding,
                fetch_limit=5,  # Limit to top 5 similar nodes
                min_similarity=0.7,  # Only link if similarity is high enough
                label_filter=["Knowledge"]
            )
            
            # Calculate and emit upload progress
            progress = (idx + 1) / total_chunks * 100
            socketio.emit('upload_progress', {'progress': progress}, room=socket_id)
            socketio.emit('status', {'message': f'Generating DB nodes from chunk {idx + 1} of {total_chunks}...'}, room=socket_id)
            
            # Create relationships to similar nodes
            for similar_node in similar_nodes:
                if similar_node["id"] != knowledge_id:  # Don't link to self
                    db_client.create_edge(
                        edge_type="RELATES_TO",
                        id1=knowledge_id,
                        id2=similar_node["id"],
                        label1="Knowledge",
                        label2="Knowledge"
                    )
            
            # Create sequential relationships with previous chunk
            if chunk['chunk_id'] > 0:  # If this is not the first chunk
                # Find the previous chunk in the same page
                with db_client.driver.session() as session:
                    result = session.run("""
                        MATCH (k:Knowledge)
                        WHERE k.source_file = $source_file 
                        AND k.source_name = $source_name
                        AND k.chunk_id = $prev_chunk_id
                        RETURN k.id as id
                    """, 
                    source_file=chunk['pdf_name'],
                    source_name=source,
                    page_num=chunk['page'],
                    prev_chunk_id=chunk['chunk_id'] - 1
                    )
                    prev_node = result.single()
                    if prev_node:
                        db_client.create_edge(
                            edge_type="NEXT_CHUNK",
                            id1=prev_node["id"],
                            id2=knowledge_id,
                            label1="Knowledge",
                            label2="Knowledge"
                        )

        socketio.emit('processing_progress', {'progress': 100}, room=socket_id)
        socketio.emit('processing_complete', {
            'message': 'PDF processed and uploaded successfully'
        }, room=socket_id)

    except Exception as e:
        error_message_to_user = "Failed to process PDF due to an internal server error."
        
        # Check for specific types of errors to provide more helpful messages
        error_str = str(e).lower()
        if "connection error" in error_str or "llm" in error_str or "openai" in error_str or "httpcore" in error_str:
            error_message_to_user = "Failed during PDF processing due to an issue communicating with the AI model. Please check LLM service configuration."
        elif "bolt" in error_str or "neo4j" in error_str or "database" in error_str or "serviceunavailable" in error_str:
            error_message_to_user = "Failed during PDF processing due to a database connection issue. Please ensure the Neo4j database is running and accessible."
        elif "no text could be extracted" in error_str: # This is a general fallback if somehow the earlier check was missed or a similar error occurs later
            error_message_to_user = "No text could be extracted from PDF. Please ensure the PDF contains searchable text."

        logger.exception(f"Error during PDF processing for source '{source}': {e}") # Log full traceback for debugging
        socketio.emit('processing_error', {'message': error_message_to_user}, room=socket_id)

    finally:
        # Delete the uploaded file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"Deleted file: {pdf_path}")
            else:
                logger.warning(f"File not found for deletion: {pdf_path}")
        except Exception as file_del_error:
            logger.error(f"Error deleting file '{pdf_path}': {file_del_error}")


# -----------------------------------------------------------------------------
#                      Socket.IO Event Handlers
# -----------------------------------------------------------------------------
@socketio.on('connect')
def handle_connect(socket_id):
    logger.info(f"Client connected: {socket_id}")


@socketio.on('disconnect')
def handle_disconnect(socket_id):
    logger.info(f"Client disconnected: {socket_id}")


# -----------------------------------------------------------------------------
#                                   Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    config = Config()
    socketio.run(app, debug=True, 
                host=config.get('web_server', 'host'),
                port=config.get('web_server', 'port'))
