from flask import Flask, send_from_directory, Response, jsonify, request
from flask_cors import CORS
import cv2
import base64
from threading import Lock, Thread
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import numpy as np
from datetime import datetime
import time
from functools import wraps
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure rate limiting
RATE_LIMIT_SECONDS = 10.0  # Minimum time between API calls
last_api_call = 0
api_lock = Lock()

def rate_limit_api_call():
    """Rate limit decorator for API calls with exponential backoff"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            global last_api_call
            with api_lock:
                current_time = time.time()
                time_since_last_call = current_time - last_api_call
                if time_since_last_call < RATE_LIMIT_SECONDS:
                    sleep_time = RATE_LIMIT_SECONDS - time_since_last_call
                    time.sleep(sleep_time + random.uniform(0, 2))  # Add more random jitter
                last_api_call = time.time()
                return f(*args, **kwargs)
        return wrapper
    return decorator

class ModelProvider:
    GOOGLE = "google"

class AIAssistant:
    def __init__(self):
        self.model = self._initialize_model()
        self.chain = self._create_inference_chain()
        self.chat_history = ChatMessageHistory()
        self.processing_lock = Lock()
        self.last_api_call = 0
        self.last_response = None
        self.api_cooldown = 10  # Minimum seconds between API calls
        
    def _initialize_model(self):
        """Initialize the Gemini model"""
        google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not google_api_key:
            raise ValueError("No API key found for Google. Please set GOOGLE_API_KEY in your .env file")
            
        logger.info("Initializing Google model")
        self.model_provider = ModelProvider.GOOGLE
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            max_output_tokens=300,
            top_p=0.8,
            top_k=40,
            convert_system_message_to_human=True
        )

    def _create_inference_chain(self):
        SYSTEM_PROMPT = """You're a clever assistant that analyzes live video streams and responds to user questions.
Keep your answers concise and direct. Focus on what you can see in the current frame.
Be engaging but professional."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        chain = prompt | self.model | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    @rate_limit_api_call()
    def process_query(self, prompt, image_base64):
        try:
            with self.processing_lock:
                current_time = time.time()
                # If we have a cached response and we're within the cooldown period, return the cached response
                if self.last_response and (current_time - self.last_api_call) < self.api_cooldown:
                    logger.info("Returning cached response due to cooldown")
                    return self.last_response
                
                response = self.chain.invoke(
                    {"input": prompt, "history": self.chat_history},
                    config={"configurable": {"session_id": "unused"}},
                ).strip()
                
                # Update cache
                self.last_response = response
                self.last_api_call = current_time
                return response
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limit exceeded for {self.model_provider} API, please wait a moment before trying again")
                # If we have a cached response, return it instead
                if self.last_response:
                    logger.info("Returning cached response due to rate limit")
                    return self.last_response
                return "I'm processing too many requests right now. Please wait a moment and try again."
            logger.error(f"Error processing query with {self.model_provider} model: {e}")
            return f"Sorry, I encountered an error processing your request with {self.model_provider} model. Please try again."

class WebcamCapture:
    def __init__(self):
        try:
            self.stream = cv2.VideoCapture(0)
            if not self.stream.isOpened():
                logger.warning("No webcam available - initializing in fallback mode")
                self.stream = None
                self.frame = None
            else:
                ret, self.frame = self.stream.read()
                if not ret:
                    logger.warning("Failed to capture initial frame - initializing in fallback mode")
                    self.stream = None
                    self.frame = None
        except Exception as e:
            logger.warning(f"Error initializing webcam: {e} - initializing in fallback mode")
            self.stream = None
            self.frame = None
        
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.stream is None:
            logger.info("Running in fallback mode without webcam")
            return self
            
        if self.running:
            return self
        
        self.running = True
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            try:
                ret, frame = self.stream.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                else:
                    logger.warning("Failed to capture frame in _update")
            except Exception as e:
                logger.error(f"Error in _update: {e}")

    def read(self):
        try:
            with self.lock:
                if self.frame is None:
                    logger.warning("Frame is None in read()")
                    return None
                frame_copy = self.frame.copy()
                if frame_copy is None:
                    logger.warning("Frame copy is None")
                    return None
                return frame_copy
        except Exception as e:
            logger.error(f"Error in read(): {e}")
            return None

    def get_jpeg(self):
        try:
            frame = self.read()
            if frame is None:
                logger.warning("No frame available for JPEG conversion")
                return None
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("Failed to encode frame as JPEG")
                return None
            return jpeg.tobytes()
        except Exception as e:
            logger.error(f"Error in get_jpeg: {e}")
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.stream.release()

def initialize_system():
    try:
        camera = WebcamCapture().start()
        if camera is None:
            logger.error("Camera initialization failed")
            raise RuntimeError("Camera initialization failed")
            
        assistant = AIAssistant()
        logger.info("System initialized successfully")
        return camera, assistant
            
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Initialize system components
    app.camera, app.assistant = initialize_system()

    # Ensure the template and static directories exist
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    @app.route('/')
    def index():
        try:
            return send_from_directory('templates', 'index.html')
        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return "Error: Could not load index.html. Please ensure the file exists in the templates directory.", 500

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(app.camera),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/process_query', methods=['POST'])
    def process_query():
        try:
            data = request.json
            prompt = data.get('prompt')
            logger.info(f"Received query prompt: {prompt}")
            
            if not prompt:
                return jsonify({"error": "No prompt provided"}), 400

            # Handle case where frame comes from client
            frame_data = data.get('frame')  # Get frame from client
            if frame_data:
                try:
                    logger.info("Processing frame from client")
                    # Remove data URL prefix if present
                    if ',' in frame_data:
                        frame_data = frame_data.split(',')[1]
                    # Use frame data directly for AI processing
                    image_base64 = frame_data
                except Exception as e:
                    logger.error(f"Error processing client frame: {e}")
                    return jsonify({"error": "Failed to process frame"}), 500
            else:
                # Fallback to server camera if available
                logger.info("Using server camera for frame")
                if app.camera and app.camera.frame is not None:
                    frame = app.camera.read()
                    if frame is None:
                        logger.error("Failed to read frame from camera")
                        return jsonify({"error": "Failed to capture frame"}), 500
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    logger.info("Successfully captured and encoded frame")
                else:
                    logger.error("No camera or frame available")
                    return jsonify({"error": "No frame available"}), 400

            # Process with AI assistant
            logger.info("Sending frame to AI assistant for processing")
            response = app.assistant.process_query(prompt, image_base64)
            
            if "429" in str(response):
                return jsonify({
                    "response": response,
                    "error": "rate_limit",
                    "retry_after": RATE_LIMIT_SECONDS
                })
            
            return jsonify({"response": response})

        except Exception as e:
            logger.error(f"Error in process_query: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @app.route('/camera_status')
    def camera_status():
        if app.camera is None:
            return jsonify({"status": "Camera not initialized"})
            
        status = {
            "initialized": True,
            "is_running": app.camera.running,
            "stream_opened": app.camera.stream.isOpened() if app.camera.stream else False,
            "has_frame": app.camera.frame is not None
        }
        return jsonify(status)

    @app.route('/process_frame', methods=['POST'])
    def process_frame():
        try:
            data = request.get_json()
            if not data or 'frame' not in data:
                return jsonify({'error': 'No frame data received'}), 400
                
            # Get the frame data and decode it from base64
            frame_data = data['frame'].split(',')[1]  # Remove the data URL prefix
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
                
            # Process the frame using your existing AI assistant
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
                
            # You can use your existing AI assistant to process the frame
            response = app.assistant.process_query("What do you see in this frame?", image_base64)
                
            return jsonify({'success': True, 'analysis': response})
                
        except Exception as e:
            app.logger.error(f"Error processing frame: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
    return app


def generate_frames(camera):
    frame_count = 0
    error_count = 0
    last_frame_time = 0
    frame_interval = 1.0  # Only process a frame every 1 second
    
    while True:
        if camera is None:
            logger.error("Camera is None in generate_frames")
            break
        try:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.1)  # Short sleep to prevent busy waiting
                continue
                
            frame = camera.get_jpeg()
            frame_count += 1
            last_frame_time = current_time
            
            if frame is None:
                error_count += 1
                logger.warning(f"No frame available (error {error_count} of {frame_count} frames)")
                if error_count > 10:  # After 10 consecutive errors
                    logger.error("Too many frame capture errors")
                    break
                continue
            
            error_count = 0  # Reset error count on successful frame
                
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"Successfully streaming: {frame_count} frames captured")
                
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            error_count += 1
            if error_count > 10:
                break

if __name__ == '__main__':
    load_dotenv()
    app = create_app()
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5001, debug=debug_mode)
