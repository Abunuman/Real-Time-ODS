from flask import Flask, send_from_directory, Response, jsonify, request
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

load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class AIAssistant:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        self.chain = self._create_inference_chain()
        self.chat_history = ChatMessageHistory()

    def _create_inference_chain(self):
        SYSTEM_PROMPT = """
        You're a clever assistant that analyzes live video streams and responds to user questions.
        Keep your answers concise and direct. Focus on what you can see in the current frame.
        Be engaging but professional.
        """

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", [
                {"type": "text", "text": "{prompt}"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},
            ]),
        ])

        chain = prompt_template | self.model | StrOutputParser()
        return RunnableWithMessageHistory(
            chain,
            lambda _: self.chat_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

    def process_query(self, prompt, image_base64):
        try:
            response = self.chain.invoke(
                {"prompt": prompt, "image_base64": image_base64},
                config={"configurable": {"session_id": "unused"}},
            ).strip()
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Sorry, I encountered an error processing your request."

# Global instances
camera = None
assistant = None

def initialize_system():
    global camera, assistant
    try:
        camera = WebcamCapture().start()
        if camera is None:
            logger.error("Camera initialization failed")
            raise RuntimeError("Camera initialization failed")
            
        assistant = AIAssistant()
        logger.info("System initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise
    
def create_app():
    app = Flask(__name__)

    # Ensure the template and static directories exist
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    
    # Initialize the system and attach global instances
    with app.app_context():
        initialize_system()  # Use the new initialize_system function
        app.camera = camera
        app.assistant = assistant
    
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
            if not prompt:
                return jsonify({"error": "No prompt provided"}), 400

            # Handle case where frame comes from client
            frame_data = data.get('frame')  # Get frame from client
            if frame_data:
                try:
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
                if app.camera and app.camera.frame is not None:
                    frame = app.camera.read()
                    if frame is None:
                        return jsonify({"error": "Failed to capture frame"}), 500
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                else:
                    return jsonify({"error": "No frame available"}), 400

            # Process with AI assistant
            response = app.assistant.process_query(prompt, image_base64)
            return jsonify({"response": response})

        except Exception as e:
            logger.error(f"Error in process_query: {e}")
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
    
    while True:
        if camera is None:
            logger.error("Camera is None in generate_frames")
            break
        try:
            frame = camera.get_jpeg()
            frame_count += 1
            
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
    app.run(host='0.0.0.0', port=5000, debug=debug_mode, threaded=True)
