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
        self.stream = cv2.VideoCapture(0)
        if not self.stream.isOpened():
            raise RuntimeError("Could not initialize webcam")
        ret, self.frame = self.stream.read()
        if not ret:
            raise RuntimeError("Failed to capture initial frame")
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        
        self.running = True
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_jpeg(self):
        frame = self.read()
        if frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return jpeg.tobytes()

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
        # Try initializing the webcam
        camera = WebcamCapture().start()
    except RuntimeError:
        # Fallback to a placeholder image or default behavior
        camera = None
        logger.warning("Webcam initialization failed. Fallback to default behavior.")

    try:
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

            # Get current frame and encode to base64
            frame = app.camera.read()
            if frame is None:
                return jsonify({"error": "Failed to capture frame"}), 500
                
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Process with AI assistant
            response = app.assistant.process_query(prompt, image_base64)
            return jsonify({"response": response})

        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return jsonify({"error": str(e)}), 500

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
    while True:
        if camera is None:
            break
        
        frame = camera.get_jpeg()
        if frame is None:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    load_dotenv()
    app = create_app()
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode, threaded=True)
