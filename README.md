# Assistive Sign Language Communication System Using Vision and Language Models
- This project is a real-time assistive communication system developed using Python, Flask, OpenCV, YOLOv11, and Language Models. The main objective of the project is to reduce the communication gap between hearing-impaired individuals and people who do not understand sign language.
- The system captures hand gestures through a webcam and detects them in real time using the YOLOv11 object detection model. The detected gestures are converted into words and then processed using a Large Language Model (LLM) to generate meaningful and grammatically correct sentences. The generated text is also converted into speech using Text-to-Speech libraries, making communication easier and more natural.
- This project combines computer vision, deep learning, natural language processing, and web technologies into a single Flask-based web application with a simple and user-friendly interface.
## Features
- Real-time hand gesture recognition using webcam
- Gesture-to-word conversion
- Sentence generation using AI and NLP
- Text-to-Speech output
- Live webcam integration
- Flask-based web application
- Real-time visual and audio feedback
- User-friendly interface for communication and learning
## Technologies Used
Python, Flask, OpenCV, YOLOv11, PyTorch, HTML, CSS, JavaScript, gTTS/pyttsx3 and Large Language Models (LLM) for real-time gesture recognition, sentence generation, and speech output.
## How the System Works
1. The webcam captures live hand gestures.
2. OpenCV processes the video frames.
3. YOLOv11 detects and classifies gestures in real time.
4. Detected gestures are mapped into words.
5. The Language Model generates meaningful sentences.
6. Text-to-Speech converts the generated text into audio.
7. The final output is displayed on the web interface.
## Project Structure
Assistive-Sign-Language-System/
│
├── static/
│   └── CSS and frontend assets
│
├── templates/
│   └── HTML pages
│
├── app.py
│   └── Main Flask application
│
├── best.pt
│   └── Trained YOLOv11 model
│
├── requirements.txt
│   └── Required libraries
│
└── README.md

## Installation
Install all required dependencies:
pip install -r requirements.txt
## Run the Project
Run the Flask application: python app.py
Open your browser and go to: http://127.0.0.1:5000
Allow camera access and start gesture detection.
## Advantages of the System
- Helps improve communication for hearing-impaired individuals
- Provides real-time gesture recognition
- Generates meaningful sentences instead of word-by-word translation
- Supports both text and speech output
- Easy to use and accessible through a web browser
## Future Enhancements
- Multi-language support
- Mobile application integration
- Improved gesture vocabulary
- Facial expression recognition
- Better sentence generation using advanced AI models
## About the Project
This project was developed as a final year B.Tech project to explore the use of Artificial Intelligence, Computer Vision, and Natural Language Processing in real-time communication systems. The project provided practical experience in Flask development, YOLO-based object detection, OpenCV integration, and AI-powered sentence generation.
