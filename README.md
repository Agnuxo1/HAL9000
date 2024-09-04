# HAL9000
Chatbot audio a texto y texto a audio
Aquí puedes descargar el .exe para Windows: [https://huggingface.co/Agnuxo/HAL_9000-QWEN2-0.5_Spanish_English_16bit/resolve/main/HAL-9000](https://huggingface.co/Agnuxo/HAL_9000-QWEN2-0.5_Spanish_English_16bit/resolve/main/HAL-9000.exe)

![Screenshot at 2024-09-04 12-12-39](https://github.com/user-attachments/assets/7f6ab32a-342c-4445-b82c-41ac86489e73)


---
model_size: 1543717376
required_memory: 5.75
metrics:
- GLUE_MRPC
license: apache-2.0
datasets:
- Agnuxo/HAL9000
language:
- es
base_model: Qwen/Qwen2-1.5B-Instruct
library_name: adapter-transformers
tags:
- spanish
- spañol
- chat
- audio
- voz
---

# Uploaded model

[<img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" width="100"/><img src="https://github.githubassets.com/assets/GitHub-Logo-ee398b662d42.png" width="100"/>](https://github.com/Agnuxo1)
- **Developed by:** [Agnuxo](https://github.com/Agnuxo1)
- **License:** apache-2.0
- **Finetuned from model:** Agnuxo/Tinytron-Qwen2-0.5B

This model was fine-tuned using [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

## Benchmark Results

This model has been fine-tuned for various tasks and evaluated on the following benchmarks:

### GLUE_MRPC
**Accuracy:** 0.6446
**F1:** 0.7709

![GLUE_MRPC_metrics](https://github.com/user-attachments/assets/5784fe77-db4d-4250-8991-7141cb5e529d)



Model Size: 1,543,717,376 parameters
Required Memory: 5.75 GB

For more details, visit my [GitHub](https://github.com/Agnuxo1).

Thanks for your interest in this model!

```python
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import numpy as np
from TTS.api import TTS
import sounddevice as sd
import threading
import queue
import time
from vosk import Model, KaldiRecognizer
import json
import pyaudio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QFrame, QToolButton,
                             QLabel, QSlider, QComboBox, QCheckBox)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QAbstractAnimation, QParallelAnimationGroup

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global configuration
SYSTEM_PROMPT = {
    "es": "Tu nombre es HAL. Eres un superordenador de la serie Nueve mil",
    "en": "speak Spanish."
}

MODELO_LLM = "Agnuxo/HAL_9000-Qwen2-1.5B-Instruct_Asistant-16bit-v2" # Puede utilizar la versión Mini "Agnuxo/HAL_9000-Qwen2-0.5B-Instruct_Asistant-16bit-v2"
MAX_TOKENS = 100
TEMPERATURA = 0.5

# Determine available device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Qwen2_1.5B language model
tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODELO_LLM,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# Initialize TTS model
tts = TTS(model_name="tts_models/es/css10/vits", progress_bar=False).to(device)

# Audio queue for generation
audio_queue = queue.Queue()

# Initialize Vosk model for offline speech recognition
vosk_model = Model(lang="es")
recognizer = KaldiRecognizer(vosk_model, 16000)

class AudioThread(QThread):
    def run(self):
        while True:
            if not audio_queue.empty():
                wav = audio_queue.get()
                sd.play(wav, tts.synthesizer.output_sample_rate)
                sd.wait()
            else:
                time.sleep(0.1)

class SpeechRecognitionThread(QThread):
    text_recognized = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        while self.running:
            data = stream.read(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                texto = result.get("text", "")
                if texto:
                    self.text_recognized.emit(texto)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self.running = False

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                background-color: #1e1e1e;
                color: #bb86fc;
                border: 1px solid #bb86fc;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: #3700b3;
            }
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QScrollArea()
        self.content_area.setWidgetResizable(True)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)

        self.toggle_animation = QParallelAnimationGroup()
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self.content_area, b"maximumHeight"))

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

    def on_toggle(self, checked):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if not checked else Qt.RightArrow)
        self.toggle_animation.setDirection(QAbstractAnimation.Forward if not checked else QAbstractAnimation.Backward)
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Assistant")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QTextEdit, QLineEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #bb86fc;
            }
            QPushButton {
                background-color: #3700b3;
                color: #ffffff;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #6200ee;
            }
            QLabel {
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #1e1e1e;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #bb86fc;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #444444;
                border: 1px solid #bb86fc;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #444444;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Chat area
        chat_layout = QVBoxLayout()

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        chat_layout.addWidget(self.chat_area)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        self.mic_button = QPushButton()
        self.mic_button.setIcon(QIcon.fromTheme("audio-input-microphone"))
        self.mic_button.setCheckable(True)
        self.mic_button.clicked.connect(self.toggle_speech_recognition)
        input_layout.addWidget(self.mic_button)

        self.speaker_button = QPushButton()
        self.speaker_button.setIcon(QIcon.fromTheme("audio-volume-high"))
        self.speaker_button.setCheckable(True)
        self.speaker_button.toggled.connect(self.toggle_speech)
        input_layout.addWidget(self.speaker_button)

        chat_layout.addLayout(input_layout)

        main_layout.addLayout(chat_layout, 7)  # Chat area takes 70% of the width

        # Settings area
        settings_layout = QVBoxLayout()
        settings_layout.setAlignment(Qt.AlignTop)

        self.settings_box = CollapsibleBox("⚙️ Configuración")
        settings_content_layout = QVBoxLayout()

        # Language selection
        language_layout = QHBoxLayout()
        language_label = QLabel("Idioma:")
        language_label.setStyleSheet("color: #000000;")  # Change font color to black
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Español", "English"])
        self.language_combo.currentIndexChanged.connect(self.change_language)
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        settings_content_layout.addLayout(language_layout)

        # LLM settings
        llm_label = QLabel("Configuración del LLM:")
        llm_label.setStyleSheet("color: #000000;")  # Change font color to black
        settings_content_layout.addWidget(llm_label)

        max_tokens_layout = QHBoxLayout()
        max_tokens_label = QLabel("Max Tokens:")
        max_tokens_label.setStyleSheet("color: #000000;")  # Change font color to black
        self.max_tokens_slider = QSlider(Qt.Horizontal)
        self.max_tokens_slider.setRange(10, 500)
        self.max_tokens_slider.setValue(MAX_TOKENS)
        self.max_tokens_slider.valueChanged.connect(self.update_max_tokens)
        self.max_tokens_value = QLabel(str(MAX_TOKENS))
        max_tokens_layout.addWidget(max_tokens_label)
        max_tokens_layout.addWidget(self.max_tokens_slider)
        max_tokens_layout.addWidget(self.max_tokens_value)
        settings_content_layout.addLayout(max_tokens_layout)

        temperature_layout = QHBoxLayout()
        temperature_label = QLabel("Temperatura:")
        temperature_label.setStyleSheet("color: #000000;")  # Change font color to black
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(int(TEMPERATURA * 100))
        self.temperature_slider.valueChanged.connect(self.update_temperature)
        self.temperature_value = QLabel(f"{TEMPERATURA:.2f}")
        temperature_layout.addWidget(temperature_label)
        temperature_layout.addWidget(self.temperature_slider)
        temperature_layout.addWidget(self.temperature_value)
        settings_content_layout.addLayout(temperature_layout)

        # Audio settings
        audio_label = QLabel("Configuración de Audio:")
        audio_label.setStyleSheet("color: #000000;")  # Change font color to black
        settings_content_layout.addWidget(audio_label)

        sample_rate_layout = QHBoxLayout()
        sample_rate_label = QLabel("Sample Rate:")
        sample_rate_label.setStyleSheet("color: #000000;")  # Change font color to black
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000", "22050", "44100", "48000"])
        self.sample_rate_combo.setCurrentText("22050")
        self.sample_rate_combo.currentTextChanged.connect(self.update_sample_rate)
        sample_rate_layout.addWidget(sample_rate_label)
        sample_rate_layout.addWidget(self.sample_rate_combo)
        settings_content_layout.addLayout(sample_rate_layout)

        # System Prompt
        system_prompt_label = QLabel("System Prompt:")
        system_prompt_label.setStyleSheet("color: #000000;")  # Change font color to black
        settings_content_layout.addWidget(system_prompt_label)
        self.system_prompt_text = QTextEdit()
        self.system_prompt_text.setPlaceholderText("Escribe el prompt del sistema aquí...")
        self.system_prompt_text.setText(SYSTEM_PROMPT["es"])
        settings_content_layout.addWidget(self.system_prompt_text)

        self.settings_box.setContentLayout(settings_content_layout)
        settings_layout.addWidget(self.settings_box)

        main_layout.addLayout(settings_layout, 3)  # Settings area takes 30% of the width

        central_widget.setLayout(main_layout)

        self.audio_thread = AudioThread()
        self.audio_thread.start()

        self.speech_recognition_thread = SpeechRecognitionThread()
        self.speech_recognition_thread.text_recognized.connect(self.on_speech_recognized)

        self.speech_enabled = False
        self.is_listening = False

    def send_message(self):
        user_message = self.input_field.text()
        self.chat_area.append(f"<span style='color: #bb86fc;'>Usuario:</span> {user_message}")
        self.input_field.clear()

        response = self.generate_response(user_message)
        self.chat_area.append(f"<span style='color: #03dac6;'>Asistente:</span> {response}")

        if self.speech_enabled:
            self.speak(response)

    def generate_response(self, texto):
        system_instructions = self.system_prompt_text.toPlainText()
        prompt = f"{system_instructions}\nUsuario: {texto}\nAsistente: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=TEMPERATURA,
            )
        respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
        respuesta = respuesta_completa.split("Asistente: ")[-1].strip()
        return respuesta

    def speak(self, text):
        wav = tts.tts(text)
        audio_queue.put(wav)

    def toggle_speech(self, checked):
        self.speech_enabled = checked
        if checked:
            self.speaker_button.setStyleSheet("background-color: #bb86fc;")
        else:
            self.speaker_button.setStyleSheet("")

    def toggle_speech_recognition(self):
        if self.mic_button.isChecked():
            self.speech_recognition_thread.start()
            self.is_listening = True
            self.mic_button.setIcon(QIcon.fromTheme("audio-input-microphone-muted"))
            self.mic_button.setStyleSheet("background-color: #bb86fc;")
        else:
            self.speech_recognition_thread.stop()
            self.is_listening = False
            self.mic_button.setIcon(QIcon.fromTheme("audio-input-microphone"))
            self.mic_button.setStyleSheet("")


    def on_speech_recognized(self, text):
        self.chat_area.append(f"<span style='color: #bb86fc;'>Usuario:</span> {text}")
        response = self.generate_response(text)
        self.chat_area.append(f"<span style='color: #03dac6;'>Asistente:</span> {response}")
        if self.speech_enabled:
            self.speak(response)

    def change_language(self, index):
        global vosk_model, recognizer, tts
        lang = "es" if index == 0 else "en"
        try:
            vosk_model = Model(lang=lang)
            recognizer = KaldiRecognizer(vosk_model, 16000)
        except Exception as e:
            print(f"Error al cambiar el modelo de reconocimiento de voz: {e}")
            # Revertir al modelo en español si hay un error
            self.language_combo.setCurrentIndex(0)
            return

        # Update TTS model based on language
        tts_model = "tts_models/es/css10/vits" if lang == "es" else "tts_models/en/ljspeech/tacotron2-DDC"
        try:
            tts = TTS(model_name=tts_model, progress_bar=False).to(device)
        except Exception as e:
            print(f"Error al cambiar el modelo TTS: {e}")
            # Revertir al modelo en español si hay un error
            self.language_combo.setCurrentIndex(0)
            return

        # Update system prompt
        self.system_prompt_text.setText(SYSTEM_PROMPT[lang])

    def update_max_tokens(self, value):
        global MAX_TOKENS
        MAX_TOKENS = value
        self.max_tokens_value.setText(str(value))

    def update_temperature(self, value):
        global TEMPERATURA
        TEMPERATURA = value / 100
        self.temperature_value.setText(f"{TEMPERATURA:.2f}")

    def update_sample_rate(self, value):
        global tts
        tts.synthesizer.output_sample_rate = int(value)

    def closeEvent(self, event):
        if self.speech_recognition_thread.isRunning():
            self.speech_recognition_thread.stop()
            self.speech_recognition_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
