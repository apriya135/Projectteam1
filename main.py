
# diabetes_app_tflite.py
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from PIL import Image as PILImage
import numpy as np
import time
import io
import os
from fpdf import FPDF
import pickle
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
from kivy.utils import platform
import streamlit as st

st.set_page_config(layout='wide')  # Place this at the top

# Your other Streamlit code follows
st.title("Diabetes Screening App")



# Android permissions
try:
    from android.permissions import request_permissions, Permission
    request_permissions([
        Permission.CAMERA,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE
    ])
except:
    pass

Window.size = (360, 640)

# Load TFLite models and scaler
interpreter = tf.lite.Interpreter(model_path="diabetes_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mobilenet_interpreter = tf.lite.Interpreter(model_path="mobilenetv2_features.tflite")
mobilenet_interpreter.allocate_tensors()
mobilenet_input_details = mobilenet_interpreter.get_input_details()
mobilenet_output_details = mobilenet_interpreter.get_output_details()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class FileChooserPopup(Popup):
    def __init__(self, file_type, on_selection, **kwargs):
        super().__init__(**kwargs)
        self.title = f"Select {file_type}"
        self.size_hint = (0.9, 0.9)
        layout = BoxLayout(orientation='vertical')
        self.filechooser = FileChooserIconView()
        if file_type == "Image":
            self.filechooser.filters = ["*.jpg", "*.jpeg", "*.png"]
        elif file_type == "Video":
            self.filechooser.filters = ["*.mp4", "*.avi"]
        layout.add_widget(self.filechooser)
        select_button = Button(text='Select', size_hint_y=None, height=50)
        select_button.bind(on_press=lambda instance: self.select_file(on_selection))
        layout.add_widget(select_button)
        self.add_widget(layout)

    def select_file(self, on_selection):
        selected = self.filechooser.selection
        if selected:
            on_selection(selected[0])
            self.dismiss()

class CameraPopup(Popup):
    def __init__(self, on_capture, **kwargs):
        super().__init__(**kwargs)
        self.title = "Capture Tongue Image"
        self.size_hint = (0.9, 0.9)
        layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True)
        self.camera.resolution = (640, 480)
        layout.add_widget(self.camera)
        capture_btn = Button(text="Capture", size_hint_y=None, height=50)
        capture_btn.bind(on_press=lambda x: self.capture(on_capture))
        layout.add_widget(capture_btn)
        self.add_widget(layout)

    def capture(self, on_capture):
        texture = self.camera.texture
        pixels = texture.pixels
        size = texture.size
        pil_image = PILImage.frombytes(mode='RGBA', size=size, data=pixels)
        path = os.path.join(App.get_running_app().user_data_dir, f"captured_tongue_{int(time.time())}.png")
        pil_image.convert("RGB").save(path)
        on_capture(path)
        self.dismiss()

class InstructionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        instruction_text = """Welcome to the Diabetes Detection App!\n\nPlease follow these steps:\n1. Upload or capture a clear tongue image.\n2. Upload or capture a short PPG video (30 seconds).\n3. Press Predict to see your result.\n4. Download a report."""
        layout.add_widget(Label(text=instruction_text, halign='left', valign='top'))
        continue_btn = Button(text="Continue", size_hint=(1, None), height=50)
        continue_btn.bind(on_press=self.go_to_main)
        layout.add_widget(continue_btn)
        self.add_widget(layout)

    def go_to_main(self, instance):
        self.manager.current = 'main'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name_input = TextInput(hint_text="Enter Name", multiline=False, size_hint_y=None, height=40)
        self.age_input = TextInput(hint_text="Enter Age", multiline=False, input_filter='int', size_hint_y=None, height=40)
        self.tongue_path = None
        self.video_path = None
        self.prediction_result = ""

        scroll = ScrollView(size_hint=(1, 1))
        layout = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=10)
        layout.bind(minimum_height=layout.setter('height'))
        layout.add_widget(self.name_input)
        layout.add_widget(self.age_input)

        btns = [
            ("Upload Tongue Image", self.open_image_chooser),
            ("Capture Tongue Image", self.capture_image),
            ("Upload PPG Video", self.open_video_chooser),
            ("Capture PPG Video", self.simulate_video_capture),
            ("Predict", self.predict),
            ("Generate Report", self.generate_report),
            ("Download PDF Report", self.download_pdf)
        ]
        for text, func in btns:
            btn = Button(text=text, size_hint_y=None, height=50)
            btn.bind(on_press=func)
            layout.add_widget(btn)

        self.result_label = Label(text="", size_hint_y=None, height=40)
        self.report_label = Label(text="", size_hint_y=None, height=100)
        layout.add_widget(self.result_label)
        layout.add_widget(self.report_label)

        scroll.add_widget(layout)
        self.add_widget(scroll)

    def open_image_chooser(self, instance):
        FileChooserPopup("Image", self.set_image_path).open()

    def open_video_chooser(self, instance):
        FileChooserPopup("Video", self.set_video_path).open()

    def capture_image(self, instance):
        CameraPopup(self.set_image_path).open()

    def simulate_video_capture(self, instance):
        path = os.path.join(App.get_running_app().user_data_dir, f"simulated_ppg_{int(time.time())}.mp4")
        with open(path, 'w') as f:
            f.write("Simulated PPG video content")
        self.set_video_path(path)

    def set_image_path(self, path):
        self.tongue_path = path
        self.result_label.text = f"Tongue Image: {os.path.basename(path)}"

    def set_video_path(self, path):
        self.video_path = path
        self.result_label.text = f"PPG Video: {os.path.basename(path)}"

    def predict(self, instance):
        try:
            if not self.tongue_path or not self.video_path:
                self.prediction_result = "Please upload or capture both image and video."
            else:
                img = image.load_img(self.tongue_path, target_size=(224, 224))
                img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

                mobilenet_interpreter.set_tensor(mobilenet_input_details[0]['index'], img_array.astype(np.float32))
                mobilenet_interpreter.invoke()
                tongue_features = mobilenet_interpreter.get_tensor(mobilenet_output_details[0]['index'])

                cap = cv2.VideoCapture(self.video_path)
                green_means = [np.mean(frame[:, :, 1]) for ret, frame in iter(lambda: cap.read(), (False, None)) if ret]
                cap.release()

                stats = np.array([
                    np.mean(green_means), np.std(green_means),
                    np.max(green_means), np.min(green_means),
                    np.median(green_means),
                    np.max(green_means) - np.min(green_means),
                    (3 * (np.mean(green_means) - np.median(green_means))) / (np.std(green_means) or 1)
                ]).reshape(1, -1)

                combined = np.concatenate([tongue_features, stats], axis=1)
                scaled = scaler.transform(combined)

                interpreter.set_tensor(input_details[0]['index'], scaled.astype(np.float32))
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

                self.prediction_result = "Diabetic" if pred >= 0.5 else "Non-Diabetic"
        except Exception as e:
            self.prediction_result = f"Error: {str(e)}"

        self.result_label.text = f"Prediction: {self.prediction_result}"

    def generate_report(self, instance):
        advice = "Consult a doctor." if self.prediction_result == "Diabetic" else "Maintain a healthy lifestyle."
        self.report_label.text = f"Name: {self.name_input.text}\nAge: {self.age_input.text}\nResult: {self.prediction_result}\nAdvice: {advice}"

    def download_pdf(self, instance):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        name = self.name_input.text or "Anonymous"
        pdf.cell(200, 10, txt="Diabetes Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {self.age_input.text}", ln=True)
        pdf.cell(200, 10, txt=f"Result: {self.prediction_result}", ln=True)
        advice = "Consult a doctor." if self.prediction_result == "Diabetic" else "Maintain a healthy lifestyle."
        pdf.multi_cell(0, 10, txt=f"Advice: {advice}")
        filepath = os.path.join(App.get_running_app().user_data_dir, f"{name}_report.pdf")
        pdf.output(filepath)
        self.result_label.text = f"PDF saved as {filepath}"

class DiabetesApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(InstructionScreen(name='instruction'))
        sm.add_widget(MainScreen(name='main'))
        return sm

    def on_start(self):
        if platform == 'android':
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.MANAGE_EXTERNAL_STORAGE])
            except Exception as e:
                print(f"Permission request failed: {e}")

if __name__ == '__main__':
    DiabetesApp().run()
