import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import sys
import os

# Import your existing detection functions
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
import warnings
import torch.nn.functional as F
from keras.models import Sequential
from keras.layers import Input, TimeDistributed, Dropout, Flatten, LSTM, Bidirectional, Dense
from keras.applications.mobilenet_v2 import MobileNetV2

warnings.filterwarnings("ignore")

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire & Violence Detection System")
        self.root.geometry("1000x700")
        
        # Variables
        self.video_running = False
        self.cap = None
        self.fire_enabled = tk.BooleanVar(value=True)
        self.violence_enabled = tk.BooleanVar(value=True)
        self.current_thread = None
        
        # Model variables
        self.fire_model = None
        self.fire_feature_extractor = None
        self.fire_device = None
        self.violence_model = None
        self.models_loaded = False
        
        # Violence detection constants
        self.SEQUENCE_LENGTH = 16
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = 64, 64
        self.CLASSES_LIST = ["NonViolence", "Violence"]
        self.MODEL_PATH = 'violence_detection_model.h5'
        self.frame_buffer = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Fire & Violence Detection System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Detection toggles
        ttk.Label(control_frame, text="Detection Options:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        fire_check = ttk.Checkbutton(control_frame, text="Fire Detection", variable=self.fire_enabled)
        fire_check.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        violence_check = ttk.Checkbutton(control_frame, text="Violence Detection", variable=self.violence_enabled)
        violence_check.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Input source
        ttk.Label(control_frame, text="Input Source:", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        self.webcam_btn = ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam)
        self.webcam_btn.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.video_btn = ttk.Button(control_frame, text="Load Video File", command=self.load_video)
        self.video_btn.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Model loading
        ttk.Label(control_frame, text="Model Management:", font=('Arial', 10, 'bold')).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        self.load_models_btn = ttk.Button(control_frame, text="Load Models", command=self.load_models)
        self.load_models_btn.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=9, column=0, sticky=tk.W, pady=(20, 0))
        
        # Video display frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video canvas
        self.canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Results text widget
        self.results_text = tk.Text(results_frame, height=10, width=30, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
        
    def log_result(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        
    def load_models(self):
        """Load the AI models"""
        try:
            self.update_status("Loading models...", "orange")
            
            # Load fire detection model
            if self.fire_enabled.get():
                self.fire_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.fire_model = ViTForImageClassification.from_pretrained('EdBianchi/vit-fire-detection')
                self.fire_model.to(self.fire_device)
                self.fire_feature_extractor = ViTFeatureExtractor.from_pretrained('EdBianchi/vit-fire-detection')
                self.fire_model.eval()
                self.log_result("Fire detection model loaded successfully")
            
            # Load violence detection model
            if self.violence_enabled.get():
                if not os.path.exists(self.MODEL_PATH):
                    messagebox.showerror("Error", f"Violence detection model not found at {self.MODEL_PATH}")
                    self.update_status("Model loading failed", "red")
                    return
                
                self.violence_model = self.build_violence_model()
                self.violence_model.load_weights(self.MODEL_PATH)
                self.log_result("Violence detection model loaded successfully")
            
            self.models_loaded = True
            self.update_status("Models loaded successfully", "green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.update_status("Model loading failed", "red")
            
    def build_violence_model(self):
        """Build the violence detection model architecture"""
        mobilenet = MobileNetV2(include_top=False, weights="imagenet")
        mobilenet.trainable = True
        for layer in mobilenet.layers[:-40]:
            layer.trainable = False
        
        model = Sequential()
        model.add(Input(shape=(self.SEQUENCE_LENGTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)))
        model.add(TimeDistributed(mobilenet))
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Flatten()))
        lstm_fw = LSTM(units=32)
        lstm_bw = LSTM(units=32, go_backwards=True)
        model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(len(self.CLASSES_LIST), activation='softmax'))
        
        return model
    
    def predict_fire(self, frame):
        """Predict fire in frame"""
        if not self.fire_enabled.get() or self.fire_model is None:
            return None, None, None
            
        try:
            image = Image.fromarray(frame)
            inputs = self.fire_feature_extractor(images=image, return_tensors="pt").to(self.fire_device)
            with torch.no_grad():
                outputs = self.fire_model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().squeeze().numpy()
                id2label = self.fire_model.config.id2label
                label = "fire" if probs[0] > 0.16 else "normal"
            return label, probs, id2label
        except Exception as e:
            self.log_result(f"Fire prediction error: {str(e)}")
            return None, None, None
    
    def predict_violence(self, frame):
        """Predict violence in frame sequence"""
        if not self.violence_enabled.get() or self.violence_model is None:
            return None, None
            
        try:
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            self.frame_buffer.append(normalized_frame)
            
            if len(self.frame_buffer) > self.SEQUENCE_LENGTH:
                self.frame_buffer.pop(0)
            
            if len(self.frame_buffer) == self.SEQUENCE_LENGTH:
                input_array = np.expand_dims(self.frame_buffer, axis=0)
                predictions = self.violence_model.predict(input_array, verbose=0)[0]
                predicted_label = np.argmax(predictions)
                predicted_class = self.CLASSES_LIST[predicted_label]
                return predicted_class, predictions
            return None, None
        except Exception as e:
            self.log_result(f"Violence prediction error: {str(e)}")
            return None, None
    
    def start_webcam(self):
        """Start webcam detection"""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please load models first")
            return
            
        if not self.fire_enabled.get() and not self.violence_enabled.get():
            messagebox.showwarning("Warning", "Please enable at least one detection type")
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
            
        self.video_running = True
        self.frame_buffer = []
        self.webcam_btn.config(state=tk.DISABLED)
        self.video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.current_thread = threading.Thread(target=self.process_webcam)
        self.current_thread.daemon = True
        self.current_thread.start()
        
        self.update_status("Webcam running", "green")
        
    def load_video(self):
        """Load and process video file"""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please load models first")
            return
            
        if not self.fire_enabled.get() and not self.violence_enabled.get():
            messagebox.showwarning("Warning", "Please enable at least one detection type")
            return
            
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not video_path:
            return
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file")
            return
            
        self.video_running = True
        self.frame_buffer = []
        self.webcam_btn.config(state=tk.DISABLED)
        self.video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.current_thread = threading.Thread(target=self.process_video)
        self.current_thread.daemon = True
        self.current_thread.start()
        
        self.update_status("Video processing", "green")
        
    def process_webcam(self):
        """Process webcam frames"""
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.process_frame(frame)
            
    def process_video(self):
        """Process video frames"""
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.process_frame(frame)
            
            # Add small delay for video playback
            cv2.waitKey(30)
            
    def process_frame(self, frame):
        """Process a single frame"""
        try:
            # Make predictions
            fire_label, fire_probs, fire_id2label = self.predict_fire(frame)
            violence_label, violence_probs = self.predict_violence(frame)
            
            # Draw overlays
            display_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            
            # Fire detection overlay
            if fire_label is not None:
                color = (0, 0, 255) if fire_label == "fire" else (0, 255, 0)
                cv2.putText(display_frame, f"Fire: {fire_label}", (10, y_offset), font, 0.7, color, 2)
                y_offset += 30
                
                if fire_id2label is not None:
                    for idx, label_name in fire_id2label.items():
                        prob = fire_probs[int(idx)]
                        text = f"{label_name}: {prob:.3f}"
                        cv2.putText(display_frame, text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
                        y_offset += 20
                
                # Log fire detection
                if fire_label == "fire":
                    self.log_result(f"🔥 FIRE DETECTED! Confidence: {fire_probs[0]:.3f}")
            
            # Violence detection overlay
            if violence_label is not None and violence_probs is not None:
                y_offset += 10
                color = (0, 0, 255) if violence_label == "Violence" else (0, 255, 0)
                cv2.putText(display_frame, f"Violence: {violence_label}", (10, y_offset), font, 0.7, color, 2)
                y_offset += 30
                
                for idx, class_name in enumerate(self.CLASSES_LIST):
                    prob = violence_probs[idx]
                    text = f"{class_name}: {prob:.2f}"
                    cv2.putText(display_frame, text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
                    y_offset += 20
                
                # Log violence detection
                if violence_label == "Violence":
                    self.log_result(f"⚠️ VIOLENCE DETECTED! Confidence: {violence_probs[1]:.3f}")
            
            # Update display
            self.update_display(display_frame)
            
        except Exception as e:
            self.log_result(f"Frame processing error: {str(e)}")
            
    def update_display(self, frame):
        """Update the video display"""
        try:
            # Resize frame to fit canvas
            height, width = frame.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / width, canvas_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
            self.canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Display update error: {str(e)}")
            
    def stop_detection(self):
        """Stop video processing"""
        self.video_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.webcam_btn.config(state=tk.NORMAL)
        self.video_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.canvas.delete("all")
        self.update_status("Stopped", "red")
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = DetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()