import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import sys
import os

# Import fire detection components
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

class FireDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire Detection System")
        self.root.geometry("1000x700")
        
        # Variables
        self.video_running = False
        self.cap = None
        self.fire_enabled = tk.BooleanVar(value=True)
        self.current_thread = None
        
        # Model variables
        self.fire_model = None
        self.fire_feature_extractor = None
        self.fire_device = None
        self.models_loaded = False
        
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
        title_label = ttk.Label(main_frame, text="Fire Detection System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Detection toggle
        ttk.Label(control_frame, text="Detection Options:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        fire_check = ttk.Checkbutton(control_frame, text="Fire Detection", variable=self.fire_enabled)
        fire_check.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Input source
        ttk.Label(control_frame, text="Input Source:", font=('Arial', 10, 'bold')).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        self.webcam_btn = ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam)
        self.webcam_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.video_btn = ttk.Button(control_frame, text="Load Video File", command=self.load_video)
        self.video_btn.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Model loading
        ttk.Label(control_frame, text="Model Management:", font=('Arial', 10, 'bold')).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        self.load_models_btn = ttk.Button(control_frame, text="Load Fire Model", command=self.load_models)
        self.load_models_btn.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Threshold adjustment
        ttk.Label(control_frame, text="Fire Threshold:", font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        
        self.threshold_var = tk.DoubleVar(value=0.16)
        threshold_scale = ttk.Scale(control_frame, from_=0.01, to=0.99, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.threshold_label = ttk.Label(control_frame, text="0.16")
        self.threshold_label.grid(row=10, column=0, sticky=tk.W, pady=2)
        
        # Update threshold label
        threshold_scale.configure(command=self.update_threshold_label)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=11, column=0, sticky=tk.W, pady=(20, 0))
        
        # Video display frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video canvas
        self.canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Fire Detection Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Results text widget
        self.results_text = tk.Text(results_frame, height=10, width=30, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
    def update_threshold_label(self, value):
        """Update the threshold label display"""
        self.threshold_label.config(text=f"{float(value):.2f}")
        
    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
        
    def log_result(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        
    def load_models(self):
        """Load the fire detection model"""
        try:
            self.update_status("Loading fire detection model...", "orange")
            
            if self.fire_enabled.get():
                self.fire_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.fire_model = ViTForImageClassification.from_pretrained('EdBianchi/vit-fire-detection')
                self.fire_model.to(self.fire_device)
                self.fire_feature_extractor = ViTFeatureExtractor.from_pretrained('EdBianchi/vit-fire-detection')
                self.fire_model.eval()
                self.log_result("✅ Fire detection model loaded successfully")
                
                # Show device info
                device_info = "GPU" if self.fire_device.type == "cuda" else "CPU"
                self.log_result(f"🔧 Using device: {device_info}")
            
            self.models_loaded = True
            self.update_status("Fire model loaded successfully", "green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fire detection model: {str(e)}")
            self.update_status("Model loading failed", "red")
            self.log_result(f"❌ Model loading failed: {str(e)}")
            
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
                
                # Use adjustable threshold
                threshold = self.threshold_var.get()
                label = "fire" if probs[0] > threshold else "normal"
                
            return label, probs, id2label
        except Exception as e:
            self.log_result(f"❌ Fire prediction error: {str(e)}")
            return None, None, None
    
    def start_webcam(self):
        """Start webcam detection"""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please load the fire detection model first")
            return
            
        if not self.fire_enabled.get():
            messagebox.showwarning("Warning", "Please enable fire detection")
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
            
        self.video_running = True
        self.webcam_btn.config(state=tk.DISABLED)
        self.video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.current_thread = threading.Thread(target=self.process_webcam)
        self.current_thread.daemon = True
        self.current_thread.start()
        
        self.update_status("Webcam running", "green")
        self.log_result("🔴 Webcam started - monitoring for fire...")
        
    def load_video(self):
        """Load and process video file"""
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please load the fire detection model first")
            return
            
        if not self.fire_enabled.get():
            messagebox.showwarning("Warning", "Please enable fire detection")
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
        self.webcam_btn.config(state=tk.DISABLED)
        self.video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.current_thread = threading.Thread(target=self.process_video)
        self.current_thread.daemon = True
        self.current_thread.start()
        
        self.update_status("Video processing", "green")
        self.log_result(f"📹 Video loaded: {os.path.basename(video_path)}")
        
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
                self.log_result("📹 Video processing completed")
                break
                
            self.process_frame(frame)
            
            # Add small delay for video playback
            cv2.waitKey(30)
            
    def process_frame(self, frame):
        """Process a single frame"""
        try:
            # Make fire prediction
            fire_label, fire_probs, fire_id2label = self.predict_fire(frame)
            
            # Draw overlays
            display_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            
            # Fire detection overlay
            if fire_label is not None:
                color = (0, 0, 255) if fire_label == "fire" else (0, 255, 0)
                cv2.putText(display_frame, f"Fire Status: {fire_label.upper()}", (10, y_offset), font, 0.8, color, 2)
                y_offset += 35
                
                # Show threshold
                threshold = self.threshold_var.get()
                cv2.putText(display_frame, f"Threshold: {threshold:.2f}", (10, y_offset), font, 0.6, (255, 255, 255), 2)
                y_offset += 25
                
                if fire_id2label is not None:
                    for idx, label_name in fire_id2label.items():
                        prob = fire_probs[int(idx)]
                        text = f"{label_name}: {prob:.3f}"
                        text_color = (0, 255, 0) if label_name == "normal" else (0, 0, 255)
                        cv2.putText(display_frame, text, (10, y_offset), font, 0.6, text_color, 2)
                        y_offset += 25
                
                # Log fire detection
                if fire_label == "fire":
                    confidence = fire_probs[0]
                    self.log_result(f"🔥 FIRE DETECTED! Confidence: {confidence:.3f} (Threshold: {threshold:.2f})")
            
            # Update display
            self.update_display(display_frame)
            
        except Exception as e:
            self.log_result(f"❌ Frame processing error: {str(e)}")
            
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
        self.log_result("⏹️ Detection stopped")
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FireDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()