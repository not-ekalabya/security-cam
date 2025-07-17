import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import sys
import os
from datetime import datetime, timedelta
import json

# Import your existing detection functions
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
import warnings
import torch.nn.functional as F
from keras.models import Sequential
from keras.layers import Input, TimeDistributed, Dropout, Flatten, LSTM, Bidirectional, Dense
from keras.applications.mobilenet_v2 import MobileNetV2

import pywhatkit as kit
import subprocess

warnings.filterwarnings("ignore")

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.config_file = "camera_config.json"
        self.load_camera_config()
        
    def load_camera_config(self):
        """Load camera configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.cameras = json.load(f)
            except:
                self.cameras = {}
        else:
            # Default cameras with officer phone numbers
            self.cameras = {
                "0": {
                    "name": "Default Camera", 
                    "location": "Main Entrance",
                    "officer_phone": "+1234567890"
                },
                "1": {
                    "name": "Camera 1", 
                    "location": "Lobby",
                    "officer_phone": "+1234567891"
                },
                "2": {
                    "name": "Camera 2", 
                    "location": "Parking",
                    "officer_phone": "+1234567892"
                }
            }
            self.save_camera_config()
            
    def save_camera_config(self):
        """Save camera configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.cameras, f, indent=2)
        except Exception as e:
            print(f"Error saving camera config: {e}")
            
    def add_camera(self, camera_id, name, location, officer_phone):
        """Add or update camera with officer phone"""
        self.cameras[str(camera_id)] = {
            "name": name, 
            "location": location,
            "officer_phone": officer_phone
        }
        self.save_camera_config()
        
    def get_camera_info(self, camera_id):
        """Get camera information"""
        return self.cameras.get(str(camera_id), {
            "name": f"Camera {camera_id}", 
            "location": "Unknown",
            "officer_phone": "+0000000000"
        })
        
    def get_all_cameras(self):
        """Get all cameras"""
        return self.cameras

import pyttsx3

class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.alerts_dir = "alerts"
        self.metadata_file = "alerts/alerts_metadata.json"
        
        # Enhanced alert tracking - track by camera location and detection type
        self.last_alert_time = {}  # Format: {f"{camera_location}_{detection_type}": datetime}
        self.last_alert_details = {}  # Format: {f"{camera_location}_{detection_type}": {"confidence": float, "count": int}}
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 170)  # Set speech rate (optional)
        
        # WhatsApp settings
        self.whatsapp_enabled = True
        
        # Create alerts directory if it doesn't exist
        if not os.path.exists(self.alerts_dir):
            os.makedirs(self.alerts_dir)
            
        # Load existing alerts metadata
        self.load_alerts_metadata()
        
        # Path to Arduino script
        self.arduino_script = "arduino.py"
        
    def load_alerts_metadata(self):
        """Load existing alerts metadata from file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.alerts = json.load(f)
            except:
                self.alerts = []
        else:
            self.alerts = []
            
    def save_alerts_metadata(self):
        """Save alerts metadata to file"""
        try:
            # Convert numpy types to Python native types for JSON serialization
            serializable_alerts = []
            for alert in self.alerts:
                serializable_alert = alert.copy()
                
                # Convert numpy float32/float64 to Python float
                if isinstance(serializable_alert['confidence'], (np.float32, np.float64)):
                    serializable_alert['confidence'] = float(serializable_alert['confidence'])
                
                # Convert numpy arrays in details to Python lists
                if 'details' in serializable_alert and serializable_alert['details']:
                    details = serializable_alert['details'].copy()
                    if 'probabilities' in details:
                        if isinstance(details['probabilities'], np.ndarray):
                            details['probabilities'] = details['probabilities'].tolist()
                        elif isinstance(details['probabilities'], list):
                            # Convert individual numpy float32/float64 to Python float
                            details['probabilities'] = [float(p) if isinstance(p, (np.float32, np.float64)) else p 
                                                       for p in details['probabilities']]
                    serializable_alert['details'] = details
                
                serializable_alerts.append(serializable_alert)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_alerts, f, indent=2)
        except Exception as e:
            print(f"Error saving alerts metadata: {e}")
            
    def can_create_alert(self, detection_type, delay_seconds, camera_location, confidence=None):
        """
        Enhanced alert prevention - check if enough time has passed since last alert 
        for this specific camera location and detection type
        """
        alert_key = f"{camera_location}_{detection_type}"
        
        if alert_key not in self.last_alert_time:
            return True
            
        time_since_last = datetime.now() - self.last_alert_time[alert_key]
        time_passed = time_since_last.total_seconds()
        
        # Check if enough time has passed
        if time_passed < delay_seconds:
            return False
            
        # Additional check: if confidence is similar to last alert, extend delay
        if alert_key in self.last_alert_details and confidence is not None:
            last_confidence = self.last_alert_details[alert_key].get('confidence', 0)
            confidence_diff = abs(confidence - last_confidence)
            
            # If confidence is very similar (within 0.1), extend delay by 50%
            if confidence_diff < 0.1:
                extended_delay = delay_seconds * 1.5
                if time_passed < extended_delay:
                    return False
        
        return True
        
    def update_alert_tracking(self, detection_type, camera_location, confidence):
        """Update tracking information for this alert"""
        alert_key = f"{camera_location}_{detection_type}"
        current_time = datetime.now()
        
        self.last_alert_time[alert_key] = current_time
        
        if alert_key not in self.last_alert_details:
            self.last_alert_details[alert_key] = {"confidence": confidence, "count": 1}
        else:
            self.last_alert_details[alert_key]["confidence"] = confidence
            self.last_alert_details[alert_key]["count"] += 1
            
    def get_alert_statistics(self):
        """Get statistics about recent alerts"""
        stats = {}
        current_time = datetime.now()
        
        for alert_key, last_time in self.last_alert_time.items():
            time_since_last = current_time - last_time
            details = self.last_alert_details.get(alert_key, {})
            
            stats[alert_key] = {
                "last_alert": last_time.isoformat(),
                "time_since_last": time_since_last.total_seconds(),
                "total_alerts": details.get("count", 0),
                "last_confidence": details.get("confidence", 0)
            }
            
        return stats
        
    def get_time_until_next_alert(self, detection_type, delay_seconds, camera_location):
        """Get time remaining until next alert can be sent"""
        alert_key = f"{camera_location}_{detection_type}"
        
        if alert_key not in self.last_alert_time:
            return 0
            
        time_since_last = datetime.now() - self.last_alert_time[alert_key]
        time_passed = time_since_last.total_seconds()
        
        if time_passed >= delay_seconds:
            return 0
            
        return delay_seconds - time_passed
        
    def send_whatsapp_message(self, phone_number, detection_type, confidence, camera_info, messaging_delay=20):
        """Send instant WhatsApp message to officer"""
        try:
            if not self.whatsapp_enabled:
                print(f"WhatsApp messaging is disabled - skipping message to {phone_number}")
                return False
                
            # Format phone number (ensure it starts with +)
            if not phone_number.startswith('+'):
                phone_number = '+' + phone_number.lstrip('+0')
            
            # Create alert message
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conf_percent = confidence * 100 if isinstance(confidence, float) else confidence
            
            message = (
                f"🚨 SECURITY ALERT 🚨\n\n"
                f"Type: {detection_type.upper()}\n"
                f"Location: {camera_info['location']}\n"
                f"Camera: {camera_info['name']}\n"
                f"Confidence: {conf_percent:.1f}%\n"
                f"Time: {timestamp}\n\n"
                f"Please respond immediately!"
            )
            
            # Send instant WhatsApp message
            kit.sendwhatmsg_instantly(
                phone_number, 
                message,
                wait_time=messaging_delay,  # Use user-set delay
                tab_close=True
            )
            
            print(f"WhatsApp message sent to {phone_number}")
            return True
            
        except Exception as e:
            print(f"WhatsApp sending error: {e}")
            return False
            
    def is_messaging_enabled(self):
        """Check if WhatsApp messaging is enabled"""
        return self.whatsapp_enabled
            
    def speak_alert(self, detection_type, confidence, camera_info):
        """Speak out the alert details using TTS"""
        try:
            location = camera_info.get('location', 'Unknown location')
            camera_name = camera_info.get('name', 'Unknown camera')
            conf_percent = confidence * 100 if isinstance(confidence, float) else confidence
            message = (
                f"Alert! {detection_type.capitalize()} detected. "
                f"Location: {location}. "
                f"Camera: {camera_name}. "
                f"Confidence: {conf_percent:.1f} percent."
            )
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            
    def save_alert(self, frame, detection_type, confidence, camera_info, details=None, messaging_delay=20):
        """Save an alert with frame snapshot, speak out the alert, and send WhatsApp"""
        timestamp = datetime.now()
        filename = f"{detection_type}_{camera_info['name'].replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self.alerts_dir, filename)
        
        # Save frame
        cv2.imwrite(filepath, frame)
        
        # Extract confidence value properly
        if hasattr(confidence, 'item'):
            conf_value = confidence.item()
        elif isinstance(confidence, (np.float32, np.float64)):
            conf_value = float(confidence)
        else:
            conf_value = confidence
        
        # Create alert record
        alert_record = {
            'timestamp': timestamp.isoformat(),
            'type': detection_type,
            'confidence': conf_value,
            'filename': filename,
            'filepath': filepath,
            'camera_name': camera_info['name'],
            'camera_location': camera_info['location'],
            'officer_phone': camera_info.get('officer_phone', 'N/A'),
            'whatsapp_sent': False,
            'details': details or {}
        }
        
        # Add to alerts list
        self.alerts.insert(0, alert_record)  # Insert at beginning for latest first
        
        # Update last alert time
        self.update_alert_tracking(detection_type, camera_info['location'], conf_value)
        
        # Save metadata
        self.save_alerts_metadata()
        
        # Run Arduino script (non-blocking)
        self.run_arduino_script()
        
        # Speak out the alert
        self.speak_alert(detection_type, conf_value, camera_info)
        
        # Send WhatsApp message in a separate thread
        if camera_info.get('officer_phone') and camera_info['officer_phone'] not in ['N/A', '+0000000000']:
            whatsapp_thread = threading.Thread(
                target=self._send_whatsapp_async,
                args=(camera_info['officer_phone'], detection_type, conf_value, camera_info, alert_record, messaging_delay),
                daemon=True
            )
            whatsapp_thread.start()
        
        return alert_record
    
    def _send_whatsapp_async(self, phone_number, detection_type, confidence, camera_info, alert_record, messaging_delay):
        """Send WhatsApp message asynchronously"""
        try:
            success = self.send_whatsapp_message(phone_number, detection_type, confidence, camera_info, messaging_delay)
            alert_record['whatsapp_sent'] = success
            self.save_alerts_metadata()  # Update with WhatsApp status
        except Exception as e:
            print(f"Async WhatsApp error: {e}")
        
    def get_alerts(self):
        """Get all alerts"""
        return self.alerts
        
    def clear_alerts(self):
        """Clear all alerts"""
        # Delete image files
        for alert in self.alerts:
            if os.path.exists(alert['filepath']):
                try:
                    os.remove(alert['filepath'])
                except:
                    pass
                    
        # Clear metadata and reset timers
        self.alerts = []
        self.last_alert_time = {}
        self.last_alert_details = {}
        self.save_alerts_metadata()

    def run_arduino_script(self):
        """Run the arduino.py script as a subprocess in a separate thread"""
        def _run():
                subprocess.run([sys.executable, self.arduino_script], check=True)
        threading.Thread(target=_run, daemon=True).start()

class CameraConfigDialog:
    def __init__(self, parent, camera_manager):
        self.parent = parent
        self.camera_manager = camera_manager
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Camera Configuration")
        self.dialog.geometry("750x550")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        self.refresh_camera_list()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        ttk.Label(main_frame, text="Camera Configuration with WhatsApp Alerts", 
                 font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Camera list
        list_frame = ttk.LabelFrame(main_frame, text="Existing Cameras", padding="10")
        list_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Treeview for cameras
        columns = ('ID', 'Name', 'Location', 'Officer Phone')
        self.camera_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        self.camera_tree.heading('ID', text='Camera ID')
        self.camera_tree.heading('Name', text='Name')
        self.camera_tree.heading('Location', text='Location')
        self.camera_tree.heading('Officer Phone', text='Officer Phone')
        
        self.camera_tree.column('ID', width=70)
        self.camera_tree.column('Name', width=120)
        self.camera_tree.column('Location', width=150)
        self.camera_tree.column('Officer Phone', width=130)
        
        # Scrollbar
        tree_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.camera_tree.yview)
        self.camera_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.camera_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Add/Edit camera form
        form_frame = ttk.LabelFrame(main_frame, text="Add/Edit Camera", padding="10")
        form_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Row 1: Camera ID and Name
        ttk.Label(form_frame, text="Camera ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.camera_id_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.camera_id_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(form_frame, text="Name:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.camera_name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.camera_name_var, width=20).grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # Row 2: Location
        ttk.Label(form_frame, text="Location:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.camera_location_var = tk.StringVar()
        location_entry = ttk.Entry(form_frame, textvariable=self.camera_location_var, width=40)
        location_entry.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Row 3: Officer Phone
        ttk.Label(form_frame, text="Officer Phone:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.officer_phone_var = tk.StringVar()
        phone_entry = ttk.Entry(form_frame, textvariable=self.officer_phone_var, width=20)
        phone_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(10, 0), padx=(0, 10))
        
        # Phone format hint
        ttk.Label(form_frame, text="(Format: +1234567890)", foreground="gray").grid(
            row=2, column=2, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        form_frame.columnconfigure(3, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=(15, 0))
        
        ttk.Button(button_frame, text="Add/Update", command=self.add_camera).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Delete", command=self.delete_camera).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Test WhatsApp", command=self.test_whatsapp).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(button_frame, text="Close", command=self.close_dialog).grid(row=0, column=3)
        
        # WhatsApp instructions
        instructions_frame = ttk.LabelFrame(main_frame, text="WhatsApp Setup Instructions", padding="10")
        instructions_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        instructions_text = """1. Make sure WhatsApp Web is logged in on your browser
2. Phone numbers must include country code (e.g., +1 for US)
3. Test messages will open WhatsApp Web automatically
4. For instant alerts, keep WhatsApp Web open in browser"""
        
        ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT, 
                 wraplength=600, foreground="blue").grid(row=0, column=0, sticky=tk.W)
        
        # Bind selection
        self.camera_tree.bind('<<TreeviewSelect>>', self.on_camera_select)
        
    def refresh_camera_list(self):
        """Refresh the camera list"""
        for item in self.camera_tree.get_children():
            self.camera_tree.delete(item)
            
        cameras = self.camera_manager.get_all_cameras()
        for camera_id, info in cameras.items():
            self.camera_tree.insert('', 'end', values=(
                camera_id, 
                info['name'], 
                info['location'],
                info.get('officer_phone', 'N/A')
            ))
            
    def on_camera_select(self, event):
        """Handle camera selection"""
        selection = self.camera_tree.selection()
        if selection:
            item = selection[0]
            values = self.camera_tree.item(item, 'values')
            self.camera_id_var.set(values[0])
            self.camera_name_var.set(values[1])
            self.camera_location_var.set(values[2])
            self.officer_phone_var.set(values[3] if values[3] != 'N/A' else '')
            
    def add_camera(self):
        """Add or update camera"""
        camera_id = self.camera_id_var.get().strip()
        name = self.camera_name_var.get().strip()
        location = self.camera_location_var.get().strip()
        officer_phone = self.officer_phone_var.get().strip()
        
        if not camera_id or not name or not location:
            messagebox.showwarning("Warning", "Please fill in Camera ID, Name, and Location fields")
            return
            
        # Validate phone number format if provided
        if officer_phone and not officer_phone.startswith('+'):
            messagebox.showwarning("Warning", "Phone number should start with + and include country code (e.g., +1234567890)")
            return
            
        # Use default phone if none provided
        if not officer_phone:
            officer_phone = "+0000000000"
            
        self.camera_manager.add_camera(camera_id, name, location, officer_phone)
        self.refresh_camera_list()
        
        # Clear form
        self.camera_id_var.set("")
        self.camera_name_var.set("")
        self.camera_location_var.set("")
        self.officer_phone_var.set("")
        
    def test_whatsapp(self):
        """Test WhatsApp functionality"""
        phone = self.officer_phone_var.get().strip()
        if not phone:
            messagebox.showwarning("Warning", "Please enter a phone number to test")
            return
            
        if not phone.startswith('+'):
            messagebox.showwarning("Warning", "Phone number should start with + and include country code")
            return
            
        try:
            # Send test message
            test_message = (
                f"🧪 TEST MESSAGE 🧪\n\n"
                f"This is a test message from the Fire & Violence Detection System.\n"
                f"Camera: {self.camera_name_var.get() or 'Test Camera'}\n"
                f"Location: {self.camera_location_var.get() or 'Test Location'}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"If you receive this, WhatsApp notifications are working correctly!"
            )
            
            # Send immediately
            kit.sendwhatmsg_instantly(
                phone, 
                test_message,
                wait_time=20,
                tab_close=True
            )
            
            messagebox.showinfo("Success", f"Test message sent to {phone}\nCheck WhatsApp Web for delivery.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send test message: {str(e)}\n\nMake sure WhatsApp Web is logged in.")
        
    def delete_camera(self):
        """Delete selected camera"""
        selection = self.camera_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a camera to delete")
            return
            
        item = selection[0]
        camera_id = self.camera_tree.item(item, 'values')[0]
        
        result = messagebox.askyesno("Confirm", f"Delete camera {camera_id}?")
        if result:
            cameras = self.camera_manager.get_all_cameras()
            if camera_id in cameras:
                del cameras[camera_id]
                self.camera_manager.cameras = cameras
                self.camera_manager.save_camera_config()
                self.refresh_camera_list()
                
    def close_dialog(self):
        """Close the dialog"""
        self.dialog.destroy()

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Responda")
        self.root.geometry("1500x900")
        
        # Variables
        self.video_running = False
        self.cap = None
        self.fire_enabled = tk.BooleanVar(value=True)
        self.violence_enabled = tk.BooleanVar(value=True)
        self.current_thread = None
        
        # Frame processing variables
        self.frame_interval = tk.IntVar(value=3)  # Process every 3rd frame by default
        self.frame_counter = 0
        self.last_predictions = {'fire': None, 'violence': None}  # Store last predictions
        
        # Camera and alert system
        self.camera_manager = CameraManager()
        self.alert_system = AlertSystem()
        self.current_camera_id = tk.StringVar(value="0")
        
        # Alert delay settings (in seconds)
        self.fire_alert_delay = tk.IntVar(value=30)  # 30 seconds between fire alerts
        self.violence_alert_delay = tk.IntVar(value=15)  # 15 seconds between violence alerts
        
        # Alert thresholds
        self.fire_threshold = tk.DoubleVar(value=0.80)
        self.violence_threshold = tk.DoubleVar(value=0.99)
        
        # Messaging toggle
        self.messaging_enabled = tk.BooleanVar(value=True)
        
        # Model variables
        self.fire_model = None
        self.fire_feature_extractor = None
        self.fire_device = None
        self.violence_model = None
        self.models_loaded = False
        
        # Violence detection constants
        self.SEQUENCE_LENGTH = 4
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = 64, 64
        self.CLASSES_LIST = ["NonViolence", "Violence"]
        self.MODEL_PATH = 'violence_detection_model.h5'
        self.frame_buffer = []
        
        self.setup_ui()
        self.refresh_alerts_display()
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main detection tab
        self.setup_main_tab(notebook)
        
        # Camera configuration tab
        self.setup_camera_tab(notebook)
        
    def setup_main_tab(self, notebook):
        # Main detection frame
        main_frame = ttk.Frame(notebook, padding="10")
        notebook.add(main_frame, text="Detection System")
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left Panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        row = 0
        
        # Camera selection
        ttk.Label(control_frame, text="Camera Settings:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1
        
        camera_select_frame = ttk.Frame(control_frame)
        camera_select_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Label(camera_select_frame, text="Camera ID:").grid(row=0, column=0, sticky=tk.W)
        camera_spinbox = ttk.Spinbox(camera_select_frame, from_=0, to=10, width=5, textvariable=self.current_camera_id)
        camera_spinbox.grid(row=0, column=1, padx=(5, 10))
        
        self.camera_info_label = ttk.Label(camera_select_frame, text="", foreground="blue")
        self.camera_info_label.grid(row=0, column=2, sticky=tk.W)
        
        # Bind camera change
        self.current_camera_id.trace('w', self.on_camera_change)
        self.on_camera_change()  # Initialize
        
        # Detection toggles
        ttk.Label(control_frame, text="Detection Options:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        fire_check = ttk.Checkbutton(control_frame, text="Fire Detection", variable=self.fire_enabled)
        fire_check.grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        violence_check = ttk.Checkbutton(control_frame, text="Violence Detection", variable=self.violence_enabled)
        violence_check.grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        # Alert Thresholds
        ttk.Label(control_frame, text="Alert Thresholds:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        # Fire threshold
        fire_threshold_frame = ttk.Frame(control_frame)
        fire_threshold_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        ttk.Label(fire_threshold_frame, text="Fire:").grid(row=0, column=0, sticky=tk.W)
        fire_threshold_scale = ttk.Scale(fire_threshold_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                                        variable=self.fire_threshold, length=100)
        fire_threshold_scale.grid(row=0, column=1, padx=(5, 5))
        self.fire_threshold_label = ttk.Label(fire_threshold_frame, text="0.80")
        self.fire_threshold_label.grid(row=0, column=2)
        fire_threshold_scale.configure(command=self.update_fire_threshold_label)
        
        # Violence threshold
        violence_threshold_frame = ttk.Frame(control_frame)
        violence_threshold_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        ttk.Label(violence_threshold_frame, text="Violence:").grid(row=0, column=0, sticky=tk.W)
        violence_threshold_scale = ttk.Scale(violence_threshold_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                                           variable=self.violence_threshold, length=100)
        violence_threshold_scale.grid(row=0, column=1, padx=(5, 5))
        self.violence_threshold_label = ttk.Label(violence_threshold_frame, text="0.99")
        self.violence_threshold_label.grid(row=0, column=2)
        violence_threshold_scale.configure(command=self.update_violence_threshold_label)
        
        # Alert Delays
        ttk.Label(control_frame, text="Alert Delays (seconds):", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        # Fire delay
        fire_delay_frame = ttk.Frame(control_frame)
        fire_delay_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        ttk.Label(fire_delay_frame, text="Fire delay:").grid(row=0, column=0, sticky=tk.W)
        fire_delay_spinbox = ttk.Spinbox(fire_delay_frame, from_=5, to=300, width=8, textvariable=self.fire_alert_delay)
        fire_delay_spinbox.grid(row=0, column=1, padx=(5, 5))
        ttk.Label(fire_delay_frame, text="sec").grid(row=0, column=2, sticky=tk.W)
        
        # Violence delay
        violence_delay_frame = ttk.Frame(control_frame)
        violence_delay_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        ttk.Label(violence_delay_frame, text="Violence delay:").grid(row=0, column=0, sticky=tk.W)
        violence_delay_spinbox = ttk.Spinbox(violence_delay_frame, from_=5, to=300, width=8, textvariable=self.violence_alert_delay)
        violence_delay_spinbox.grid(row=0, column=1, padx=(5, 5))
        ttk.Label(violence_delay_frame, text="sec").grid(row=0, column=2, sticky=tk.W)
        
        # Messaging Settings
        ttk.Label(control_frame, text="Messaging Settings:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        messaging_frame = ttk.Frame(control_frame)
        messaging_frame.grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        messaging_check = ttk.Checkbutton(messaging_frame, text="Enable WhatsApp Messaging", 
                                         variable=self.messaging_enabled, 
                                         command=self.on_messaging_toggle)
        messaging_check.grid(row=0, column=0, sticky=tk.W)
        
        # Messaging status indicator
        self.messaging_status_label = ttk.Label(messaging_frame, text="✓ Enabled", foreground="green")
        self.messaging_status_label.grid(row=0, column=1, padx=(10, 0))
        
        # Messaging delay
        ttk.Label(messaging_frame, text="Message Delay (sec):").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self.messaging_delay = tk.IntVar(value=20)
        self.messaging_delay_spinbox = ttk.Spinbox(messaging_frame, from_=10, to=60, width=5, textvariable=self.messaging_delay)
        self.messaging_delay_spinbox.grid(row=1, column=1, sticky=tk.W, pady=(8, 0), padx=(5, 0))
        
        # Warning label
        ttk.Label(messaging_frame, text="(Minimum 15–20 sec recommended for WhatsApp Web)", foreground="orange").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        
        # Frame processing interval
        ttk.Label(control_frame, text="Performance Settings:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        interval_frame = ttk.Frame(control_frame)
        interval_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Label(interval_frame, text="Process every").grid(row=0, column=0, sticky=tk.W)
        interval_spinbox = ttk.Spinbox(interval_frame, from_=1, to=10, width=5, 
                                      textvariable=self.frame_interval, 
                                      command=self.on_interval_change)
        interval_spinbox.grid(row=0, column=1, padx=(5, 5))
        ttk.Label(interval_frame, text="frames").grid(row=0, column=2, sticky=tk.W)
        
        # Performance indicator
        self.performance_label = ttk.Label(control_frame, text="Performance: Normal", foreground="blue")
        self.performance_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        # Input source
        ttk.Label(control_frame, text="Input Source:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        self.webcam_btn = ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam)
        self.webcam_btn.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        self.video_btn = ttk.Button(control_frame, text="Load Video File", command=self.load_video)
        self.video_btn.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Model loading
        ttk.Label(control_frame, text="Model Management:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(20, 10))
        row += 1
        
        self.load_models_btn = ttk.Button(control_frame, text="Load Models", command=self.load_models)
        self.load_models_btn.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=row, column=0, sticky=tk.W, pady=(20, 0))
        
        # Right Panel - Video and Alerts
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=2)
        right_panel.columnconfigure(1, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Video display
        video_frame = ttk.LabelFrame(right_panel, text="Video Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Video canvas
        self.canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Detection results below video
        results_frame = ttk.LabelFrame(video_frame, text="Detection Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.results_text = tk.Text(results_frame, height=6, width=50, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        
        # Alerts Panel
        alerts_frame = ttk.LabelFrame(right_panel, text="Alert System", padding="10")
        alerts_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Alert controls
        alert_controls = ttk.Frame(alerts_frame)
        alert_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.refresh_alerts_btn = ttk.Button(alert_controls, text="Refresh", command=self.refresh_alerts_display)
        self.refresh_alerts_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.clear_alerts_btn = ttk.Button(alert_controls, text="Clear All", command=self.clear_all_alerts)
        self.clear_alerts_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.stats_btn = ttk.Button(alert_controls, text="Stats", command=self.show_alert_statistics)
        self.stats_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.alert_count_label = ttk.Label(alert_controls, text="Alerts: 0", font=('Arial', 10, 'bold'))
        self.alert_count_label.grid(row=0, column=3, padx=(10, 0))
        
        # Alerts list
        alerts_list_frame = ttk.Frame(alerts_frame)
        alerts_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        alerts_frame.columnconfigure(0, weight=1)
        alerts_frame.rowconfigure(1, weight=1)
        
        # Treeview for alerts
        columns = ('Time', 'Type', 'Location', 'Confidence')
        self.alerts_tree = ttk.Treeview(alerts_list_frame, columns=columns, show='headings', height=12)
        
        # Configure columns
        self.alerts_tree.heading('Time', text='Time')
        self.alerts_tree.heading('Type', text='Type')
        self.alerts_tree.heading('Location', text='Location')
        self.alerts_tree.heading('Confidence', text='Confidence')
        
        self.alerts_tree.column('Time', width=100)
        self.alerts_tree.column('Type', width=70)
        self.alerts_tree.column('Location', width=100)
        self.alerts_tree.column('Confidence', width=80)
        
        # Scrollbar for alerts
        alerts_scrollbar = ttk.Scrollbar(alerts_list_frame, orient=tk.VERTICAL, command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=alerts_scrollbar.set)
        
        self.alerts_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        alerts_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        alerts_list_frame.columnconfigure(0, weight=1)
        alerts_list_frame.rowconfigure(0, weight=1)
        
        # Bind double-click to view alert
        self.alerts_tree.bind('<Double-1>', self.view_alert_image)
        
        # Alert preview
        preview_frame = ttk.LabelFrame(alerts_frame, text="Preview", padding="5")
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.preview_canvas = tk.Canvas(preview_frame, bg="gray", width=200, height=150)
        self.preview_canvas.grid(row=0, column=0)
        
        # Initialize performance label
        self.update_performance_label()
        
    def setup_camera_tab(self, notebook):
        """Setup camera configuration tab"""
        camera_frame = ttk.Frame(notebook, padding="20")
        notebook.add(camera_frame, text="Camera Configuration")
        
        # Title
        ttk.Label(camera_frame, text="Camera Configuration", font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=(0, 30))
        
        # Instructions
        instructions = """Configure cameras by assigning names and locations to camera IDs.
These locations will appear in alert records for easy identification.

Camera ID corresponds to the OpenCV camera index (0, 1, 2, etc.)
Use the Detection System tab to select which camera to use for monitoring."""
        
        ttk.Label(camera_frame, text=instructions, justify=tk.LEFT, wraplength=500).grid(row=1, column=0, pady=(0, 20))
        
        # Camera configuration button
        config_btn = ttk.Button(camera_frame, text="Open Camera Configuration", 
                               command=self.open_camera_config, width=30)
        config_btn.grid(row=2, column=0, pady=20)
        
        # Current camera list display
        list_frame = ttk.LabelFrame(camera_frame, text="Current Camera Configuration", padding="10")
        list_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=20)
        
        # Camera display tree
        cam_columns = ('ID', 'Name', 'Location')
        self.camera_display_tree = ttk.Treeview(list_frame, columns=cam_columns, show='headings', height=8)
        
        self.camera_display_tree.heading('ID', text='Camera ID')
        self.camera_display_tree.heading('Name', text='Camera Name')
        self.camera_display_tree.heading('Location', text='Location')
        
        self.camera_display_tree.column('ID', width=100)
        self.camera_display_tree.column('Name', width=200)
        self.camera_display_tree.column('Location', width=300)
        
        cam_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.camera_display_tree.yview)
        self.camera_display_tree.configure(yscrollcommand=cam_scrollbar.set)
        
        self.camera_display_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cam_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(3, weight=1)
        
        # Refresh camera display
        self.refresh_camera_display()
        
    def on_camera_change(self, *args):
        """Handle camera selection change"""
        camera_id = self.current_camera_id.get()
        camera_info = self.camera_manager.get_camera_info(camera_id)
        self.camera_info_label.config(text=f"{camera_info['name']} - {camera_info['location']}")
        
    def open_camera_config(self):
        """Open camera configuration dialog"""
        dialog = CameraConfigDialog(self.root, self.camera_manager)
        self.root.wait_window(dialog.dialog)
        self.refresh_camera_display()
        self.on_camera_change()  # Update current camera info
        
    def refresh_camera_display(self):
        """Refresh camera display in the configuration tab"""
        for item in self.camera_display_tree.get_children():
            self.camera_display_tree.delete(item)
            
        cameras = self.camera_manager.get_all_cameras()
        for camera_id, info in sorted(cameras.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
            self.camera_display_tree.insert('', 'end', values=(camera_id, info['name'], info['location']))
        
    def update_fire_threshold_label(self, value=None):
        """Update fire threshold label"""
        self.fire_threshold_label.config(text=f"{self.fire_threshold.get():.2f}")
        
    def update_violence_threshold_label(self, value=None):
        """Update violence threshold label"""
        self.violence_threshold_label.config(text=f"{self.violence_threshold.get():.2f}")
        
    def on_interval_change(self):
        """Handle frame interval change"""
        self.update_performance_label()
        
    def update_performance_label(self):
        """Update performance indicator based on frame interval"""
        interval = self.frame_interval.get()
        if interval == 1:
            text = "Performance: High Load (every frame)"
            color = "red"
        elif interval <= 3:
            text = "Performance: Normal"
            color = "blue"
        elif interval <= 5:
            text = "Performance: Fast"
            color = "green"
        else:
            text = "Performance: Very Fast"
            color = "darkgreen"
        
        self.performance_label.config(text=text, foreground=color)
        
    def on_messaging_toggle(self):
        """Handle messaging toggle state change"""
        if self.messaging_enabled.get():
            self.alert_system.whatsapp_enabled = True
            self.messaging_status_label.config(text="✓ Enabled", foreground="green")
            self.log_result("WhatsApp messaging enabled")
        else:
            self.alert_system.whatsapp_enabled = False
            self.messaging_status_label.config(text="✗ Disabled", foreground="red")
            self.log_result("WhatsApp messaging disabled")
        
    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
        
    def log_result(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        
    def refresh_alerts_display(self):
        """Refresh the alerts display"""
        # Clear existing items
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
            
        # Add alerts to tree
        alerts = self.alert_system.get_alerts()
        for alert in alerts:
            timestamp = datetime.fromisoformat(alert['timestamp'])
            time_str = timestamp.strftime('%H:%M:%S')
            date_str = timestamp.strftime('%m/%d')
            
            confidence = f"{alert['confidence']:.2f}"
            location = alert.get('camera_location', 'Unknown')
            
            self.alerts_tree.insert('', 'end', values=(f"{date_str} {time_str}", 
                                                      alert['type'].title(), 
                                                      location,
                                                      confidence))
        
        # Update count
        self.alert_count_label.config(text=f"Alerts: {len(alerts)}")
        
    def view_alert_image(self, event):
        """View the selected alert image"""
        selection = self.alerts_tree.selection()
        if not selection:
            return
            
        # Get selected item index
        item = selection[0]
        index = self.alerts_tree.index(item)
        
        alerts = self.alert_system.get_alerts()
        if index < len(alerts):
            alert = alerts[index]
            self.show_alert_preview(alert['filepath'])
            
    def show_alert_preview(self, filepath):
        """Show alert image in preview canvas"""
        try:
            if os.path.exists(filepath):
                # Load and resize image
                image = cv2.imread(filepath)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize to fit preview canvas
                height, width = image_rgb.shape[:2]
                canvas_width = 200
                canvas_height = 150
                
                scale = min(canvas_width / width, canvas_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized_image = cv2.resize(image_rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
                
                # Update preview canvas
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
                self.preview_canvas.image = photo  # Keep reference
                
        except Exception as e:
            print(f"Error showing preview: {e}")
            
    def clear_all_alerts(self):
        """Clear all alerts"""
        result = messagebox.askyesno("Confirm", "Are you sure you want to clear all alerts and delete saved images?")
        if result:
            self.alert_system.clear_alerts()
            self.refresh_alerts_display()
            self.preview_canvas.delete("all")
            self.log_result("All alerts cleared")
            
    def show_alert_statistics(self):
        """Show alert statistics in a popup window"""
        stats = self.alert_system.get_alert_statistics()
        
        if not stats:
            messagebox.showinfo("Alert Statistics", "No alert statistics available.")
            return
            
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Alert Statistics")
        stats_window.geometry("600x400")
        stats_window.transient(self.root)
        stats_window.grab_set()
        
        # Center the window
        stats_window.update_idletasks()
        x = (stats_window.winfo_screenwidth() // 2) - (stats_window.winfo_width() // 2)
        y = (stats_window.winfo_screenheight() // 2) - (stats_window.winfo_height() // 2)
        stats_window.geometry(f"+{x}+{y}")
        
        # Create text widget for statistics
        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(stats_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        stats_window.columnconfigure(0, weight=1)
        stats_window.rowconfigure(0, weight=1)
        
        # Format and display statistics
        text_widget.insert(tk.END, "ALERT STATISTICS\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        
        for alert_key, data in stats.items():
            location, detection_type = alert_key.split('_', 1)
            text_widget.insert(tk.END, f"Location: {location}\n")
            text_widget.insert(tk.END, f"Detection Type: {detection_type.title()}\n")
            text_widget.insert(tk.END, f"Total Alerts: {data['total_alerts']}\n")
            text_widget.insert(tk.END, f"Last Alert: {data['last_alert'][:19]}\n")
            text_widget.insert(tk.END, f"Time Since Last: {data['time_since_last']:.1f} seconds\n")
            text_widget.insert(tk.END, f"Last Confidence: {data['last_confidence']:.3f}\n")
            text_widget.insert(tk.END, "-" * 30 + "\n\n")
        
        # Add messaging status
        text_widget.insert(tk.END, f"Messaging Status: {'Enabled' if self.alert_system.is_messaging_enabled() else 'Disabled'}\n")
        
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        close_btn = ttk.Button(stats_window, text="Close", command=stats_window.destroy)
        close_btn.grid(row=1, column=0, pady=10)
        
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
                label = "fire" if probs[0] > self.fire_threshold.get() else "normal"
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
        
        camera_id = int(self.current_camera_id.get())
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {camera_id}")
            return
            
        self.video_running = True
        self.frame_buffer = []
        self.frame_counter = 0
        self.last_predictions = {'fire': None, 'violence': None}
        
        self.webcam_btn.config(state=tk.DISABLED)
        self.video_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.current_thread = threading.Thread(target=self.process_webcam)
        self.current_thread.daemon = True
        self.current_thread.start()
        
        camera_info = self.camera_manager.get_camera_info(camera_id)
        self.update_status(f"Camera {camera_id} running - {camera_info['location']}", "green")
        
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
        self.frame_counter = 0
        self.last_predictions = {'fire': None, 'violence': None}
        
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
        """Process a single frame with interval control"""
        try:
            # Increment frame counter
            self.frame_counter += 1
            
            # Check if we should process this frame
            should_process = (self.frame_counter % self.frame_interval.get() == 0)
            
            if should_process:
                # Make predictions on this frame
                fire_label, fire_probs, fire_id2label = self.predict_fire(frame)
                violence_label, violence_probs = self.predict_violence(frame)
                
                # Get current camera info
                camera_id = self.current_camera_id.get()
                camera_info = self.camera_manager.get_camera_info(camera_id)
                
                # Check for alerts and save snapshots with delay control
                alert_triggered = False
                
                # Fire alert
                if fire_label == "fire" and fire_probs is not None:
                    confidence = fire_probs[0]
                    if self.alert_system.can_create_alert("fire", self.fire_alert_delay.get(), camera_info['location'], confidence):
                        alert = self.alert_system.save_alert(frame, "fire", confidence, camera_info,
                                                           {"probabilities": fire_probs.tolist()},
                                                           self.messaging_delay.get())
                        self.log_result(f"🔥 FIRE ALERT! {camera_info['location']} - Confidence: {confidence:.3f}")
                        alert_triggered = True
                    else:
                        # Show time remaining until next alert
                        time_remaining = self.alert_system.get_time_until_next_alert("fire", self.fire_alert_delay.get(), camera_info['location'])
                        self.log_result(f"🔥 Fire detected (alert suppressed) - {camera_info['location']} - Next alert in {time_remaining:.1f}s")
                
                # Violence alert
                if violence_label == "Violence" and violence_probs is not None:
                    confidence = violence_probs[1]  # Violence class confidence
                    if confidence >= self.violence_threshold.get():
                        if self.alert_system.can_create_alert("violence", self.violence_alert_delay.get(), camera_info['location'], confidence):
                            alert = self.alert_system.save_alert(frame, "violence", confidence, camera_info,
                                                               {"probabilities": violence_probs.tolist()},
                                                               self.messaging_delay.get())
                            self.log_result(f"⚠️ VIOLENCE ALERT! {camera_info['location']} - Confidence: {confidence:.3f}")
                            alert_triggered = True
                        else:
                            # Show time remaining until next alert
                            time_remaining = self.alert_system.get_time_until_next_alert("violence", self.violence_alert_delay.get(), camera_info['location'])
                            self.log_result(f"⚠️ Violence detected (alert suppressed) - {camera_info['location']} - Next alert in {time_remaining:.1f}s")
                
                # Refresh alerts display if new alert was triggered
                if alert_triggered:
                    self.root.after(0, self.refresh_alerts_display)
                
                # Update last predictions
                if fire_label is not None:
                    self.last_predictions['fire'] = (fire_label, fire_probs, fire_id2label)
                if violence_label is not None and violence_probs is not None:
                    self.last_predictions['violence'] = (violence_label, violence_probs)
            
            # Always update display with current frame and last predictions
            self.update_frame_display(frame, should_process)
            
        except Exception as e:
            self.log_result(f"Frame processing error: {str(e)}")
            
    def update_frame_display(self, frame, was_processed):
        """Update frame display with overlays"""
        try:
            # Draw overlays using last predictions
            display_frame = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 45
            
            # Add processing indicator
            if was_processed:
                cv2.putText(display_frame, "●", (10, 20), font, 0.5, (0, 255, 0), 2)  # Green dot
            else:
                cv2.putText(display_frame, "○", (10, 20), font, 0.5, (128, 128, 128), 2)  # Gray dot
            
            # Add camera info
            camera_id = self.current_camera_id.get()
            camera_info = self.camera_manager.get_camera_info(camera_id)
            cv2.putText(display_frame, f"Cam {camera_id}: {camera_info['location']}", (40, 20), font, 0.5, (255, 255, 255), 1)
            
            # Overlay date-time (top-right corner)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            text_size, _ = cv2.getTextSize(now, font, 0.7, 2)
            x = display_frame.shape[1] - text_size[0] - 10
            y = text_size[1] + 10
            cv2.putText(display_frame, now, (x, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Fire detection overlay
            if self.last_predictions['fire'] is not None:
                fire_label, fire_probs, fire_id2label = self.last_predictions['fire']
                color = (0, 0, 255) if fire_label == "fire" else (0, 255, 0)
                cv2.putText(display_frame, f"Fire: {fire_label}", (10, y_offset), font, 0.7, color, 2)
                y_offset += 30
                
                if fire_id2label is not None:
                    for idx, label_name in fire_id2label.items():
                        prob = fire_probs[int(idx)]
                        text = f"{label_name}: {prob:.3f}"
                        cv2.putText(display_frame, text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
                        y_offset += 20
            
            # Violence detection overlay
            if self.last_predictions['violence'] is not None:
                violence_label, violence_probs = self.last_predictions['violence']
                y_offset += 10
                color = (0, 0, 255) if violence_label == "Violence" else (0, 255, 0)
                cv2.putText(display_frame, f"Violence: {violence_label}", (10, y_offset), font, 0.7, color, 2)
                y_offset += 30
                
                for idx, class_name in enumerate(self.CLASSES_LIST):
                    prob = violence_probs[idx]
                    text = f"{class_name}: {prob:.2f}"
                    cv2.putText(display_frame, text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
                    y_offset += 20
            
            # Update display
            self.update_display(display_frame)
            
        except Exception as e:
            self.log_result(f"Display update error: {str(e)}")
            
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
        self.frame_counter = 0
        self.last_predictions = {'fire': None, 'violence': None}
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