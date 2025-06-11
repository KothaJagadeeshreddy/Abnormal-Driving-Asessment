import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from typing import Optional
from .detector import DrivingDetector
from .assessment import DrivingAssessment
from .config import Config

class DrivingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = Config()
        self.detector = DrivingDetector(self.config)
        self.assessor = DrivingAssessment(self.config)
        self.current_video: Optional[str] = None
        self.is_processing = False
        self.setup_ui()

    def setup_ui(self) -> None:
        self.root.title("Abnormal Driving Detection System")
        self.root.geometry("1024x768")
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            control_frame, 
            text="Load Video", 
            command=self.load_video
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Start Analysis", 
            command=self.start_analysis
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Stop Analysis", 
            command=self.stop_analysis
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Generate Report", 
            command=self.generate_report
        ).pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN
        ).pack(fill=tk.X, pady=(0, 10))

    def load_video(self) -> None:
        file_path = filedialog.askopenfilename(
            initialdir=self.config.VIDEO_DIR,
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.current_video = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            self.preview_video(file_path)

    def preview_video(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 600))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        cap.release()

    def start_analysis(self) -> None:
        if not self.current_video:
            messagebox.showerror("Error", "Please load a video first")
            return
            
        self.is_processing = True
        self.status_var.set("Processing video...")
        self.process_video()

    def process_video(self) -> None:
        # Implementation would go here
        pass

    def stop_analysis(self) -> None:
        self.is_processing = False
        self.status_var.set("Analysis stopped")

    def generate_report(self) -> None:
        report = self.assessor.generate_report()
        # Implementation to display/save report would go here
        messagebox.showinfo("Report Generated", f"Found {report['total_abnormal_events']} abnormal events")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrivingApp(root)
    root.mainloop()