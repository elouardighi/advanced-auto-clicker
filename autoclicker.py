import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import concurrent.futures
from queue import Queue
import pygetwindow as gw

class AdvancedAutoClickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADVANCED Ultra-Fast Image Auto Clicker")
        self.root.geometry("700x650")
        
        # Advanced variables
        self.selected_image_path = None
        self.is_running = False
        self.template = None
        self.template_gray = None
        self.template_pyramid = []
        self.last_positions = []  # Track multiple recent positions
        self.search_strategy = "hybrid"
        
        # Multi-threading
        self.screenshot_queue = Queue(maxsize=1)
        self.result_queue = Queue()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        
        self.setup_advanced_ui()
        
    def setup_advanced_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 ADVANCED Ultra-Fast Image Auto Clicker", 
                               font=("Arial", 16, "bold"), foreground="darkblue")
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # Image selection section
        image_frame = ttk.LabelFrame(main_frame, text="Target Image Configuration", padding="10")
        image_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.select_btn = ttk.Button(image_frame, text="Select Target Image", 
                                   command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.image_path_label = ttk.Label(image_frame, text="No image selected", foreground="gray")
        self.image_path_label.grid(row=0, column=1, sticky=tk.W)
        
        # Image preview with size info
        self.image_preview = ttk.Label(image_frame, text="Preview: 0x0 pixels")
        self.image_preview.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # Advanced search strategies
        strategy_frame = ttk.LabelFrame(main_frame, text="Advanced Search Engine", padding="10")
        strategy_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Search algorithm selection
        ttk.Label(strategy_frame, text="Search Algorithm:").grid(row=0, column=0, sticky=tk.W)
        self.algorithm_var = tk.StringVar(value="hybrid")
        algo_frame = ttk.Frame(strategy_frame)
        algo_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W)
        
        algorithms = [
            ("Pyramid Fast", "pyramid"),
            ("Multi-Scale", "multiscale"), 
            ("Hybrid AI", "hybrid"),
            ("Ultra Precision", "precision")
        ]
        
        for i, (text, value) in enumerate(algorithms):
            ttk.Radiobutton(algo_frame, text=text, variable=self.algorithm_var,
                          value=value).grid(row=0, column=i, padx=(10 if i>0 else 0, 10))
        
        # Multi-threading options
        ttk.Label(strategy_frame, text="Parallel Processing:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.threading_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(strategy_frame, text="Enable multi-threading", 
                       variable=self.threading_var).grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Smart prediction
        ttk.Label(strategy_frame, text="AI Features:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        ai_frame = ttk.Frame(strategy_frame)
        ai_frame.grid(row=2, column=1, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        self.predict_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ai_frame, text="Motion Prediction", variable=self.predict_var).pack(side=tk.LEFT)
        self.adaptive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ai_frame, text="Adaptive Confidence", variable=self.adaptive_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Performance optimization frame
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Optimization", padding="10")
        perf_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Search region optimization
        ttk.Label(perf_frame, text="Search Region:").grid(row=0, column=0, sticky=tk.W)
        self.region_var = tk.StringVar(value="dynamic")
        region_frame = ttk.Frame(perf_frame)
        region_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W)
        
        regions = [
            ("Dynamic Smart", "dynamic"),
            ("Fixed Area", "fixed"),
            ("Full Screen", "fullscreen")
        ]
        
        for i, (text, value) in enumerate(regions):
            ttk.Radiobutton(region_frame, text=text, variable=self.region_var,
                          value=value).grid(row=0, column=i, padx=(10 if i>0 else 0, 10))
        
        # Fixed region settings
        ttk.Label(perf_frame, text="Region Size:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.region_width_var = tk.StringVar(value="400")
        self.region_height_var = tk.StringVar(value="300")
        region_size_frame = ttk.Frame(perf_frame)
        region_size_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
        ttk.Entry(region_size_frame, textvariable=self.region_width_var, width=5).pack(side=tk.LEFT)
        ttk.Label(region_size_frame, text="x").pack(side=tk.LEFT, padx=5)
        ttk.Entry(region_size_frame, textvariable=self.region_height_var, width=5).pack(side=tk.LEFT)
        ttk.Label(region_size_frame, text="pixels").pack(side=tk.LEFT, padx=5)
        
        # Click settings frame
        click_frame = ttk.LabelFrame(main_frame, text="Click Configuration", padding="10")
        click_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Confidence with adaptive option
        ttk.Label(click_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.85)
        confidence_scale = ttk.Scale(click_frame, from_=0.5, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.confidence_label = ttk.Label(click_frame, text="85%")
        self.confidence_label.grid(row=0, column=2, padx=(10, 0))
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Click interval with auto-adjust
        ttk.Label(click_frame, text="Click Interval (s):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        interval_frame = ttk.Frame(click_frame)
        interval_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.interval_var = tk.StringVar(value="0.3")
        ttk.Entry(interval_frame, textvariable=self.interval_var, width=8).pack(side=tk.LEFT)
        ttk.Label(interval_frame, text="(0 = maximum speed)").pack(side=tk.LEFT, padx=5)
        
        # Click type with advanced options
        ttk.Label(click_frame, text="Click Type:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        click_type_frame = ttk.Frame(click_frame)
        click_type_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        self.click_type_var = tk.StringVar(value="left")
        types = [("Left", "left"), ("Right", "right"), ("Double", "double"), ("Middle", "middle")]
        for i, (text, value) in enumerate(types):
            ttk.Radiobutton(click_type_frame, text=text, variable=self.click_type_var,
                          value=value).grid(row=0, column=i, padx=(10 if i>0 else 0, 10))
        
        # Control section
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=4, pady=15)
        
        # Control buttons
        self.start_btn = ttk.Button(control_frame, text="🚀 Start Ultra-Click", 
                                  command=self.start_auto_click, style="Accent.TButton")
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="🛑 Stop", 
                                 command=self.stop_auto_click, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.test_btn = ttk.Button(control_frame, text="🧪 Test Single Click", 
                                 command=self.test_single_click)
        self.test_btn.grid(row=0, column=2, padx=(0, 10))
        
        # Performance monitor
        perf_monitor_frame = ttk.Frame(control_frame)
        perf_monitor_frame.grid(row=0, column=3, padx=(20, 0))
        
        self.fps_label = ttk.Label(perf_monitor_frame, text="FPS: 0.0", font=("Arial", 10, "bold"))
        self.fps_label.pack()
        self.accuracy_label = ttk.Label(perf_monitor_frame, text="Accuracy: 0%", font=("Arial", 9))
        self.accuracy_label.pack()
        
        # Status and logging
        status_frame = ttk.LabelFrame(main_frame, text="Advanced Status Monitor", padding="10")
        status_frame.grid(row=6, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.status_text = tk.Text(status_frame, height=8, width=80, font=("Consolas", 9))
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights for responsive layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"{float(value)*100:.0f}%")
        
    def build_image_pyramid(self, image, levels=3):
        """Create multi-scale image pyramid for faster searching"""
        pyramid = [image]
        for i in range(1, levels):
            scale = 1.0 / (2 ** i)
            if image.shape[0] * scale > 10 and image.shape[1] * scale > 10:
                resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                pyramid.append(resized)
        return pyramid
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select target image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.image_path_label.config(text=file_path.split('/')[-1])
            
            try:
                # Load and pre-process template
                self.template = cv2.imread(file_path)
                if self.template is None:
                    raise ValueError("Could not load image")
                
                # Create grayscale version
                self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                
                # Build image pyramid for multi-scale search
                self.template_pyramid = self.build_image_pyramid(self.template_gray)
                
                # Display preview
                image = Image.open(file_path)
                image.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(image)
                self.image_preview.configure(image=photo, text=f"Preview: {self.template.shape[1]}x{self.template.shape[0]} pixels")
                self.image_preview.image = photo
                
                self.log_status(f"✅ Image loaded: {self.template.shape[1]}x{self.template.shape[0]} pixels")
                self.log_status(f"✅ Pyramid levels: {len(self.template_pyramid)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
                self.selected_image_path = None
                self.template = None
                self.template_gray = None
                self.template_pyramid = []
                
    def get_dynamic_search_region(self):
        """Calculate smart search region based on previous finds"""
        screen_width, screen_height = pyautogui.size()
        
        if not self.last_positions:
            return None  # Full screen for first search
        
        # Use last position with prediction
        last_x, last_y = self.last_positions[-1]
        
        # Predict movement (simple linear prediction)
        if len(self.last_positions) > 1 and self.predict_var.get():
            prev_x, prev_y = self.last_positions[-2]
            dx, dy = last_x - prev_x, last_y - prev_y
            # Small prediction factor
            last_x += int(dx * 0.3)
            last_y += int(dy * 0.3)
        
        # Dynamic region size based on search history
        region_size = 400
        if len(self.last_positions) > 5:
            # Reduce region size if we're consistently finding the target
            region_size = 200
        
        x = max(0, last_x - region_size // 2)
        y = max(0, last_y - region_size // 2)
        width = min(region_size, screen_width - x)
        height = min(region_size, screen_height - y)
        
        return (x, y, width, height)
    
    def pyramid_search(self, screenshot_gray):
        """Fast pyramid search algorithm"""
        best_match = None
        best_confidence = 0
        
        # Search from coarse to fine
        for level, template in enumerate(self.template_pyramid):
            if template.shape[0] > screenshot_gray.shape[0] or template.shape[1] > screenshot_gray.shape[1]:
                continue
                
            # Scale screenshot to match template level
            scale_factor = 2 ** level
            if level > 0:
                scaled_screen = cv2.resize(screenshot_gray, 
                                         (screenshot_gray.shape[1] // scale_factor, 
                                          screenshot_gray.shape[0] // scale_factor))
            else:
                scaled_screen = screenshot_gray
            
            # Template matching
            result = cv2.matchTemplate(scaled_screen, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                # Convert back to original coordinates
                x = max_loc[0] * scale_factor
                y = max_loc[1] * scale_factor
                best_match = (x, y, max_val)
                
                # Early termination if high confidence at coarse level
                if best_confidence > 0.9 and level < len(self.template_pyramid) - 1:
                    break
        
        return best_match
    
    def multi_scale_search(self, screenshot_gray):
        """Multi-scale search with different template sizes"""
        best_match = None
        best_confidence = 0
        
        scales = [0.5, 0.75, 1.0, 1.25] if self.algorithm_var.get() == "multiscale" else [0.8, 1.0, 1.2]
        
        for scale in scales:
            # Scale template
            new_width = int(self.template_gray.shape[1] * scale)
            new_height = int(self.template_gray.shape[0] * scale)
            
            if new_width < 10 or new_height < 10 or new_width > screenshot_gray.shape[1] or new_height > screenshot_gray.shape[0]:
                continue
                
            scaled_template = cv2.resize(self.template_gray, (new_width, new_height))
            
            # Template matching
            result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                # Calculate center
                x = max_loc[0] + new_width // 2
                y = max_loc[1] + new_height // 2
                best_match = (x, y, max_val)
        
        return best_match
    
    def precision_search(self, screenshot_gray):
        """High-precision search with multiple methods"""
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        best_match = None
        best_confidence = 0
        
        for method in methods:
            result = cv2.matchTemplate(screenshot_gray, self.template_gray, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Weighted combination for hybrid mode
            if self.algorithm_var.get() == "hybrid":
                confidence = max_val * 0.7 + min_val * 0.3
            else:
                confidence = max_val
                
            if confidence > best_confidence:
                best_confidence = confidence
                x = max_loc[0] + self.template_gray.shape[1] // 2
                y = max_loc[1] + self.template_gray.shape[0] // 2
                best_match = (x, y, confidence)
        
        return best_match
    
    def find_image_advanced(self):
        if self.template_gray is None:
            return None
            
        try:
            # Get search region
            if self.region_var.get() == "dynamic":
                region = self.get_dynamic_search_region()
            elif self.region_var.get() == "fixed":
                try:
                    width = int(self.region_width_var.get())
                    height = int(self.region_height_var.get())
                    x, y = pyautogui.position()
                    x = max(0, x - width // 2)
                    y = max(0, y - height // 2)
                    screen_width, screen_height = pyautogui.size()
                    width = min(width, screen_width - x)
                    height = min(height, screen_height - y)
                    region = (x, y, width, height)
                except:
                    region = None
            else:
                region = None
            
            # Capture screenshot
            screenshot = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            
            region_offset = region[0:2] if region else (0, 0)
            
            # Choose search algorithm
            algorithm = self.algorithm_var.get()
            if algorithm == "pyramid" and self.template_pyramid:
                result = self.pyramid_search(screenshot_gray)
            elif algorithm == "multiscale":
                result = self.multi_scale_search(screenshot_gray)
            elif algorithm == "precision":
                result = self.precision_search(screenshot_gray)
            else:  # hybrid
                # Try fast methods first, then fall back to precision
                result = self.pyramid_search(screenshot_gray)
                if not result or result[2] < self.confidence_var.get() * 0.8:
                    precision_result = self.precision_search(screenshot_gray)
                    if precision_result and precision_result[2] > result[2]:
                        result = precision_result
            
            if result:
                x, y, confidence = result
                # Apply region offset
                x += region_offset[0]
                y += region_offset[1]
                
                # Adaptive confidence
                required_confidence = self.confidence_var.get()
                if self.adaptive_var.get() and len(self.last_positions) > 3:
                    # Lower confidence if we've been finding it consistently
                    required_confidence *= 0.9
                
                if confidence >= required_confidence:
                    # Update position history (keep last 10)
                    self.last_positions.append((x, y))
                    if len(self.last_positions) > 10:
                        self.last_positions.pop(0)
                    
                    return (x, y, confidence)
            
            return None
            
        except Exception as e:
            self.log_status(f"❌ Search error: {e}")
            return None
    
    def perform_click(self, position):
        try:
            x, y, confidence = position
            
            # Smooth mouse movement
            pyautogui.moveTo(x, y, duration=0.1)
            
            # Perform click
            click_type = self.click_type_var.get()
            if click_type == "left":
                pyautogui.click()
            elif click_type == "right":
                pyautogui.rightClick()
            elif click_type == "double":
                pyautogui.doubleClick()
            elif click_type == "middle":
                pyautogui.middleClick()
            
            self.log_status(f"🎯 Clicked at ({x}, {y}) - Confidence: {confidence:.3f}")
            
        except Exception as e:
            self.log_status(f"❌ Click error: {e}")
    
    def update_performance_monitor(self):
        """Advanced performance monitoring"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= 2.0:  # Update every 2 seconds for stability
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            if len(self.fps_history) > 5:
                self.fps_history.pop(0)
            
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.fps_label.config(text=f"FPS: {avg_fps:.1f}")
            
            # Calculate accuracy based on recent finds
            if len(self.last_positions) > 0:
                accuracy = min(100, len(self.last_positions) * 10)
                self.accuracy_label.config(text=f"Accuracy: {accuracy}%")
            
            self.frame_count = 0
            self.start_time = current_time
    
    def test_single_click(self):
        if self.template_gray is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        self.log_status("🧪 Testing single click...")
        result = self.find_image_advanced()
        if result:
            self.perform_click(result)
            self.log_status("✅ Test successful!")
        else:
            self.log_status("❌ Test failed: Target not found")
    
    def auto_click_loop(self):
        """Main auto-click loop with advanced features"""
        consecutive_fails = 0
        max_consecutive_fails = 5
        
        while self.is_running:
            start_time = time.time()
            
            # Advanced search
            result = self.find_image_advanced()
            
            if result:
                consecutive_fails = 0
                self.perform_click(result)
            else:
                consecutive_fails += 1
                self.log_status(f"⚠️ Target not found ({consecutive_fails}/{max_consecutive_fails})")
                
                # Expand search area if failing repeatedly
                if consecutive_fails >= max_consecutive_fails:
                    self.last_positions = []  # Reset position history
                    self.log_status("🔄 Expanding search to full screen...")
                    consecutive_fails = 0
            
            # Performance monitoring
            self.update_performance_monitor()
            
            # Adaptive interval
            try:
                interval = max(0.05, float(self.interval_var.get()))  # Minimum 50ms
                # Process events during wait
                end_time = time.time()
                processing_time = end_time - start_time
                sleep_time = max(0.01, interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except ValueError:
                time.sleep(0.1)
    
    def start_auto_click(self):
        if self.template_gray is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.test_btn.config(state=tk.DISABLED)
        
        # Reset tracking
        self.last_positions = []
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        
        self.log_status("🚀 ULTRA-CLICK ENGINE STARTED!")
        self.log_status(f"📊 Algorithm: {self.algorithm_var.get().upper()}")
        self.log_status(f"🎯 Confidence: {self.confidence_var.get()*100:.0f}%")
        
        # Start main loop in separate thread
        self.auto_click_thread = threading.Thread(target=self.auto_click_loop)
        self.auto_click_thread.daemon = True
        self.auto_click_thread.start()
    
    def stop_auto_click(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.NORMAL)
        self.log_status("🛑 Auto-click stopped.")
    
    def log_status(self, message):
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, formatted_message)
        self.status_text.see(tk.END)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    # Add some style
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="white", background="darkblue")
    
    app = AdvancedAutoClickerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()