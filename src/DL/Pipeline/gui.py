import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw
import io
import threading
from tensorflow.keras.models import load_model
from src.training_pipeline import TrainingPipeline


class DrawingPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Drawing Prediction")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f0f0")
        
        self.model = None
        self.model_loaded = False
        self.is_predicting = False
        
        # Create PIL image for drawing
        self.image = Image.new('L', (280, 280), color=255)  # White background
        self.draw = ImageDraw.Draw(self.image)
        
        self.create_widgets()
        self.load_model()
    
    def create_widgets(self):
        """Create GUI widgets"""
        
        # Title
        title_label = tk.Label(
            self.root,
            text="Draw a Digit (0-9)",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        title_label.pack(pady=10)
        
        # Canvas for drawing
        canvas_frame = tk.Frame(self.root, bg="white", relief="sunken", bd=2)
        canvas_frame.pack(pady=10, padx=20)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=280,
            height=280,
            bg="white",
            cursor="cross"
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Button Frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        self.predict_btn = ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict
        )
        self.predict_btn.pack(side="left", padx=5)
        
        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas
        )
        self.clear_btn.pack(side="left", padx=5)
        
        # Results Frame
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results", padding=15)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Prediction Label
        ttk.Label(results_frame, text="Predicted Digit:").pack(anchor="w", pady=5)
        self.prediction_label = tk.Label(
            results_frame,
            text="---",
            font=("Arial", 48, "bold"),
            bg="#ffffff",
            fg="#2196F3",
            relief="sunken",
            padx=20,
            pady=10
        )
        self.prediction_label.pack(fill="x", pady=5)
        
        # Confidence
        ttk.Label(results_frame, text="Confidence:").pack(anchor="w", pady=5)
        self.confidence_bar = ttk.Progressbar(
            results_frame,
            length=400,
            mode='determinate'
        )
        self.confidence_bar.pack(fill="x", pady=5)
        
        self.confidence_label = tk.Label(
            results_frame,
            text="0%",
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        self.confidence_label.pack(anchor="w", pady=2)
        
        # Status Label
        self.status_label = ttk.Label(
            self.root,
            text="Ready",
            relief="sunken",
            anchor="w"
        )
        self.status_label.pack(fill="x", padx=20, pady=5)
    
    def on_canvas_press(self, event):
        """Handle mouse press on canvas"""
        self.last_x = event.x
        self.last_y = event.y
    
    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas"""
        # Draw on PIL image
        self.draw.line(
            [(self.last_x, self.last_y), (event.x, event.y)],
            fill=0,  # Black color
            width=15  # Brush size
        )
        
        # Draw on tkinter canvas
        self.canvas.create_line(
            self.last_x, self.last_y,
            event.x, event.y,
            fill="black",
            width=15,
            capstyle="round",
            smooth=True
        )
        
        self.last_x = event.x
        self.last_y = event.y
    
    def on_canvas_release(self, event):
        """Handle mouse release on canvas"""
        pass
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="---")
        self.confidence_bar.config(value=0)
        self.confidence_label.config(text="0%")
        self.status_label.config(text="Canvas cleared")
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.status_label.config(text="Loading model...")
            self.root.update()
            
            # Try to load existing model, otherwise train new one
            try:
                self.model = load_model('mnist_model.h5')
                self.model_loaded = True
                self.status_label.config(text="Model loaded successfully")
                messagebox.showinfo("Info", "Model loaded successfully!")
            except:
                self.status_label.config(text="Training new model (this may take a while)...")
                messagebox.showinfo(
                    "Info",
                    "No pre-trained model found. Training a new model...\n\nThis may take a few minutes."
                )
                
                # Train model in background thread
                train_thread = threading.Thread(target=self.train_model, daemon=True)
                train_thread.start()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.status_label.config(text="Error loading model")
    
    def train_model(self):
        """Train the model"""
        try:
            pipeline = TrainingPipeline()
            results = pipeline.run()
            
            # Save the model
            results['model'].save('mnist_model.h5')
            self.model = results['model']
            self.model_loaded = True
            self.status_label.config(text="Model trained and saved successfully")
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            self.status_label.config(text="Error training model")
    
    def preprocess_image(self):
        """Preprocess the drawn image for prediction"""
        # Resize to 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype='float32')
        
        # Invert colors (MNIST uses black digits on white background, but we draw white on black)
        img_array = 255 - img_array
        
        # Normalize to 0-1
        img_array = img_array / 255.0
        
        # Flatten for model input
        img_array = img_array.flatten().reshape(1, -1)
        
        return img_array
    
    def predict(self):
        """Make prediction on drawn digit"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Model is still loading. Please wait...")
            return
        
        if self.is_predicting:
            messagebox.showwarning("Warning", "Prediction in progress...")
            return
        
        try:
            self.is_predicting = True
            self.predict_btn.config(state="disabled")
            self.status_label.config(text="Making prediction...")
            self.root.update()
            
            # Preprocess image
            processed_image = self.preprocess_image()
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_digit]) * 100
            
            # Update results
            self.prediction_label.config(text=str(predicted_digit))
            self.confidence_bar.config(value=confidence)
            self.confidence_label.config(text=f"{confidence:.1f}%")
            
            self.status_label.config(
                text=f"Prediction: {predicted_digit} (Confidence: {confidence:.1f}%)"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")
            self.status_label.config(text="Error during prediction")
        
        finally:
            self.is_predicting = False
            self.predict_btn.config(state="normal")


def main():
    """Run the GUI application"""
    root = tk.Tk()
    gui = DrawingPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
