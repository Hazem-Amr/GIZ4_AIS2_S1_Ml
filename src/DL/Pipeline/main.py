from src.training_pipeline import TrainingPipeline
from gui import main as run_gui
import sys

class Main:
    def start(self):
        """Start the training pipeline"""
        pipeline = TrainingPipeline()
        results = pipeline.run()
        return results

if __name__ == "__main__":
    # Check for --gui flag to run GUI mode
    if "--gui" in sys.argv or "-g" in sys.argv:
        print("Starting GUI mode...")
        run_gui()
    else:
        # Run in command-line mode
        main = Main()
        main.start()
