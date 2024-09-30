import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import keras
import threading
import time




class ContinuousPredictionThread(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
    
    def stopped(self):
        return self._stop_event.is_set()
    
    def run(self):
        while not self.stopped():
            drawn_image = self.app.capture_canvas()
            preprocessed_image = self.app.preprocess_image(drawn_image)
            predicted_categories = self.app.predict_image(preprocessed_image)
            self.app.update_prediction(predicted_categories)
            time.sleep(1)






class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")

        # Set up the canvas for drawing
        self.canvas = Canvas(master, width=400, height=400, bg="black")
        self.canvas.grid(row=0, columnspan=2)

        self.canvaspredict = Canvas(master, width=200, height=200, bg="white")
        self.canvaspredict.grid(row=1, columnspan=2)

        self.predict_label = Label(master, text="", font=("Helvetica", 24, "bold"), padx=10, pady=10)
        self.predict_label.grid(row=4, columnspan=2, sticky="ew")


        self.clear_btn = Button(master, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=2, column=0, sticky="ew")

        self.start_btn = Button(master, text="Start Predcition", command=self.start_prediction)
        self.start_btn.grid(row=2, column=1, sticky="ew")

        self.stop_btn = Button(master, text="Stop Prediction", command=self.stop_prediction, state="disabled")
        self.stop_btn.grid(row=3, column=1, sticky="ew")

        self.image = Image.new("L", (400, 400), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.prediction_thread = None
        self.predicting = False


        csv_file_path = "C:/Users/tacoc/Desktop/Data/X_train/train_data.txt"

        image_data = np.loadtxt(csv_file_path, delimiter=',')  

        self.X_train = image_data.reshape(-1, 28, 28, 1)



        labels_path = "C:/Users/tacoc/Desktop/Data/y_train/train_label.txt"
        with open(labels_path, 'r') as f:
            labels = [np.array(line.split(','), dtype=float) for line in f]   
        self.y_train = np.array(labels)

     


        self.build_model()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvaspredict.delete("all")
        self.image = Image.new("L", (400, 400), "black")
        self.draw = ImageDraw.Draw(self.image)

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(5, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=4, validation_split=0.1)

    def start_prediction(self):
        if not self.predicting:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.predicting = True
            self.prediction_thread = ContinuousPredictionThread(self)
            self.prediction_thread.start()
    
    def stop_prediction(self):
        if self.predicting:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.predicting = False
            self.prediction_thread.stop()
    
    def capture_canvas(self):
        if self.predicting:
            drawn_image = self.image.copy()
            return drawn_image
    
    def preprocess_image(self, drawn_image):
        resized_image = drawn_image.resize((28, 28))
        numpy_image = np.array(resized_image) / 255.0
        return numpy_image.reshape(1, 28, 28, 1)
    
    def predict_image(self, preprocessed_image):
        predictied_categories = self.model.predict(preprocessed_image)[0]
        return predictied_categories
    
    def update_prediction(self, predicted_categories):
        categories = ["Car", "Boat", "Tree", "Face", "Shirt"]
        self.canvaspredict.delete("all")
        for i, prob in enumerate(predicted_categories):
            x1 = 10
            y1 = i * 20
            x2 = int(prob * 100) + 10
            y2 = (i + 1) * 20
            color = "#{:02x}0000".format(int(prob * 255))
            self.canvaspredict.create_rectangle(x1, y1, x2, y2, fill=color)
            self.canvaspredict.create_text(120, y1 + 10, text=f"{categories[i]}: {prob:.4f}")
        prediction_index = np.argmax(predicted_categories)
        predict = categories[prediction_index]
        self.predict_label.config(text=predict)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_line(x1, y1, x2, y2, capstyle="round", fill="white", width=20)
        self.draw.line([x1, y1, x2, y2], fill="white", width=20)

def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.bind("<B1-Motion>", app.paint)
    root.mainloop()

if __name__ == "__main__":
    main()
