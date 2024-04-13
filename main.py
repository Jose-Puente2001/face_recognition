import tkinter as tk
import cv2
from PIL import Image, ImageTk

window = tk.Tk()
window.title("Reconocimiento Facial para Control de Asistencia - Alcaldía Muncipio Carirubana")
window.geometry("800x600")

canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
current_image = None

def capture_image():
    global current_image
    ret, frame = video_capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    image_pil = Image.fromarray(image)
    current_image = ImageTk.PhotoImage(image_pil)

    canvas.create_image(0, 0, anchor=tk.NW, image=current_image)

    window.update()

    window.after(10, capture_image)


def turn_on():
    global video_capture
    capture_image()


on_button = tk.Button(window, text="Encender", command=turn_on)
on_button.pack()

window.mainloop()