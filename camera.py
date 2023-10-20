import cv2
import time
import datetime

# Inicjalizacja obiektu przechwytującego wideo dla domyślnej kamery (0)
cap = cv2.VideoCapture(0)

# Ładowanie klasyfikatorów Haara do detekcji twarzy i pełnych sylwetek
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

while True:
    # Odczytanie klatki z kamery
    _, frame = cap.read()

    # Konwersja klatki na odcienie szarości do detekcji twarzy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy w klatce w odcieniach szarości
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Wykrywanie pełnych sylwetek w klatce w odcieniach szarości
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)


    # Rysowanie prostokątów wokół wykrytych twarzy
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    # Rysowanie prostokątów wokół wykrytych pełnych sylwetek
    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)

    # Wyświetlenie klatki z prostokątami
    cv2.imshow("Kamera", frame)

    # Jeśli naciśniesz klawisz 'q', to wyjdziesz z pętli
    if cv2.waitKey(1) == ord('q'):
        break

# Zwolnienie kamery i zamknięcie wszystkich okien OpenCV
cap.release()
cv2.destroyAllWindows()
