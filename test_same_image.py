from deepface import DeepFace
import numpy as np

img = "faces/Fayyaz/fayyaz.jpeg"  # same image, twice

e1 = DeepFace.represent(
    img_path=img,
    model_name="Facenet",
    detector_backend="opencv",
    enforce_detection=True
)[0]["embedding"]

e2 = DeepFace.represent(
    img_path=img,
    model_name="Facenet",
    detector_backend="opencv",
    enforce_detection=True
)[0]["embedding"]

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Cosine distance (same image):", cosine_distance(e1, e2))