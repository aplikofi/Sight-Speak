import urllib.request
import os
import pygame
import pygame.camera
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download model
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print("Downloading model...")
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded.")

# 2. Setup detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 3. Capture image
pygame.init()
pygame.camera.init()
cameras = pygame.camera.list_cameras()
cam = pygame.camera.Camera(cameras[0], (640, 480))
cam.start()

# Warm up camera
for _ in range(5):
    cam.get_image()

img = cam.get_image()
cam.stop()

# 4. Process image
# Pygame surfarray returns (W, H, 3). MediaPipe needs (H, W, 3).
img_array = pygame.surfarray.pixels3d(img)
img_array = np.transpose(img_array, (1, 0, 2))
# Convert to contiguous array
img_array = np.ascontiguousarray(img_array)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
result = detector.detect(mp_image)

if result.face_landmarks:
    print(f"Detected face with {len(result.face_landmarks[0])} landmarks.")
else:
    print("No face detected.")

pygame.quit()
