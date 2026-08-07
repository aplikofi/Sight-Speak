import pygame
import pygame.camera
import sys

pygame.init()
pygame.camera.init()
cameras = pygame.camera.list_cameras()
print("Available cameras:", cameras)

if cameras:
    cam = pygame.camera.Camera(cameras[0], (640, 480))
    cam.start()
    img = cam.get_image()
    print("Image captured successfully. Size:", img.get_size())
    cam.stop()
else:
    print("No cameras found.")
pygame.quit()
