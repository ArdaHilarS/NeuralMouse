
import pygame
from settings import *

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.image.load("NeuralMouseAI/assets/Images/mouse.png")
        self.image = pygame.transform.scale(self.image, (32, 32))

    def move(self, dx, dy, maze):
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < COLS and 0 <= new_y < ROWS:
            if maze[new_y][new_x] != "#":
                self.x = new_x
                self.y = new_y

    def draw(self, screen, offset_x, offset_y):
        screen.blit(self.image, (offset_x + self.x * TILE_SIZE + 3, offset_y + self.y * TILE_SIZE + 3))