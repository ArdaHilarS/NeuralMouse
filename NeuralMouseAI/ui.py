
import pygame
from settings import *

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont("Arial", 32, bold=True)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        color = BLUE
        if self.rect.collidepoint(mouse_pos):
            color = GREEN

        pygame.draw.rect(screen, color, self.rect, border_radius=15)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=15)

        text_surface = self.font.render(self.text, True, WHITE)
        screen.blit(
            text_surface,
            (
                self.rect.x + self.rect.width // 2 - text_surface.get_width() // 2,
                self.rect.y + self.rect.height // 2 - text_surface.get_height() // 2
            )
        )

    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False