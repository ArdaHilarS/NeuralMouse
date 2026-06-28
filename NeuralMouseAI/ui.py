import pygame
from settings import *

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        font_size = 24 if len(text) > 15 else 28
        self.font = pygame.font.SysFont("Segoe UI", font_size, bold=True)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        
        color = RETRO_ORANGE if self.rect.collidepoint(mouse_pos) else RETRO_CYAN

        pygame.draw.rect(screen, BLACK, self.rect, border_radius=4)
        pygame.draw.rect(screen, color, self.rect, width=2, border_radius=4)

        text_surface = self.font.render(self.text, True, color)
        
        text_x = self.rect.centerx - text_surface.get_width() // 2
        text_y = self.rect.centery - text_surface.get_height() // 2
        screen.blit(text_surface, (text_x, text_y))

    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False