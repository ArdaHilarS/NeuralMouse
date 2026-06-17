import pygame
from settings import *

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont("Arial", 28, bold=True)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        
        color = NEON_CYAN
        if self.rect.collidepoint(mouse_pos):
            color = NEON_PINK

        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect.inflate(-4, -4), border_radius=6)

        text_surface = self.font.render(self.text, True, color)
        screen.blit(
            text_surface,
            (
                self.rect.centerx - text_surface.get_width() // 2,
                self.rect.centery - text_surface.get_height() // 2
            )
        )

    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False