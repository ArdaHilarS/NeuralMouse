
import pygame
from settings import *

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont("Arial", 32, bold=True)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = self.rect.collidepoint(mouse_pos)
        
        bg_color = (40, 15, 60) if is_hovered else (25, 10, 40)
        border_color = NEON_CYAN if is_hovered else NEON_PINK
        text_color = WHITE if is_hovered else NEON_CYAN

        pygame.draw.rect(screen, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=4)

        if is_hovered:
            glow_rect = self.rect.inflate(4, 4)
            pygame.draw.rect(screen, NEON_CYAN, glow_rect, 1, border_radius=6)

        text_surface = self.font.render(self.text, True, text_color)
        text_x = self.rect.x + (self.rect.width - text_surface.get_width()) // 2
        text_y = self.rect.y + (self.rect.height - text_surface.get_height()) // 2
        screen.blit(text_surface, (text_x, text_y))

    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False