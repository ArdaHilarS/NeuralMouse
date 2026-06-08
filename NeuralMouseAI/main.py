
import pygame
import sys
import random
import heapq
import numpy as np
import os

from settings import *
from maze import maze
from player import Player
from ui import Button
from dqn_agent import DQN_Agent

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Mouse AI")

clock = pygame.time.Clock()

big_font = pygame.font.SysFont("Arial", 70, bold=True)
small_font = pygame.font.SysFont("Arial", 28)

cheese_img = pygame.image.load("NeuralMouseAI/assets/Images/cheese.png")
cheese_img = pygame.transform.scale(cheese_img, (34, 34))

cat_img = pygame.image.load("NeuralMouseAI/assets/Images/cat2.png")
cat_img = pygame.transform.scale(cat_img, (34, 34))

background = pygame.image.load("NeuralMouseAI/assets/Images/background.png")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

tree_img = pygame.image.load("NeuralMouseAI/assets/Images/tree.png")
tree_img = pygame.transform.scale(tree_img, (TILE_SIZE, TILE_SIZE))

start_button = Button((WIDTH - 260) // 2, 300, 260, 80, "PLAY")
ai_button = Button((WIDTH - 260) // 2, 410, 260, 80, "TRAIN AI")
ai_play_default = Button((WIDTH - 260) // 2, 520, 260, 80, "AI PLAYS")
quit_button = Button((WIDTH - 260) // 2, 630, 260, 80, "QUIT")
resume_button = Button((900 - 260) // 2, 320, 260, 80, "RESUME") 
restart_button = Button((WIDTH - 260) // 2, 420, 260, 80, "RESTART")

agent = DQN_Agent(state=10, action=4)

MENU = 0
GAME = 1
WIN = 2
LOSE = 3
PAUSE = 4
AI = 5
AI_PLAY = 6

state = MENU

game_time_ms = 0
play_history = []

player = None
cheese_pos = None
cats = []

valid_positions = []
for y, row in enumerate(maze):
    for x, tile in enumerate(row):
        if tile == " ":
            valid_positions.append((x, y))

CAT_MOVE_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(CAT_MOVE_EVENT, 500)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            nx, ny = neighbor[0], neighbor[1]

            if nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS:
                continue

            if maze[ny][nx] == "#":
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def reset_game():
    global player, cheese_pos, cats, game_time_ms, play_history
    game_time_ms = 0
    play_history = []
    
    positions = random.sample(valid_positions, 3)
    
    player = Player(positions[0][0], positions[0][1])
    cheese_pos = positions[1]
    cats = [[positions[2][0], positions[2][1]]]

def get_state(player, cats, cheese_pos, maze):
    up_wall = 1 if player.y - 1 < 0 or maze[player.y - 1][player.x] == "#" else 0
    down_wall = 1 if player.y + 1 >= ROWS or maze[player.y + 1][player.x] == "#" else 0
    left_wall = 1 if player.x - 1 < 0 or maze[player.y][player.x - 1] == "#" else 0
    right_wall = 1 if player.x + 1 >= COLS or maze[player.y][player.x + 1] == "#" else 0
    
    return [
        player.x / COLS,
        player.y / ROWS,
        cats[0][0] / COLS,
        cats[0][1] / ROWS,
        cheese_pos[0] / COLS,
        cheese_pos[1] / ROWS,
        up_wall, down_wall, left_wall, right_wall
    ]

reset_game()

while True:
    dt = clock.tick(FPS)

    if state == GAME:
        game_time_ms += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if state == MENU:
            if start_button.clicked(event):
                reset_game()
                state = GAME
            if ai_button.clicked(event):
                state = AI
            if quit_button.clicked(event):
                pygame.quit()
                sys.exit()
            if ai_play_default.clicked(event):
                reset_game()
                if agent.load("mouse_ai.pth"):
                    agent.epsilon = 0
                else:
                    agent.epsilon = 0.1
                state = AI_PLAY

        elif state == AI:
            TOTAL_EPISODES = 3000
            for episode in range(TOTAL_EPISODES):
                reset_game()
                current_state = get_state(player, cats, cheese_pos, maze)
                done = False
                step_count = 0
                
                while not done and step_count < 300:
                    step_count += 1
                    action = agent.choose_action(current_state)
                    old_x, old_y = player.x, player.y

                    if action == 0: player.move(0, -1, maze)
                    elif action == 1: player.move(0, 1, maze)
                    elif action == 2: player.move(-1, 0, maze)
                    elif action == 3: player.move(1, 0, maze)
                    
                    reward = -0.1
                    
                    if player.x == old_x and player.y == old_y:
                        reward -= 0.5

                    old_distance = abs(old_x - cheese_pos[0]) + abs(old_y - cheese_pos[1])
                    new_distance = abs(player.x - cheese_pos[0]) + abs(player.y - cheese_pos[1])

                    if new_distance < old_distance:
                        reward += 0.2
                    else:
                        reward -= 0.2

                    if (player.x, player.y) == (cats[0][0], cats[0][1]):
                        reward = -10.0
                        done = True
                    
                    if (player.x, player.y) == cheese_pos:
                        reward = 20.0
                        done = True

                    if not done and step_count % 2 == 0:
                        start = (cats[0][0], cats[0][1])
                        target = (player.x, player.y)
                        path = astar(start, target)
                        if len(path) > 0:
                            cats[0][0], cats[0][1] = path[0]
                            if (player.x, player.y) == (cats[0][0], cats[0][1]):
                                reward = -10.0
                                done = True

                    next_state = get_state(player, cats, cheese_pos, maze)                    
                    agent.store_transition(current_state, action, reward, next_state, done)
                    agent.replay()
                    current_state = next_state

                if episode % 5 == 0:
                    agent.update_target_network()

                agent.epsilon = max(
                    agent.epsilon_min,
                    agent.epsilon * agent.epsilon_reduce
                )

                if episode % 10 == 0:
                    percent = round(episode / TOTAL_EPISODES * 100)
                    screen.blit(background, (0, 0)) 
                    
                    title = big_font.render("AI LEARNING...", True, (0, 150, 255))
                    screen.blit(title, (WIDTH // 2 - (title.get_width() // 2), 200))
                    
                    info = small_font.render(f"Progress: {percent}%  |  Epsilon: {agent.epsilon:.3f}", True, WHITE)
                    screen.blit(info, (WIDTH // 2 - (info.get_width() // 2), 300))
                    
                    pygame.display.update()
                    
            agent.save("mouse_ai.pth")
            reset_game()
            agent.load("mouse_ai.pth")                    
            agent.epsilon = 0                    
            state = AI_PLAY           

        elif state == GAME:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: player.move(0, -1, maze)
                if event.key == pygame.K_DOWN: player.move(0, 1, maze)
                if event.key == pygame.K_LEFT: player.move(-1, 0, maze)
                if event.key == pygame.K_RIGHT: player.move(1, 0, maze)
                if event.key == pygame.K_ESCAPE: state = PAUSE

            if event.type == CAT_MOVE_EVENT and player is not None:
                for cat in cats:
                    start = (cat[0], cat[1])
                    target = (player.x, player.y)
                    path = astar(start, target)
                    if len(path) > 0:
                        cat[0], cat[1] = path[0]

        elif state == PAUSE:
            if resume_button.clicked(event): state = GAME
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: state = GAME

        elif state == WIN or state == LOSE:
            if restart_button.clicked(event):
                reset_game()
                state = GAME

    if state == AI_PLAY:
        current_state = get_state(player, cats, cheese_pos, maze)
        
        if len(play_history) >= 4 and play_history[-1] == play_history[-3] and play_history[-2] == play_history[-4]:
            action = random.randint(0, 3) 
            play_history.clear()
        else:
            action = agent.choose_action(current_state)

        old_pos = (player.x, player.y)

        if action == 0: player.move(0, -1, maze)
        elif action == 1: player.move(0, 1, maze)
        elif action == 2: player.move(-1, 0, maze)
        elif action == 3: player.move(1, 0, maze)
        
        if (player.x, player.y) != old_pos:
            play_history.append((player.x, player.y))
            if len(play_history) > 10:
                play_history.pop(0)
       
        if pygame.time.get_ticks() % 2 == 0:
            for cat in cats:
                start = (cat[0], cat[1])
                target = (player.x, player.y)
                path = astar(start, target)
                if len(path) > 0:
                    cat[0], cat[1] = path[0]
       
        pygame.time.delay(80)

    if state == GAME or state == AI_PLAY:
        if (player.x, player.y) == cheese_pos:
            if state == AI_PLAY: 
                reset_game()
            else: 
                state = WIN
                
        for cat in cats:
            if (player.x, player.y) == (cat[0], cat[1]):
                if state == AI_PLAY: 
                    reset_game()
                else: 
                    state = LOSE

    screen.blit(background, (0, 0))

    if state == MENU:
        title = big_font.render("NEURAL MOUSE", True, WHITE)
        title_rect = title.get_rect(centerx=WIDTH/2, top=120)
        screen.blit(title, title_rect)
        
        info = small_font.render("AI learns to solve the maze", True, WHITE)
        info_rect = info.get_rect(centerx=WIDTH/2, top=220)
        screen.blit(info, info_rect)

        start_button.draw(screen)
        ai_button.draw(screen)
        ai_play_default.draw(screen)
        quit_button.draw(screen)

    elif state == GAME or state == PAUSE or state == AI_PLAY:
        panel_width = 300
        game_area_width = WIDTH - panel_width
        maze_width = COLS * TILE_SIZE
        maze_height = ROWS * TILE_SIZE

        maze_offset_x = (game_area_width - maze_width) // 2
        maze_offset_y = (HEIGHT - maze_height) // 2

        for y, row in enumerate(maze):
            for x, tile in enumerate(row):
                rect = pygame.Rect(
                    maze_offset_x + x * TILE_SIZE,
                    maze_offset_y + y * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE
                )
                if tile == "#":
                    screen.blit(tree_img, rect)     
                else:
                    pygame.draw.rect(screen, PATH_COLOR, rect, border_radius=4)

        screen.blit(cheese_img, (maze_offset_x + cheese_pos[0] * TILE_SIZE + 2, maze_offset_y + cheese_pos[1] * TILE_SIZE + 2))

        for cat in cats:
            screen.blit(cat_img, (maze_offset_x + cat[0] * TILE_SIZE + 2, maze_offset_y + cat[1] * TILE_SIZE + 2))

        if player is not None:
            player.draw(screen, maze_offset_x, maze_offset_y)

        pygame.draw.rect(screen, PANEL_COLOR, (WIDTH - panel_width, 0, panel_width, HEIGHT))
        pygame.draw.line(screen, OUTLINE_COLOR, (WIDTH - panel_width, 0), (WIDTH - panel_width, HEIGHT), 3)

        panel_title = big_font.render("INFO", True, WHITE)
        title_x = WIDTH - panel_width + (panel_width - panel_title.get_width()) // 2
        screen.blit(panel_title, (title_x, 40))
        
        pygame.draw.line(screen, BLUE, (title_x - 15, 120), (title_x + panel_title.get_width() + 15, 120), 4)

        margin_left = WIDTH - panel_width + 40

        t1_icon = small_font.render("►", True, GREEN)
        t1_text = small_font.render("Move: Arrows", True, LIGHT_GRAY)
        screen.blit(t1_icon, (margin_left, 180))
        screen.blit(t1_text, (margin_left + 35, 180))

        t2_icon = small_font.render("★", True, YELLOW)
        t2_text = small_font.render("Goal: Cheese", True, LIGHT_GRAY)
        screen.blit(t2_icon, (margin_left, 240))
        screen.blit(t2_text, (margin_left + 35, 240))

        t3_icon = small_font.render("■", True, RED)
        t3_text = small_font.render("Pause: ESC", True, LIGHT_GRAY)
        screen.blit(t3_icon, (margin_left, 300))
        screen.blit(t3_text, (margin_left + 35, 300))

        pygame.draw.line(screen, OUTLINE_COLOR, (margin_left, 370), (WIDTH - 40, 370), 2)

        seconds = game_time_ms // 1000
        timer_label = small_font.render("TIME", True, BLUE)
        screen.blit(timer_label, (margin_left, 400))
        
        timer_val = big_font.render(f"{seconds}s", True, WHITE)
        screen.blit(timer_val, (margin_left, 440))

        if state == PAUSE:
            overlay = pygame.Surface((game_area_width, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 10, 15, 200)) 
            screen.blit(overlay, (0, 0))

            pause_title = big_font.render("PAUSED", True, YELLOW)
            pause_rect = pause_title.get_rect(centerx=game_area_width // 2, top=180)
            screen.blit(pause_title, pause_rect)
            resume_button.draw(screen)

    elif state == WIN:
        win_text = big_font.render("YOU WIN!", True, GREEN)
        win_rect = win_text.get_rect(centerx=WIDTH//2, top=150)
        screen.blit(win_text, win_rect)
        
        info = small_font.render("The mouse found the cheese!", True, WHITE)
        info_rect = info.get_rect(centerx=WIDTH//2, top=260)
        screen.blit(info, info_rect)
        
        final_seconds = game_time_ms // 1000
        time_info = small_font.render(f"Clear Time: {final_seconds}s", True, LIGHT_GRAY)
        time_rect = time_info.get_rect(centerx=WIDTH//2, top=320)
        screen.blit(time_info, time_rect)

        restart_button.draw(screen)

    elif state == LOSE:
        lose_text = big_font.render("YOU LOST!", True, RED)
        lose_rect = lose_text.get_rect(centerx=WIDTH//2, top=180)
        screen.blit(lose_text, lose_rect)
        
        info = small_font.render("The cat caught the mouse!", True, WHITE)
        info_rect = info.get_rect(centerx=WIDTH//2, top=290)
        screen.blit(info, info_rect)
        
        restart_button.draw(screen)

    pygame.display.update()