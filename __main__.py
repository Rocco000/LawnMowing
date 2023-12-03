import gymnasium as gym
import gym_game
import pygame


if __name__ == "__main__":
    env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
    print("ciao")
    env.reset()
    print("dopo reset")
    running = True
    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()