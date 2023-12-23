import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
import pygame
import os

import numpy as np
import time
import threading

class LawnMowingEnvironment(Env):
    
    #Specify the render-modes 
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=8, weather_interval=60):
        self.size = size  # The size of the square grid
        self.state = np.random.randint(low=1, high=4, size=(size, size)) #initialize the grid
        
        #Initialize the agent points and penalties which it can reach during the game
        self.points = 0
        self.penalty = 0

        self._agent_location = np.array([0,0]) #agent's location

        self.num_obstacle = size-2
        self.obstacles_location = np.zeros((self.num_obstacle,2), dtype=int) #obstacles' location

        self.window_size = 512  # The size of the PyGame window

        #Load textures
        current_dir = os.path.dirname(os.path.abspath(__file__))
        gym_game_dir = os.path.dirname(current_dir)
        self.shaved_grass = pygame.image.load(os.path.join(gym_game_dir, 'texture','shaved_grass.png'))
        self.short_grass = pygame.image.load(os.path.join(gym_game_dir, 'texture','short_grass.png'))
        self.medium_grass = pygame.image.load(os.path.join(gym_game_dir, 'texture','medium_grass.png'))
        self.strong_grass = pygame.image.load(os.path.join(gym_game_dir, 'texture','strong_grass.png'))
        self.rock = pygame.image.load(os.path.join(gym_game_dir, 'texture','rock.png'))
        self.tree = pygame.image.load(os.path.join(gym_game_dir, 'texture','tree.png'))
        
        self.short_grass_lw = pygame.image.load(os.path.join(gym_game_dir, 'texture','shaved_grass_lw.png'))
        self.shaved_grass_lw = pygame.image.load(os.path.join(gym_game_dir, 'texture','short_grass_lw.png'))
        self.medium_grass_lw = pygame.image.load(os.path.join(gym_game_dir, 'texture','medium_grass_lw.png'))
        self.strong_grass_lw = pygame.image.load(os.path.join(gym_game_dir, 'texture','strong_grass_lw.png'))
        self.rock_lw = pygame.image.load(os.path.join(gym_game_dir, 'texture','rock_lw.png'))
        self.tree_lw = pygame.image.load(os.path.join(gym_game_dir, 'texture','tree_lw.png'))

        #Weather
        self.sunny = pygame.image.load(os.path.join(gym_game_dir, 'texture','house_sunny.png'))
        self.cloudy = pygame.image.load(os.path.join(gym_game_dir, 'texture','house_cloudy.png'))
        self.rainy = pygame.image.load(os.path.join(gym_game_dir, 'texture','house_rainy.png'))
        
        #Create thread to update weather and grass
        #self.weather_thread = threading.Thread(target=self.update_weather_and_grow)
        #self._stop_event = threading.Event()
        
        self.game_over = False

        #To count the number of steps
        self.step_counter = 0 

        # Observations are dictionaries with the agent's location and the position of its surrounding.
        # The agent's position is represented by a vector that contains the coordinate. The vector values are in the range 0-7
        self.observation_space = spaces.Box(low=0, high=11, shape=(size, size), dtype=int)

        """self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=size-1, shape=(2,), dtype=int),
                "grid": spaces.Box(low=0, high=11, shape=(size, size), dtype=int)
            }
        )"""

        # We have 5 actions, corresponding to "right", "up", "left", "down", "cut"
        self.action_space = spaces.Discrete(7)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
            5: np.array([0, 0]),
            6: np.array([0, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        #Weather
        self.weather_interval = weather_interval
        self.weather_probabilities = {'sunny': 0.4, 'cloudy': 0.3, 'rainy': 0.3}
        self.current_weather = "sunny"

    def render(self):
        #if self.render_mode == "rgb_array":
        return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        

        top_surface = pygame.Surface((self.window_size, 80))
        canvas = pygame.Surface((self.window_size, self.window_size-80))
        canvas.fill((255, 255, 255))
        top_surface.fill((255, 255, 255))
        pix_square_size_x = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels
        pix_square_size_y = ((self.window_size-80) / self.size)

        #Compute offset to center the grid
        grid_size_x = self.size * pix_square_size_x
        grid_size_y = self.size * pix_square_size_y
    
        x_offset = (canvas.get_width() - grid_size_x) // 2
        y_offset = (canvas.get_height() - grid_size_y) // 2
        
        #Render weather
        match self.current_weather:
            case "sunny":
                sky = pygame.transform.scale(self.sunny, (self.window_size, 80))
                top_surface.blit(sky, (0,0))
            case "cloudy":
                sky = pygame.transform.scale(self.cloudy, (self.window_size, 80))
                top_surface.blit(sky, (0,0))
            case "rainy":
                sky = pygame.transform.scale(self.rainy, (self.window_size, 80))
                top_surface.blit(sky, (0,0))

        #Render the grid
        cell_image = None
        for i in range(self.size):
            for j in range(self.size):
                match self.state[i,j]:
                    case 0:
                        #Resize the image 
                        cell_image = pygame.transform.scale(self.shaved_grass, (int(pix_square_size_x), int(pix_square_size_y)))
                    case 1:
                        cell_image = pygame.transform.scale(self.short_grass, (int(pix_square_size_x), int(pix_square_size_y)))
                    case 2:
                        cell_image = pygame.transform.scale(self.medium_grass, (int(pix_square_size_x), int(pix_square_size_y)))
                    case 3:
                        cell_image = pygame.transform.scale(self.strong_grass, (int(pix_square_size_x), int(pix_square_size_y)))
                    case 4:
                        cell_image = pygame.transform.scale(self.tree, (int(pix_square_size_x), int(pix_square_size_y)))
                    case 5:
                        cell_image = pygame.transform.scale(self.rock, (int(pix_square_size_x), int(pix_square_size_y)))
                    case 6:
                        cell_image = pygame.transform.scale(self.short_grass_lw, (int(pix_square_size_x), int(pix_square_size_y))) #shaved grass with agent
                    case 7:
                        cell_image = pygame.transform.scale(self.shaved_grass_lw, (int(pix_square_size_x), int(pix_square_size_y))) #short grass with agent
                    case 8:
                        cell_image = pygame.transform.scale(self.medium_grass_lw, (int(pix_square_size_x), int(pix_square_size_y))) #medium grass with agent
                    case 9:
                        cell_image = pygame.transform.scale(self.strong_grass_lw, (int(pix_square_size_x), int(pix_square_size_y))) #high grass with agent
                    case 10:
                        cell_image = pygame.transform.scale(self.tree_lw, (int(pix_square_size_x), int(pix_square_size_y))) #tree with agent
                    case 11:
                        cell_image = pygame.transform.scale(self.rock_lw, (int(pix_square_size_x), int(pix_square_size_y))) #rock with agent
                
                canvas.blit(cell_image, (j * pix_square_size_x + x_offset, i * pix_square_size_y + y_offset))
        
        # Draw the canvas onto the window
        self.window.blit(top_surface, (0, 0))
        self.window.blit(canvas, (0, 80))
        pygame.display.flip()

        # Update the screen to avoid freezing
        pygame.event.get()
    
    def create_weather_thread(self):
        if not self.game_over:
            self.weather_thread = threading.Thread(target=self.update_weather_and_grow)    
            self.weather_thread.start() #Start thread

    def destroy_thread(self):
        if self.weather_thread.is_alive():
            print("Sto aspettando il thread")
            self._stop_event.set()
            self.weather_thread.join(timeout=0) #Close thread
            #self.weather_thread = None

    def close(self):
        pygame.quit() #To close the game window
    
    def step(self, action):
        self.step_counter = self.step_counter + 1
        #print("Num azione: ",self.step_counter)
        reward = None

        #Check agent's location (If it is outside of the grid)
        if self._agent_location[0]<0 or self._agent_location[0]>=self.size or self._agent_location[1]<0 or self._agent_location[1]>=self.size: 
            reward = -1000
        else:
            #Check if the agent go outside of the game grid
            if self._agent_location[0] == 0:
                #The agent is on the first row and the first column of the game grid and the action is backward
                if self._agent_location[1]==0 and action==3:
                    reward=-1000
                #The agent is on the first row and the last column of the game grid and the action is forward
                elif self._agent_location[1]==(self.size-1) and action==1:
                    reward=-1000
                #The agent is on the first row of the game grid
                elif action == 2: #Go to left
                    reward = -1000
            elif self._agent_location[0] == (self.size-1):
                #The agent is on the last row and the first column of the game grid and the action is backward
                if self._agent_location[1]==0 and action==3:
                    reward=-1000
                #The agent is on the last row and the last column of the game grid and the action is forward
                elif self._agent_location[1]==(self.size-1) and action==1:
                    reward=-1000
                #The agent is on the last row of the game grid
                elif action == 0: #Go to right
                    reward = -1000
            elif self._agent_location[1] == 0:#The agent is on the first column
                if self._agent_location[0]==0 and action==2:
                    reward=-1000
                elif self._agent_location[0]==(self.size-1) and action==0:
                    reward=-1000
                elif action == 3: #Go to back
                    reward = -1000
            elif self._agent_location[1] == (self.size-1):
                if self._agent_location[0]==0 and action==2:
                    reward=-1000
                elif self._agent_location[0]==(self.size-1) and action==0:
                    reward=-1000
                elif action == 1: #Go ahead
                    reward = -1000

            if reward is None:
                #Take grid cell
                cell = self.state[self._agent_location[0], self._agent_location[1]]-6
                
                match cell:
                    case 0: #shaved grass
                        if action >=0 and action<=3 : #right-up-left-down
                            reward = 0
                            self.move_agent(action)
                        elif action == 4: #weak cut
                            reward = -1
                        elif action == 5: #medium cut
                            reward = -2
                        elif action == 6: #strong cut
                            reward = -3
                    case 1: #short grass
                        if action >=0 and action<=3 : #right-up-left-down
                            reward = -1
                            self.move_agent(action)
                        elif action == 4: #weak cut
                            reward = 3
                            self.state[self._agent_location[0], self._agent_location[1]] = 6
                        elif action == 5: #medium cut
                            reward = -1
                        elif action == 6: #strong cut
                            reward = -2
                    case 2: #grass medium height
                        if action >=0 and action<=3 : #right-up-left-down
                            reward = -1
                            self.move_agent(action)
                        elif action == 4: #weak cut
                            reward = -1
                        elif action == 5: #medium cut
                            reward = 6
                            self.state[self._agent_location[0], self._agent_location[1]] = 6
                        elif action == 6: #strong cut
                            reward = -1
                    case 3: #high grass
                        if action >=0 and action<=3 : #right-up-left-down
                            reward = -1
                            self.move_agent(action)
                        elif action == 4: #weak cut
                            reward = -2
                        elif action == 5: #medium cut
                            reward = -1
                        elif action == 6: #strong cut
                            reward = 9
                            self.state[self._agent_location[0], self._agent_location[1]] = 6
                    case 4: #tree
                        if action >=0 and action<=3 : #right-up-left-down
                            reward = 0
                            self.move_agent(action)
                        elif action == 4: #weak cut
                            reward = -3
                        elif action == 5: #medium cut
                            reward = -3
                        elif action == 6: #strong cut
                            reward = -3
                    case 5: #rock
                        if action >=0 and action<=3 : #right-up-left-down
                            reward = 0
                            self.move_agent(action)
                        elif action == 4: #weak cut
                            reward = -4
                        elif action == 5: #medium cut
                            reward = -4
                        elif action == 6: #strong cut
                            reward = -4

        #Update agent points
        self.points += reward

        #Update total penalty
        if reward<0:
            self.penalty += reward

        terminated = False
        if self.points >=200 or self.penalty<=-50: #point threshold - penalty threshold ?????
            terminated = True
            self.game_over = True
        
        if not terminated:
            if self.step_counter == self.weather_interval:
                #print("Aggiorno meteo!")
                self.step_counter = 0
                self.update_weather()
            else:
                #print("Erba cresce")
                self.grow_grass()

        
        #Get the next observation due to the agent actions
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        if self.game_over:
            self.game_over = False
            #self.destroy_thread()
            #self._stop_event.clear()
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Reset the agent point and penalties
        self.points = 0
        self.penalty = 0
        self.step_counter = 0 

        #Refresh the grass
        self.state = np.random.randint(low=1, high=4, size=(self.size, self.size))

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        

        #Set shaved grass in the agent's location
        self.state[self._agent_location[0], self._agent_location[1]]= 6 
        

        self.generate_obstacles()

        """
        if np.array_equal(self._agent_location, [0,0]):

            if any(np.array_equal([0,1], obstacle) for obstacle in self.obstacles_location) and any(np.array_equal([1,0], obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal([0,1], self.obstacles_location[i]) or np.array_equal([1,0], self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break       
        elif np.array_equal(self._agent_location, [0,self.size-1]):
            if any(np.array_equal([0,self.size-2], obstacle) for obstacle in self.obstacles_location) and any(np.array_equal([1,self.size-1], obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal([0,self.size-2], self.obstacles_location[i]) or np.array_equal([1,self.size-1], self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break  
        elif np.array_equal(self._agent_location, [self.size-1,0]):
            if any(np.array_equal([self.size-2,0], obstacle) for obstacle in self.obstacles_location) and any(np.array_equal([self.size-1,1], obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal([self.size-2,0], self.obstacles_location[i]) or np.array_equal([self.size-1,1], self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break  
        elif np.array_equal(self._agent_location, [self.size-1,self.size-1]):
            if any(np.array_equal([self.size-1,self.size-2], obstacle) for obstacle in self.obstacles_location) and any(np.array_equal([self.size-2,self.size-1], obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal([self.size-1,self.size-2], self.obstacles_location[i]) or np.array_equal([self.size-2,self.size-1], self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break  
        elif self._agent_location[1]==0: #agent in the first column
            agent_position = self._agent_location
            p1 = agent_position+[-1,0]
            p2 = agent_position+[0,1]
            p3 = agent_position+[1,0]
            if any(np.array_equal(p1, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p2, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p3, obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal(p1, self.obstacles_location[i]) or np.array_equal(p2, self.obstacles_location[i]) or np.array_equal(p3, self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break
        elif self._agent_location[1]==(self.size-1): #aget in the last column
            agent_position = self._agent_location
            p1 = agent_position+[-1,0]
            p2 = agent_position+[0,-1]
            p3 = agent_position+[1,0]
            if any(np.array_equal(p1, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p2, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p3, obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal(p1, self.obstacles_location[i]) or np.array_equal(p2, self.obstacles_location[i]) or np.array_equal(p3, self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break
        elif self._agent_location[0]==0: #agent in the first row
            agent_position = self._agent_location
            p1 = agent_position+[0,-1]
            p2 = agent_position+[1,0]
            p3 = agent_position+[0,1]
            if any(np.array_equal(p1, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p2, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p3, obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal(p1, self.obstacles_location[i]) or np.array_equal(p2, self.obstacles_location[i]) or np.array_equal(p3, self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break
        elif self._agent_location[0]==(self.size-1): #agent in the last row
            agent_position = self._agent_location
            p1 = agent_position+[0,-1]
            p2 = agent_position+[-1,0]
            p3 = agent_position+[0,1]
            if any(np.array_equal(p1, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p2, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p3, obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal(p1, self.obstacles_location[i]) or np.array_equal(p2, self.obstacles_location[i]) or np.array_equal(p3, self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break
        else:
            agent_position = self._agent_location
            p1 = agent_position+[0,-1]
            p2 = agent_position+[-1,0]
            p3 = agent_position+[0,1]
            p4 = agent_position+[1,0]
            if any(np.array_equal(p1, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p2, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p3, obstacle) for obstacle in self.obstacles_location) and any(np.array_equal(p4, obstacle) for obstacle in self.obstacles_location):

                for i in range(self.num_obstacle):

                    if np.array_equal(p1, self.obstacles_location[i]) or np.array_equal(p2, self.obstacles_location[i]) or np.array_equal(p3, self.obstacles_location[i]) or np.array_equal(p4, self.obstacles_location[i]):
                        ps = self.obstacles_location[i]
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                        self.check_obstacles_position(position=ps)
                        break       
        """

        #Add the obstacles on matrix
        i = 0        
        for location in self.obstacles_location:
            if i % 2 == 0:
                self.state[location[0], location[1]] = 4 # tree
            else:
                self.state[location[0], location[1]] = 5 # rock
            
            i = i +1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        return self.state
        #return {"agent": self._agent_location, "grid":self.state}
    
    def _get_info(self):
        return {"around":self.observation_space} #????
    
    def generate_obstacles(self):
        # We will sample the obstacle's location randomly until it does not coincide with the agent's location or other obstacles
        j = 0
        for i in range(self.num_obstacle):

            #Until the obstacle's location is equal to agent's location we re-generate its position
            while np.array_equal(self.obstacles_location[i], self._agent_location):
                self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
            
            #Now check if two obstacles have the same position
            flag = False
            while j<i:
                if np.array_equal(self.obstacles_location[j], self.obstacles_location[i]):
                    flag = True
                    #Until the two obstacles have the same location or the i obstacle has the same agent's location
                    while np.array_equal(self.obstacles_location[j], self.obstacles_location[i]) or np.array_equal(self._agent_location, self.obstacles_location[i]):
                        
                        self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                
                if flag:
                    j = 0
                    flag = False
                else:
                    j = j +1
            
            j = 0
    
    def move_agent(self, action):
        #Subtract +6 from the current cell because it represents the agent
        self.state[self._agent_location[0], self._agent_location[1]] = self.state[self._agent_location[0], self._agent_location[1]]-6
        match action:
            case 0:
                self._agent_location = self._agent_location + self._action_to_direction[0]
            case 1:
                self._agent_location = self._agent_location + self._action_to_direction[1]
            case 2:
                self._agent_location = self._agent_location + self._action_to_direction[2]
            case 3:
                self._agent_location = self._agent_location + self._action_to_direction[3]

        #Add +6 in the next cell
        self.state[self._agent_location[0], self._agent_location[1]] = self.state[self._agent_location[0], self._agent_location[1]]+6


    def update_weather(self):
        #Define array of probabilities
        weather_probabilities = np.array(list(self.weather_probabilities.values()))

        #Generate random weather
        self.current_weather = np.random.choice(list(self.weather_probabilities.keys()), p=weather_probabilities)
        self.grow_grass()
    

    def grow_grass(self, weather_interval= 30):
        match self.current_weather:
            case "sunny":
                #start_time = time.time()
                #while time.time() - start_time <= weather_interval and not self.game_over:

                #Get only shaved, short and medium grass
                only_grass = (self.state!=3) & (self.state != 4) & (self.state != 5) & (self.state<6)
                #Update grass height
                if self.step_counter!=0 and self.step_counter%20 == 0:
                    #print("Matrice prima:")
                    #print(self.state)
                    #time.sleep(3)
                    self.state[only_grass] = self.state[only_grass] + 1
                    #print("Matrice dopo:")
                    #print(self.state)
                    #time.sleep(3)
                
                #time.sleep(10)
            case "cloudy":
                #start_time = time.time()
                #while time.time() - start_time <= weather_interval and not self.game_over:
                
                #Get only shaved, short and medium grass
                only_grass = (self.state!=3) & (self.state != 4) & (self.state != 5) & (self.state<6)
                #Update grass height
                if self.step_counter!=0 and self.step_counter%40 == 0:
                    #print("Matrice prima:")
                    #print(self.state)
                    #time.sleep(3)
                    self.state[only_grass] = self.state[only_grass]+1
                    #print("Matrice dopo:")
                    #print(self.state)
                    #time.sleep(3)
                
                #time.sleep(20)
            case "rainy":
                #start_time = time.time()
                #while time.time() - start_time <= weather_interval and not self.game_over:
                
                #Get only shaved, short and medium grass
                only_grass = (self.state!=3) & (self.state != 4) & (self.state != 5) & (self.state<6)
                #Update grass height
                if self.step_counter!=0 and self.step_counter%30 == 0:
                    #print("Matrice prima:")
                    #print(self.state)
                    #time.sleep(3)
                    self.state[only_grass] = self.state[only_grass]+1
                    #print("Matrice dopo:")
                    #print(self.state)
                    #time.sleep(3)
                
                #time.sleep(15)

    def update_weather_and_grow(self):
        flag = False
        while not self.game_over:
            if not flag:
                flag = True
            else:
                self.update_weather()
    


