import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
import pygame

import numpy as np

class LawnMowingEnvironment(Env):
    
    #Specify the render-modes 
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="None", size=8):
        self.size = size  # The size of the square grid
        self.state = np.random.randint(low=0, high=4, size=(size, size)) #initialize the grid
        
        self._agent_location = np.array([0,0]) #agent's location

        self.num_obstacle = size-2
        self.obstacles_location = np.zeros((self.num_obstacle,2), dtype=int) #obstacles' location

        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's location and the position of its surrounding.
        # The agent's position is represented by a vector that contains the coordinate. The vector values are in the range 0-7
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "right": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "left": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "up": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "down": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

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
            6: np.array([0, 0]),
            7: np.array([0, 0])
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

    def render(self):
        pass
    
    def step(self):
        pass

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self.generate_obstacles()

        """if np.array_equal(self._agent_location, [0,0]):

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
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        return {"agent": self._agent_location}
    
    def _get_info(self):
        return {"around":self.observation_space} #????
    
    def generate_obstacles(self, position=None):
        # We will sample the obstacle's location randomly until it does not coincide with the agent's location or other obstacles
        j = 0
        for i in range(self.num_obstacle):

            if position is None:
                #until the obstacle's location is equal to agent's location we re-generate its position
                while np.array_equal(self.obstacles_location[i], self._agent_location): 
                    self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
            else:
                #Until the obstacle's location is equal to agent's location or in a no valid position
                while np.array_equal(self.obstacles_location[i], self._agent_location) or np.array_equal(self.obstacles_location[i], position): 
                    self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
            
            #Now check if two obstacles have the same position
            flag = False
            while j<i:
                if np.array_equal(self.obstacles_location[j], self.obstacles_location[i]):
                    flag = True
                    if position is None:
                        #Until the two obstacles have the same location or the i obstacle has the same agent's location
                        while np.array_equal(self.obstacles_location[j], self.obstacles_location[i]) or np.array_equal(self._agent_location, self.obstacles_location[i]): 
                            self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                    else:
                        #Until the two obstacles have the same location or the i-obstacle has the same agent's location or has a no vali position
                        while np.array_equal(self.obstacles_location[j], self.obstacles_location[i]) or np.array_equal(self._agent_location, self.obstacles_location[i]) or np.array_equal(position, self.obstacles_location[i]): 
                            self.obstacles_location[i] = self.np_random.integers(0, self.size, size=2, dtype=int)
                
                if flag:
                    j = 0
                else:
                    j+=1
            
            j = 0
    


