import sys

# Path to pddlgym
sys.path.append('../../pddlgym')

import pddlgym
from gym import error
from pddlgym.core import InvalidAction
import imageio

class Environment(object):
    '''
    Wrapper for pddlgym environment.
    '''

    def __init__(self, 
                env_name="PDDLEnvSimple-v0",
                problem_index=0):
        '''
        Initializes pddlgym environment.

        env_name: pddlgym style environment name string.
        problem_index: pddl problem file index
        returns: nothing.
        '''
        self.env = None

        while not self.env:
            try:
                # Create environmet
                self.env = pddlgym.make(env_name, raise_error_on_invalid_action=True)

                # Fix problem index
                problem_index_fixed = False
                while not problem_index_fixed:
                    try:
                        self.env.fix_problem_index(problem_index)
                        problem_index_fixed = True
                    except: # add exception
                        # Ask user to input existing problem index
                        problem_index = input("Please input existing problem index")

            except error.NameNotFound as e:
                print(e)
                # Ask user to input correct environment name
                env_name = input("Please input correct environment name: ")

        # Variable to track timesteps
        self.timestep = None 
        self.reset()
        
        
    def render(self):
        '''
        Wrapper for render function for gym environment.
        returns: image array of environment at current timestep.
        '''
        return self.env.render()

    def save_render(self, 
                    img, 
                    prefix="results/"):
        '''
        Wrapper to save rendered image of environment for current timestep.
        img: image array of render.
        prefex: directory to save render image file
        '''

        imageio.imsave(f'{prefix}{str(self.timestep).zfill(3)}.png', img)

    def step(self, 
            action):
        '''
        Wrapper for gym step function.
        '''

        # Try executing the actions 
        try:
            self.env.step(action)
            self.timestep += 1

        # Action fails
        except InvalidAction as e:
            print(e)
            img = self.render()
            self.save_render(img)

    def reset(self):
        '''
        Wrapper for gym reset function. Fixes problem index for pddl problem file.
        '''

        # Set timesteps to 0
        self.timestep = 0

        # Reset environment
        self.env.reset()