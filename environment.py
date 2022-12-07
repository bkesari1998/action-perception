import sys

# Path to pddlgym
sys.path.append('/Users/bharatkesari/Documents/tufts/academic/cs_137/project/pddlgym')
# Path to pddlgym planners
sys.path.append('/Users/bharatkesari/Documents/tufts/academic/cs_137/project')

import pddlgym
from pddlgym.structs import Literal
from gym import error
from pddlgym.core import InvalidAction

class Environment(object):
    '''
    Wrapper for pddlgym environment.
    '''

    def __init__(self, 
                env_name: str):
        '''
        Initializes pddlgym environment
        '''
        self.env = None

        while not self.env:
            try:
                # Create environmet
                self.env = pddlgym.make(env_name, raise_error_on_invalid_action=True)
            except error.NameNotFound as e:
                print(e)
                # Ask user to input correct environment name
                env_name = input("Please input correct environment name: ")
        
        
    def get_img(self):
        '''
        Wrapper for render function for gym environment.
        Returns image file of environment at current timestep.
        '''
        return self.env.render()

    def step(self, act: Literal):
        '''
        Wrapper for gym step function.
        '''

        # Try executing the actions 
        try:
            self.env.step(act)
        except InvalidAction as e:
            return False
        
        return True
    