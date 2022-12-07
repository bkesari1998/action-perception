import sys

# Path to pddlgym
sys.path.append('/Users/bharatkesari/Documents/tufts/academic/cs_137/project/pddlgym')
# Path to pddlgym planners
sys.path.append('/Users/bharatkesari/Documents/tufts/academic/cs_137/project')

import pddlgym
import pddlgym_planners
from gym import error

class Environment(object):
    '''
    Wrapper for pddlgym environment.
    '''

    def __init__(self, env_name: str) -> None:
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
        Returns image file of environment at current timestep
        '''
        return self.env.render()


if __name__ == '__main__':

# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvSimple-v0", raise_error_on_invalid_action=True)
print(env.action_space)
env.fix_problem_index(0)
obs, debug_info = env.reset()
planner = FD()
plan = planner(env.domain, obs)
print("Plan:", plan)
for act in plan:
    print("Obs:", obs)
    print("Act:", act)
    action_str = str(act).split("(")[0]
    action = env.action_space.sample(obs)
    print (action)
    obs, reward, done, info = env.step(action_str)
print("Final obs, reward, done:", obs, reward, done)