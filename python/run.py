import time
import math
import sys
import os
import numpy as np

# Path to pddlgym
sys.path.append('../../pddlgym')
# path to pddl
sys.path.append('../pddl')
sys.path.append('../')

from environment import Environment
from problem_gen import ProblemGenSimple
from planner import Planner
from model import CNN
from sat_solver_model import SATSolverModel

class Experiment(object):
    def __init__(self):
        self.environment = Environment()
        self.problem_generator = ProblemGenSimple()
        self.planner = Planner()
        self.model = CNN()
        self.satsolver = SATSolverModel()

    def run(self):
        '''
        Runs the experiment.
        returns: nothing.
        '''
        # Get first observation
        obs, info = self.environment.reset()
        # Initialize the model
        prediction = self.model.forward(obs)
        agent_loc, goal_loc = self.get_locations(prediction)
        # Generate the problem file for planner
        self.problem_generator.generate(agent_loc, goal_loc)
        prob_path = '/../pddl/simple/generated_problem.pddl'
        # domain_path = '/../pddl/simple.pddl'

        plan = self.planner.create_plan(prob_path)


        for act in plan:
            print(act)
            obs,info = self.environment.step(act)
            if info['result'] != 'success':
                # start reasoning now for the loss function
                reasoned_samples = self.satsolver.get_models()
                self.model.train(x = obs, y = reasoned_samples)
                break

    def load_model():
        pass

    def train_model():
        pass

    def get_locations(prediction):
        '''
        Gets the agent and goal locations from the prediction.

        prediction: prediction from the model.
        returns: agent location, goal location
        '''
        agent_loc = None
        goal_loc = None
        return agent_loc, goal_loc
        

if __name__ == '__main__':

    exp_1 = Experiment()
    exp_1.run()
    # send to print results
    # print results

    # # Initialize environment
    # env = Environment()
    # # initialize the problem generator
    # gen = ProblemGenSimple()
    # # get first observation
    # obs, info = env.reset()
    # # initialize the model
    # network = CNN(info['num_of_locs'])
    # prediction = network.forward(obs)
    # agent_loc, goal_loc = get_locations(prediction)
    # # generate the problem file for planner
    # gen.generate(agent_loc, goal_loc)
    # prob_path = '/../pddl/simple/generated_problem.pddl'
    # domain_path = '/../pddl/simple.pddl'

    # planner = Planner(domain_path)
    # plan = planner.create_plan(prob_path)

    # for act in plan:
    #     print(act)
    #     obs,info = env.step(act)
    #     if info['result'] != 'success':
    #         break
    






    # # send the image to the model.
    # # get the action from the model.
    # # step the environment with the action.
    # # repeat.






    # # Load model
    # model = load_model()

    # # Train model
    # model = train_model()


    # # Initialize planner
    # domain_path = '../pddl/simple.pddl'
    # planner = Planner(domain_path)

    # # Initialize problem generator
    # gen = ProblemGenSimple()

    # # Generate problem
    # at = 'f0-0f'
    # goal = 'f5-4f'
    # gen.generate(at, goal)
    # prob_path = '../pddl/simple/generated_problem.pddl'

    # # Create plan
    # plan = planner.create_plan(prob_path)

    # # Execute plan
    # for act in plan:
    #     print(act)
    #     env.step(act)

    # # Save plan
    # # ...

    # # Save model
    # # ...

    # # Save environment
    # # ...

    # # Save planner
    # # ...

    # # Save problem generator
    # # ...

    # # Save experiment
    # # ...

    # # Save results
    # # ...

    # # Save data
    # # ...

    # # Save logs
    # # ...

    # # Save images
    # # ...

    # # Save videos
    # # ...

    # # Save graphs
    # # ...

    # # Save plots
    # # ...

    # # Save figures
    # # ...

    # # Save tables
    # # ...

    # # Save statistics
    # # ...

    # # Save metrics
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...

    # # Save scores
    # # ...