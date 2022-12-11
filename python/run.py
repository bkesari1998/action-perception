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

default_domain_path = '../pddl/simple.pddl'

MONTE_CARLO_SAMPLES=10


class Experiment(object):
    def __init__(self, domain_path=default_domain_path):
        self.environment = Environment()
        self.problem_generator = ProblemGenSimple()
        self.planner = Planner(domain_path=domain_path)
        self.domain_path = domain_path
        self.model = CNN(num_locations=self.problem_generator.num_locations)
        self.satsolver = None

    def run(self, reset=False):
        '''
        Runs the experiment.
        returns: nothing.
        '''
        # Get first observation

        obs = None
        
        if not reset:
            obs = self.environment.rendering_to_obs(self.environment.render())
            self.environment.timestep = 0
        else:
            obs, _ = self.environment.reset()
        
        # Initialize the model
        prediction = self.model.forward(obs)
        agent_loc, goal_loc = self.get_locations(prediction)
        # Generate the problem file for planner
        self.problem_generator.generate(agent_loc, goal_loc)
        prob_path = '../pddl/simple/generated_problem.pddl'
        plan = self.planner.create_plan(prob_path)
        # Initialize the SAT solver
        self.satsolver = SATSolverModel(domain_file=self.domain_path, problem_file=prob_path, plan_horizon=len(plan))


        last_obs, last_act = None, None
        for i, act in enumerate(plan):
            print(act)
            obs, timestep, done, info = self.environment.step(act)
            last_obs = obs
            last_act = act
            if info['result'] != 'success':
                # start reasoning now for the loss function
                reasoned_samples = self.satsolver.get_models()
                self.model.train(x = obs, y = reasoned_samples)
                break
        # now also update the model for the last action
        self.satsolver.report_action_result(action=last_act.predicate.name, iter=i, success=False)
        reasoned_samples = self.satsolver.get_start_rates(num_samples=MONTE_CARLO_SAMPLES)
        self.model.train(x = last_obs, y = reasoned_samples)
        return done

    def load_model():
        pass

    def train_model(self, num_eps):
        reset = True
        for episode in range(num_eps):
            reset = self.run(reset)


    def get_locations(self, prediction):
        '''
        Gets the agent and goal locations from the prediction.
        agent location first, goal second.

        prediction: prediction from the model.
        returns: agent location, goal location
        '''
        prediction = prediction.detach().numpy().squeeze()
        agent_loc = np.argmax(prediction[:self.problem_generator.num_locations])
        goal_loc = np.argmax(prediction[self.problem_generator.num_locations:])
        return self.problem_generator.locations[agent_loc], \
            self.problem_generator.locations[goal_loc]
        

if __name__ == '__main__':

    exp_1 = Experiment()
    exp_1.run()