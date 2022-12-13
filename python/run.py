import time
import math
import sys
import os
import copy
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

from model_evaluate import evaluate_model

default_domain_path = '../pddl/simple.pddl'

MONTE_CARLO_SAMPLES=1000


class Experiment(object):
    def __init__(self, domain_path=default_domain_path):
        self.environment = Environment()
        self.problem_generator = ProblemGenSimple()
        self.planner = Planner(domain_path=domain_path)
        self.domain_path = domain_path
        self.model = CNN(num_locations=self.problem_generator.num_locations)
        self.satsolver = None

    def run(self, epi, reset=False):
        '''
        Runs the experiment.
        returns: nothing.
        '''
        # Get first observation
        print("Episode ", epi)

        obs = None
        
        if not reset:
            obs = self.environment.rendering_to_obs(self.environment.render())
            self.environment.timestep = 0
        else:
            obs, _ = self.environment.reset()
        
        # Initialize the model
        prediction = self.model.forward(obs)

        ### here, we manually set the goal location
        # location 9 (f5-4f) is the goal location, so exclude it
        agent_loc, goal_loc = self.get_locations(prediction, exclude_agent_loc=[9])
        goal_loc = "f5-4f"

        # Generate the problem file for planner
        self.problem_generator.generate(agent_loc, goal_loc)
        print(f"Predicted initial loc: {agent_loc}, goal: {goal_loc}")
        prob_path = '../pddl/simple/generated_problem.pddl'
        plan = self.planner.create_plan(prob_path)
        # Initialize the SAT solver
        self.satsolver = SATSolverModel(domain_file=self.domain_path, problem_file=prob_path, plan_horizon=len(plan))
        print("Generated Plan:", plan)

        if (len(plan) == 0):
            print ('No plan needed. Already at Goal.')
            print()
            return True, None, None

        last_obs, last_act = None, None
        for i, act in enumerate(plan):
            obs, timestep, done, info = self.environment.step(act)
            # last_obs = copy.deepcopy(obs)
            # last_act = copy.deepcopy(act)

            self.satsolver.report_action_result(action=act.predicate.name, iter=i, success=info['result'])
            print("    TimeStep", i, 
                "Prev Location:", info['location']['before'],
                "Action:", act, 
                "New Location:", info['location']['after'], 
                "successful" if info['result'] else "failed"
            )

            if not info['result']:
                # start reasoning now for the loss function
                # reasoned_samples = self.satsolver.get_start_rates(num_samples=MONTE_CARLO_SAMPLES)
                # loss, accuracy = self.model.train(x = obs, y = reasoned_samples)
                break
            elif done:
                break
        reasoned_samples = self.satsolver.get_start_rates(num_samples=MONTE_CARLO_SAMPLES)
        print(reasoned_samples)
        loss, accuracy = self.model.train(x = obs, y = reasoned_samples)

        return done, loss, accuracy

    def load_model():
        pass

    def train_model(self, num_eps):
        reset = True
        for episode in range(num_eps):
            reset, loss, accuracy = self.run(reset)
            print('Episode: ', episode, ' done: ', reset, 'loss: ', loss, 'accuracy: ', accuracy)


    def get_locations(self, prediction, exclude_agent_loc=[]):
        '''
        Gets the agent and goal locations from the prediction.
        agent location first, goal second.

        prediction: prediction from the model.
        returns: agent location, goal location
        '''
        prediction = prediction.detach().numpy().squeeze()
        for loc in exclude_agent_loc:
            # manually manipulate the prediction to exclude the goal
            prediction[loc] = -100
        # pick loc to be the one with the highest probability
        agent_loc = np.argmax(prediction[:self.problem_generator.num_locations])
        if prediction.shape[0] >= self.problem_generator.num_locations * 2:
            goal_loc = np.argmax(prediction[self.problem_generator.num_locations:])
        else:
            goal_loc = None
        return self.problem_generator.locations[agent_loc], \
            self.problem_generator.locations[goal_loc] if goal_loc is not None else None

        

if __name__ == '__main__':

    exp_1 = Experiment()
    success_rate_history = []
    done = True
    for i in range(1000):
        done, _, _ = exp_1.run(epi=i, reset=done)
        if done:
            print("Done!")
        acc, cross_entropy = evaluate_model(exp_1.model)
        print("Evaluation On pre-generated dataset: Accuracy:", acc.numpy(), "  Cross Entropy:", cross_entropy.numpy())
        print()
