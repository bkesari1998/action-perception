import time
import math
import sys
import os
import copy
import numpy as np
import torch

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
        self.goal = 'f5-4f'

        self.loc_dict = {
            "f0-0f": 0,
            "f0-1f": 1,
            "f0-2f": 2,
            "f1-2f": 3,
            "f2-2f": 4,
            "f3-2f": 5,
            "f4-2f": 6,
            "f5-2f": 7,
            "f5-3f": 8,
            "f5-4f": 9,
        }

    def run(self, epi):
        '''
        Runs the experiment.
        returns: nothing.
        '''

        print("Episode ", epi)

        # Get first observation
        obs, _ = self.environment.reset()

        # Ground truth init location
        gt_loc = self.environment.location
        optimal_plan_length = self.loc_dict[self.goal] - self.loc_dict[gt_loc]

        
        # Initialize the model
        prediction = self.model.forward(obs)

        # Get agent location
        agent_loc, _ = self.get_locations(prediction, exclude_agent_loc=[9])

        # Generate the problem file for planner
        self.problem_generator.generate(agent_loc, self.goal)
        print(f"Predicted initial loc: {agent_loc}, goal: {self.goal}")

        prob_path = '../pddl/simple/generated_problem.pddl'
        plan = self.planner.create_plan(prob_path)

        # Initialize the SAT solver
        self.satsolver = SATSolverModel(domain_file=self.domain_path, problem_file=prob_path, plan_horizon=len(plan))
        print("Generated Plan:", plan)

        if (len(plan) == 0):
            print ('No plan needed. Already at Goal.')
            print()
            return True, None, None

        Done = None
        num_successful_actions = 0

        for i, act in enumerate(plan):
            _, _, done, info = self.environment.step(act)

            self.satsolver.report_action_result(action=act.predicate.name, iter=i, success=info['result'])
            print("    TimeStep", i, 
                "Prev Location:", info['location']['before'],
                "Action:", act, 
                "New Location:", info['location']['after'], 
                "successful" if info['result'] else "failed"
            )

            if info['result']:
                num_successful_actions += 1

            if not info['result']:
                # start reasoning now for the loss function
                # reasoned_samples = self.satsolver.get_start_rates(num_samples=MONTE_CARLO_SAMPLES)
                # loss, accuracy = self.model.train(x = obs, y = reasoned_samples)
                break
            elif done:
                break
        
        successful_actions_vs_opt_plan_len = num_successful_actions / optimal_plan_length

        reasoned_samples = self.satsolver.get_start_rates(num_samples=MONTE_CARLO_SAMPLES)
        print("Reasoned Samples: ", reasoned_samples.squeeze().round(2))

        # Get probabilities of predicted locations
        model_probs = self.model.get_probability(prediction)
        print("Model Probabilities: ", model_probs.detach().numpy().squeeze().round(2))

        # Train over the samples for 100 iterations
        for i in range(100):
            loss = self.model.train(x = obs, y = reasoned_samples)

        return done, loss, successful_actions_vs_opt_plan_len

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
    loss_history = []
    action_success_rate = []
    batch_action_success_rate = []

    for i in range(1000):
        done, loss, rate = exp_1.run(i)
        loss_history.append(loss.detach().numpy())
        action_success_rate.append(rate)

        if done:
            print("Agent reached goal!")
        else:
            print("Agent failed to reach goal!")

        if i % 100 == 0:
            print("Saving loss history...")
            np.save('loss_history.npy', np.array(loss_history))
        
        if i % 10 == 0:
            print("Training Loss: ", loss.detach().numpy())
            mean_action_success_rate = np.mean(action_success_rate)
            print("Action Success Rate: ", mean_action_success_rate)
            action_success_rate = []
            batch_action_success_rate.append(mean_action_success_rate)
        
        print()
        print()

