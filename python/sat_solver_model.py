# sat_resolver.py
# given a problem file and initial states, generate CNF.
#
#

from pyperplan.search.sat import get_plan_formula
from pyperplan.planner import _parse, _ground

from pysat.solvers import Glucose3
from pysat.formula import CNF
from cnf_writer import CnfWriter
from cnf_formulas import format_cnfstr_to_dimacs, get_init_goal_state_from_model, \
    free_goal_state, free_initial_state, generate_formula, get_exactly_one_clauses, \
    get_initial_state_clauses
from sat_solver_sampler import cmsgen_solve

import numpy as np

import os
from copy import deepcopy

# def get_goal(problem):
# 
# 

class SATSolverModel:
    def __init__(self, domain_file, problem_file, plan_horizon=9):
        # parse the problem and domain file. TODO parameterize this
        domain_file = os.path.dirname(__file__) + "/" + domain_file
        problem_file = os.path.dirname(__file__) + "/" + problem_file

        self.loc_dict = {
            "(at f0-0f)-0": 0,
            "(at f0-1f)-0": 1,
            "(at f0-2f)-0": 2,
            "(at f1-2f)-0": 3,
            "(at f2-2f)-0": 4,
            "(at f3-2f)-0": 5,
            "(at f4-2f)-0": 6,
            "(at f5-2f)-0": 7,
            "(at f5-3f)-0": 8,
        }
        # TODO add goals

        self.problem = _parse(domain_file, problem_file)
        self.task = _ground(self.problem)
        self.plan_horizon = plan_horizon

        # TODO: also read and parse a problem file with empty init state (not really empty, only whatever predicates we perceive)
        # get the formula for every possible situation at each timestep regardless of plan
        self.formula = get_plan_formula(self.task, self.plan_horizon)

        # convert formula into cnf, storing the num/var correlation dict.
        writer = CnfWriter()
        fml_copy = deepcopy(self.formula)
        self.vars_to_num = writer.write(fml_copy)
        self.num_to_vars = {v: k for k, v in self.vars_to_num.items()}
        cnf_str = writer.get_cnf_str()
        # self.dimacs_str = format_cnfstr_to_dimacs(self.cnf_str)
        self.cnf = CNF(from_string=cnf_str)

        # free cnf
        goals = [goal + "-" + str(self.plan_horizon) for goal in list(self.task.goals)]

        # remove the goal states since we made an assumption but don't need it
        self.freed_cnf = free_goal_state(goals, self.cnf, self.vars_to_num)
        init_states = get_initial_state_clauses(self.freed_cnf, self.num_to_vars)
        self.freed_cnf = free_initial_state(self.freed_cnf, self.num_to_vars)

        # add exactly one clause constraints on the initial pos and goal pos
        atleast_one, at_most_one = get_exactly_one_clauses(init_states)
        self.freed_cnf.append(atleast_one)
        self.freed_cnf += at_most_one
        pass

    def report_action_result(self, action: str, iter: int, success: bool=False):
        """
        Receives report from action executer about whether one action is
        successfully executed, and update its belief based on that.
        """
        # get all possible grounded operators
        operators = self.task.operators
        
        # see what could have failed
        possible_preconditions = []
        possible_postconditions = []
        for op in operators:
            # if the grounded operator is the current action
            if op.name.startswith("(" + action):
                possible_preconditions += list(op.preconditions)
                possible_postconditions += list(op.add_effects)

        #any of these preconditions could have failed
        # print("       Possible related preconditions:", "; ".join(possible_preconditions))
        
        #add step number of failure
        possible_preconditions = [p + f"-{iter}" for p in possible_preconditions]
        possible_postconditions = [p + f"-{iter + 1}" for p in possible_postconditions]

        # print(possible_postconditions)
        # print(possible_preconditions)

        #translate to numbers and **negate** them
        if success:
            # if success, one of it should be true
            possible_precondition_nums = [self.vars_to_num[p] for p in possible_preconditions]
            # possible_postcondition_nums = []
            possible_postcondition_nums = [self.vars_to_num[p] for p in possible_postconditions]
            self.freed_cnf.append(possible_precondition_nums)
        else:
            # if failed, all of it should be false
            possible_precondition_nums = [-self.vars_to_num[p] for p in possible_preconditions]
            # possible_postcondition_nums = [-self.vars_to_num[p] for p in possible_postconditions]
            possible_postcondition_nums = []
            for element in possible_precondition_nums:
                self.freed_cnf.append([element])
            # for element in possible_postcondition_nums:
            #     self.freed_cnf.append([element])

        # append it to the existing formula
        if len(possible_postcondition_nums) > 0:
            self.freed_cnf.append(possible_postcondition_nums)

    def sample(self, num_samples):
        """
        Samples from the current belief, give a solution to the SAT problem.
        """
        return cmsgen_solve(CNF(from_clauses=self.freed_cnf), num_samples)

    def sample_init_states(self, num_samples):
        """
        Returns init states from sampled models.
        """

        # Sample models
        models = self.sample(num_samples)

        init_goal_states = []

        # Loop over models
        for model in models:

            # Extract positive clauses from initial states
            ig_state = get_init_goal_state_from_model(model, self.num_to_vars, plan_horizon=self.plan_horizon)
            ig_state_filtered = [s for s in ig_state if 'not' not in s]

            # Append to list
            init_goal_states.append(ig_state_filtered)
        # DEBUG INFO
        # print("       Sampled initial states:", "; ".join([state[0].split("(at ")[1].split(")-0")[0] for state in init_goal_states][:20]))
        return init_goal_states
    
    def get_start_rates(self, num_samples, include_goals=False):

        # Sample initial states from models
        init_states = self.sample_init_states(num_samples)

        # Create count of each starting location
        start_counts = np.zeros(len(self.loc_dict))
        goal_counts = np.zeros(len(self.loc_dict))

        # Count the number of times each starting location appears
        for state in init_states:
            for state_var in state:
                if ")-0" in state_var:
                    start_counts[self.loc_dict[state_var]] += 1
                else:
                    new_name = state_var.split("-")[:-1]
                    new_name.append("0")
                    state_var_2 = "-".join(new_name)
                    goal_counts[self.loc_dict[state_var_2]] += 1
        
        # Get number of possible starting locations
        start_counts = start_counts[:len(start_counts) - 1]
        num_possible = np.sum(start_counts > 0)

        # Give locations that are possible a value of 1
        start_counts[start_counts > 0] = 1

        # Give equal probability to each possible location
        start_counts /= num_possible
        goal_counts /= len(init_states)

        # Return rate of each location
        if include_goals:
            return np.concatenate([start_counts, goal_counts])[np.newaxis, :]
        else:
            return start_counts[np.newaxis, :]

