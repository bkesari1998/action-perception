# sat_resolver.py
# given a problem file and initial states, generate CNF.
#
#

from pyperplan.search.sat import get_plan_formula
from pyperplan.planner import _parse, _ground

from pysat.solvers import Glucose3
from pysat.formula import CNF
from cnf_writer import CnfWriter
from cnf_formulas import format_cnfstr_to_dimacs, get_init_state_from_model, \
    free_goal_state, free_initial_state, generate_formula

import os
from copy import deepcopy
import random
from datetime import datetime

random.seed(datetime.now)

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
            "(at f5-4f)-0": 3,
        }

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
        self.freed_cnf = free_goal_state(goals, self.cnf, self.vars_to_num)
        self.freed_cnf = free_initial_state(self.freed_cnf, self.num_to_vars)

        # configure a solver
        self.solver = Glucose3(bootstrap_with=self.freed_cnf)
        
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
        for op in operators:
            # if the grounded operator is the current action
            if op.name.startswith("(" + action):
                
                for pre in op.preconditions:
                    possible_preconditions.append(pre)

        #any of these preconditions could have failed
        print(possible_preconditions)
        
        #add step number of failure
        possible_preconditions = [p + f"-{iter}" for p in possible_preconditions]

        #translate to numbers and **negate** them
        if success:
            # if success, one of it should be true
            possible_precondition_nums = [self.vars_to_num[p] for p in possible_preconditions]
        else:
            # if failed, one of it should be false
            possible_precondition_nums = [-self.vars_to_num[p] for p in possible_preconditions]

        # append it to the existing formula
        self.freed_cnf.append(possible_precondition_nums)

    def sample(self, num_samples):
        """
        Samples from the current belief, give a solution to the SAT problem.
        """
        solver = Glucose3(bootstrap_with=self.freed_cnf)
        isSAT = solver.solve()
        if isSAT:
            models = solver.enum_models()

            return random.choices(list(models), k=num_samples)
        else:
            return None

    def sample_init_states(self, num_samples):
        """
        Returns init states from sampled models.
        """

        # Sample models
        models = self.sample(num_samples)

        init_states = []

        # Loop over models
        for model in models:

            # Extract positive clauses from initial states
            init_state = get_init_state_from_model(model, self.nums_to_vars)
            init_state = [s for s in init_state if not s.startswith("(not ")]

            # Append to list
            init_states.append(init_state)

        return init_states
    
    def get_start_rates(self, num_samples):

        # Sample initial states from models
        init_states = self.sample_init_states(num_samples)

        # Create count of each starting location
        start_counts = [0] * len(self.loc_dict.items)

        # Count the number of times each starting location appears
        for state in init_states:
            start_counts[self.loc_dict[state[0]]] += 1
        
        start_counts /= len(start_counts)

        # Return rate of each location
        return start_counts


