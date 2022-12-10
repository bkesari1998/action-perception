# sat_resolver.py
# given a problem file and initial states, generate CNF.
#
#

from pyperplan.search.sat import get_plan_formula
from pyperplan.planner import _parse, _ground

from pysat.solvers import Glucose3
from pysat.formula import CNF
from cnfwriter import CnfWriter, format_cnfstr_to_dimacs

import os
from copy import deepcopy

class SATSolverModel:
    def __init__(self, domain_file, problem_file, plan_horizon=9):
        # parse the problem and domain file.
        domain_file = os.path.dirname(__file__) + "/../../pddl/simple.pddl"
        problem_file = os.path.dirname(__file__) + "/../../pddl/simple/problem_0.pddl"

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

        # configure a solver
        self.solver = Glucose3(bootstrap_with=cnf_str)
        
        pass

    def report_action_result(self, action: str, iter: int, success: bool=False):
        """
        Receives report from action executer about whether one action is
        successfully executed, and update its belief based on that.
        """
        pass

    def sample(self, n=1):
        """
        Samples from the current belief, give a solution to the SAT problem.
        """
        isSAT = self.solver.solve()
        if isSAT:
            pass
