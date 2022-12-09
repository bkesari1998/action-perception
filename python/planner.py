import sys
sys.path.append('../../')

from pddlgym.structs import Literal, Predicate

from pyperplan.search.sat import sat_solve
from pyperplan.planner import _parse, _ground
import os

class Planner(object):
    '''
    Wrapper for pyperplan sat_solver planner.
    '''

    def __init__(self, 
                domain_path):
        '''
        Initializes planner class.

        domain: rel path to pddl domain file.
        returns: none
        '''

        self.domain_path = os.path.dirname(__file__) + domain_path
        self.task = None

    def create_plan(self,
                    problem_path):
        '''
        Creates task for solver.

        problem: path to pddl problem file.
        returns: plan created by sat solver
        '''

        problem_path = os.path.dirname(__file__) + problem_path

        problem = _parse(self.domain_path, problem_path)
        task = _ground(problem)

        # Use sat solver to create plan
        plan = sat_solve(task)

        # Convert plan to pddlgym style plan
        pddlgym_plan = []
        pred = None

        for act in plan:
            
            if "move_down" in act.name:
                pred = Predicate("move_down", 0)
            elif "move_up" in act.name:
                pred = Predicate("move_up", 0)
            elif "move_left" in act.name:
                pred = Predicate("move_left", 0)
            else:
                pred = Predicate("move_right", 0)

            pddlgym_plan.append(Literal(pred, []))

        return pddlgym_plan

if __name__ == '__main__':

    domain_file =  "/../pddl/simple.pddl"
    problem_file = "/../pddl/simple/problem_0.pddl"

    planner = Planner(domain_file)

    plan = planner.create_plan(problem_file)
    print(plan)