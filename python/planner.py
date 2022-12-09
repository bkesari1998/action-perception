from pyperplan.search.sat import sat_solve
from pyperplan.planner import _parse, _ground
import os

class Planner(object):
    '''
    Wrapper for pyperplan sat_solver planner.
    '''

    def __init__(self, 
                domain):
        '''
        Initializes planner class.

        domain: path to pddl domain file.
        '''

        self.domain = domain
        self.task = None

    def create_plan(self,
                    problem):
        '''
        Creates task for solver.

        problem: path to pddl problem file.
        returns: plan created by sat solver
        '''

        problem = _parse(self.domain, problem)
        task = _ground(problem)
        return sat_solve(task)

if __name__ == '__main__':

    domain_file = os.path.dirname(__file__) + "/../pddl/simple.pddl"
    problem_file = os.path.dirname(__file__) + "/../pddl/simple/problem_0.pddl"

    planner = Planner(domain_file)

    plan = planner.create_plan(problem_file)

    print(plan)