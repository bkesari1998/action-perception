from pyperplan.search.sat import sat_solve
from pyperplan.planner import _parse, _ground

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
