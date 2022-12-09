from pyperplan.search.sat import sat_solve

class Planner(object):
    '''
    Wrapper for pyperplan sat_solver planner.
    '''

    def __init__(self, domain):