import os

class ProblemGenSimple(object):
    '''
    Class to generate problem files on the fly for simple domain.
    '''

    def __init__(self, 
                problem_basis_path='../pddl/simple/problem_basis.pddl',
                problem_file_path='../pddl/simple/generated_problem.pddl'):
        
        '''
        Initializes problem generator for simple domain.

        problem_file_path: relative path to problem file to base new problem files from.
        returns: none.
        '''

        self.problem_basis_path = os.path.dirname(__file__) + problem_basis_path

    def generate(self,
                at,
                goal):
        '''
        Generates a problem file given the agent's position and goal position.

        returns: none.
        '''

        problem_file = open(self.problem_basis_path, "r")
        problem_file = open(self. )