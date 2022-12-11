import os

class ProblemGenSimple(object):
    '''
    Class to generate problem files on the fly for simple domain.
    '''

    def __init__(self, 
                problem_file_path='../pddl/simple/generated_problem.pddl',
                problem_file_template='../pddl/simple/problem_template.pddl'):
        
        '''
        Initializes problem generator for simple domain.

        problem_file_path: relative path to the newly created problem files.
        returns: none.
        '''

        self.problem_file_path = os.path.dirname(__file__) + "/" + problem_file_path
        self.problem_file_template = os.path.dirname(__file__) + "/" + problem_file_template

        with open(self.problem_file_template) as f:
            self.problem = f.read()
            self.num_locations = self.problem.count("- location")


    def generate(self,
                at,
                goal):
        '''
        Generates a problem file given the agent's position and goal position.

        at: str describing agents position
        goal: str describing goal
        returns: none.
        '''

        # Check types for input
        if type(at) != str and type(goal) != str:
            raise TypeError("Parameters 'at' and 'goal' must be 'str'.")


        # Write formatted problem
        problem_str = self.problem % (at, goal, goal)

        # Open file
        problem_file = open(self.problem_file_path, "w")

        # Write to the file
        problem_file.write(problem_str)


if __name__ == '__main__':

    gen = ProblemGenSimple()
    gen.generate("test", "test")