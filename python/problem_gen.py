import os

class ProblemGenSimple(object):
    '''
    Class to generate problem files on the fly for simple domain.
    '''

    def __init__(self, 
                problem_file_path='/../pddl/simple/generated_problem.pddl'):
        
        '''
        Initializes problem generator for simple domain.

        problem_file_path: relative path to the newly created problem files.
        returns: none.
        '''

        self.problem_file_path = os.path.dirname(__file__) + problem_file_path

        self.problem = \
        "(define (problem simple_problem) (:domain simple_domain)\n\
(:objects\n\
    f0-0f - location\n\
    f0-1f - location\n\
    f0-2f - location\n\
    f1-2f - location\n\
    f2-2f - location\n\
    f3-2f - location\n\
    f4-2f - location\n\
    f5-2f - location\n\
    f5-3f - location\n\
    f5-4f - location\n\
)\n\n\
(:init\n\
    (move_up)\n\
    (move_down)\n\
    (move_right)\n\
    (move_left)\n\
    (at %s)\n\n\
    (below f0-1f f0-0f)\n\
    (above f0-0f f0-1f)\n\n\
    (below f0-2f f0-1f)\n\
    (above f0-1f f0-2f)\n\n\
    (right f1-2f f0-2f)\n\
    (left f0-2f f1-2f)\n\n\
    (right f2-2f f1-2f)\n\
    (left f1-2f f2-2f)\n\n\
    (right f3-2f f2-2f)\n\
    (left f2-2f f3-2f)\n\n\
    (right f4-2f f3-2f)\n\
    (left f3-2f f4-2f)\n\n\
    (right f5-2f f4-2f)\n\
    (left f4-2f f5-2f)\n\n\
    (below f5-3f f5-2f)\n\
    (above f5-2f f5-3f)\n\n\
    (below f5-4f f5-3f)\n\
    (above f5-3f f5-4f)\n\n\
    (is-goal %s)\n\
        )\n\n\
(:goal (and\n\
    (at %s)\n\
    ))\n\
)"

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