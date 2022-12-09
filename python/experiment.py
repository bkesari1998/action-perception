from environment import Environment
from problem_gen import ProblemGenSimple
from planner import Planner

if __name__ == '__main__':

    at = 'f0-0f'
    goal = 'f5-4f'

    env = Environment()
    domain_path = '/../pddl/simple.pddl'
    planner = Planner(domain_path)

    gen = ProblemGenSimple()

    gen.generate(at, goal)
    prob_path = '/../pddl/simple/generated_problem.pddl'

    plan = planner.create_plan(prob_path)

    for act in plan:
        print(act)
        env.step(act)