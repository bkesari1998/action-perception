import pddlgym
from pddlgym_planners.fd import FD

# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvSimple-v0", raise_error_on_invalid_action=True)
print(env.action_space)
env.fix_problem_index(0)
obs, debug_info = env.reset()
planner = FD()
plan = planner(env.domain, obs)
print("Plan:", plan)
for act in plan:
    print("Obs:", obs)
    print("Act:", act)
    action_str = str(act).split("(")[0]
    action = env.action_space.sample(obs)
    print (action)
    obs, reward, done, info = env.step(action_str)
print("Final obs, reward, done:", obs, reward, done)