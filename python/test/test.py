
import sys
sys.path.append('../../../pddlgym')
sys.path.append('/Users/bharatkesari/Documents/tufts/academic/cs_137/project')

import pddlgym
from pddlgym_planners.fd import FD

import matplotlib; matplotlib.use('agg')
import imageio
 
# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvSimple-v0")
# env.fix_problem_index(2)
obs, debug_info = env.reset()
planner = FD()
plan = planner(env.domain, obs)

def render_and_save(env, i, prefix="results/"):
    img = env.render()
    imageio.imsave(f'{prefix}{str(i).zfill(3)}.png', img)

render_and_save(env, i=0)

for i, act in enumerate(plan):
    # print("Obs:", obs)
    print("Act:", act)
    obs, reward, done, info = env.step(act[0])
    render_and_save(env, i=i + 1)
    
print("Final obs, reward, done:", obs, reward, done)
