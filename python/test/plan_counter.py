import subprocess
from pyperplan.search.sat import get_plan_formula
from pyperplan.planner import _parse, _ground
from cnfwriter import CnfWriter
from pysat.solvers import Glucose3
from io import StringIO
from pysat.formula import CNF
import sys
import os
import logging

def format_cnfstr_to_dimacs(cnf_str: str):
    """
    Convert a CNF object to a DIMACS string
    """
    cnf = CNF(from_string=cnf_str)
    cnf_str = writer.get_cnf_str()
    n_vars = cnf.nv
    result = f"p cnf {n_vars} {len(cnf.clauses)}\n"+cnf_str

    return result


def print_positions(model):

    for var in model:
        
        if not abs(var) in num_to_vars:
            continue
        #if var<0:
        #    continue

        name= num_to_vars[abs(var)]
        if name.endswith("-3"):
            break
        if name.startswith("(not-at"):
            continue
        if var<0:
            continue
        if name.startswith("(at"):
            print(var, name)
    



if __name__ == "__main__":
    domain_file = os.path.dirname(__file__) + "/../../pddl/simple.pddl"
    problem_file = os.path.dirname(__file__) + "/../../pddl/simple/problem_0.pddl"
    problem = _parse(domain_file, problem_file)
    task = _ground(problem)
    #also read and parse a problem file with empty init state (not really empty, only whatever predicates we perceive)

    plan_horizon = 20

    formula = get_plan_formula(task, plan_horizon)
    
    writer = CnfWriter()
    vars_to_numbers = writer.write(formula)
    num_to_vars = {v: k for k, v in vars_to_numbers.items()}
    #print(num_to_vars)
    
    cnf_str = writer.get_cnf_str()
    print(cnf_str)
    dimacs_str = format_cnfstr_to_dimacs(cnf_str)

    cnf = CNF(from_string=cnf_str)

    solver = Glucose3(bootstrap_with=cnf.clauses)
    with open("test.cnf", "w") as f:
        f.write(dimacs_str)

    outfile = 'models.txt'

    isSAT = solver.solve()
    print(isSAT)
    if isSAT:
        #enumerate models
        #TODO: add assumptions based on action failure ..

        #TODO:collect models consistent with assumptions
        for i, m in enumerate(solver.enum_models()):
            pass
        
        #TODO:extract init states from models
        #
    
    can_be_at_t_4 = "(at player-1 loc-3-4)-4"
    cannot_be_at_t_4 = "(at player-1 loc-2-3)-20"

    print(vars_to_numbers[can_be_at_t_4])
    print(vars_to_numbers[cannot_be_at_t_4])

    is_sat = solver.solve(assumptions=[vars_to_numbers[can_be_at_t_4], -vars_to_numbers[cannot_be_at_t_4]])
    print(is_sat)



	# THIS HERE IS SLOW STUPID AND NOT NECESSARY. WE DONT NEED TO CALL AN EXTERNAL SOLVER AS PYSAT CAN ENUMERATE MODELS FOR US
	# https://pysathq.github.io/docs/html/api/solvers.html#pysat.solvers.Solver
	# should let you enumerate over solutions to the formula
