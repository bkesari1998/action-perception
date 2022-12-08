import subprocess
from pyperplan.search.sat import get_plan_formula
from pyperplan.planner import _parse, _ground
from python.test.cnfwriter import CnfWriter
from pysat.solvers import Glucose3
from io import StringIO
from pysat.formula import CNF
import sys

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


if __name__ == "__main__":

    domain_file = "./pddlgym/pddlgym/pddl/gripper.pddl"
    problem_file = "./pddlgym/pddlgym/pddl/gripper/prob01.pddl"
    problem = _parse(domain_file, problem_file)
    task = _ground(problem)

    plan_horizon = 15

    formula = get_plan_formula(task, plan_horizon)

    writer = CnfWriter()
    writer.write(formula)
    cnf_str = writer.get_cnf_str()
    dimacs_str = format_cnfstr_to_dimacs(cnf_str)

    cnf = CNF(from_string=cnf_str)

    solver = Glucose3(bootstrap_with=cnf.clauses)
    with open("test.cnf", "w") as f:
        f.write(dimacs_str)

    outfile = 'models.txt'

    isSAT = solver.solve()
    print(isSAT)
    if isSAT:
	# THIS HERE IS SLOW STUPID AND NOT NECESSARY. WE DONT NEED TO CALL AN EXTERNAL SOLVER AS PYSAT CAN ENUMERATE MODELS FOR US
	# https://pysathq.github.io/docs/html/api/solvers.html#pysat.solvers.Solver
	# solver.enum_models() should let you enumerate over solutions to the formula

        try:
            print("Counting with  relsat")
            process = subprocess.Popen(
                ['./relsat', "-o", outfile, "-#10", "test.cnf"], stderr=subprocess.PIPE, stdout=subprocess.PIPE
            )
            code = process.wait()
            result = process.stdout.readlines()
            print(code)
        except OSError:
            logging.error(
                "relsat could not be found. "
                f'Please make the executable "relsat" available on the path '
                "(e.g. /usr/bin)."
            )
            sys.exit(1)

        print(result[-1])
