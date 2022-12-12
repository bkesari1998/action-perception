from pysat.formula import CNF
import tempfile
import os
from typing import List

def parse_result(results_str: List[str]):
    """
    Parses the result of cmsgen.
    """
    results_rep = []
    for solution in results_str:
        vars = solution.split(" ")
        arr_curr_result = [int(var) for var in vars]
        results_rep.append(arr_curr_result)
    return results_rep


def cmsgen_solve(cnf: CNF, num_samples: int):
    """
    Solves a CNF formula using cmsgen.
    Returns a list of solutions, each solution is a list of variables.
    If the formula is unsatisfiable, returns an empty list.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cnf.to_file(tmpdir + "/cnf.cnf")
        os.system(
            f"cd {tmpdir}; cmsgen cnf.cnf --samples {num_samples} --samplefile cnf.out"
        )
        with open(tmpdir + "/cnf.out") as f:
            result = f.readlines()
            return parse_result(result)


if __name__ == "__main__":
    cnf = CNF(from_file="test/test.cnf")
    result = cmsgen_solve(cnf, 10)
    print(result)
