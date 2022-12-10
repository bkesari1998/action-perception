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
import copy

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
    


def dump_formula(formula, filename):

    fl = open(filename, "w")
    
    for clause in formula:
        
        fl.write(str(clause))
        fl.write("\n")
    fl.close()

def translate_dimacs(dimacs,nums_to_vars):

    translated_str = ''
    for clause in dimacs:
        for var in clause:
            if abs(var) not in nums_to_vars:
                translated_str+=f"unknown var {var}, "
                continue

            if var>0:
                translated_str+=f"{nums_to_vars[var]}, "
            else:
                translated_str+=f"(not {nums_to_vars[abs(var)]}), "


        translated_str+="\n"

    return translated_str

def interpret_formula(formula,m,vars_to_nums, nums_to_vars):

    truths = [nums_to_vars[i] for i in m if i>0 and i in vars_to_nums.values()]
    falses = [nums_to_vars[abs(i)] for i in m if i<0 and abs(i) in vars_to_nums.values()]

def get_init_state_from_model(m, nums_to_vars):
    '''
    Extract init state from a model of the CNF formula
    '''
    init_state = []
    for var in m:
        if abs(var) not in nums_to_vars:
            continue
            
        if not nums_to_vars[abs(var)].endswith("-0"):
            continue
        
        if var>0:
            init_state.append(nums_to_vars[var])
        else:
            init_state.append(f"(not {nums_to_vars[abs(var)]})")

    
    return init_state
    
    
def free_goal_state(goal_var_names,cnf_formula,vars_to_nums):
    """
    This function should remove the clauses that enforce the goal state (single variable clauses in the CNF formula)
    """

    new_cnf_formula = []
    for gvar in goal_var_names:
        var_num = vars_to_nums[gvar]
        for clause in cnf_formula:
            if var_num in clause and len(clause)==1:
                continue
            new_cnf_formula.append(clause)

    return new_cnf_formula

def free_initial_state(cnf_formula,nums_to_vars):
    """
    This function should remove the clauses that enforce the initial state (single variable clauses in the CNF formula)
    """
    new_cnf_formula = []
    for clause in cnf_formula:
        if len(clause)==1: #single variable clause
            var = clause[0]
            if abs(var) in nums_to_vars: 
                name = nums_to_vars[abs(var)]
                if name.endswith("-0"): # for the initial state
                    continue

        new_cnf_formula.append(clause)

    return new_cnf_formula



def generate_formula(task,plan_horizon):
    """
    This function should generate the CNF formula for the given task and plan horizon
    """
    formula = get_plan_formula(task, plan_horizon)
    fml = copy.deepcopy(formula)

    writer = CnfWriter()
    vars_to_numbers = writer.write(formula)
    num_to_vars = {v: k for k, v in vars_to_numbers.items()}
    #print(num_to_vars)
    
    cnf_str = writer.get_cnf_str()
    
    dimacs_str = format_cnfstr_to_dimacs(cnf_str)

    cnf = CNF(from_string=cnf_str)

    return formula, cnf, vars_to_numbers, num_to_vars


if __name__ == "__main__":
    domain_file = os.path.dirname(__file__) + "/../../pddl/simple.pddl"
    problem_file = os.path.dirname(__file__) + "/../../pddl/simple/problem_0.pddl"
    
    # parse the problem and a task file.
    problem = _parse(domain_file, problem_file)
    task = _ground(problem)
    #also read and parse a problem file with empty init state (not really empty, only whatever predicates we perceive)

    # define the number of timesteps to plan for
    plan_horizon = 9

    # get the plan formula from the task
    formula = get_plan_formula(task, plan_horizon)
    fml = copy.deepcopy(formula)

    
    writer = CnfWriter()
    vars_to_numbers = writer.write(formula)
    num_to_vars = {v: k for k, v in vars_to_numbers.items()}
    #print(num_to_vars)
    
    cnf_str = writer.get_cnf_str()
    
    dimacs_str = format_cnfstr_to_dimacs(cnf_str)

    cnf = CNF(from_string=cnf_str)

    solver = Glucose3(bootstrap_with=cnf.clauses)
    with open("test.cnf", "w") as f:
        f.write(dimacs_str)

    
    
    
    isSAT = solver.solve()
    print(isSAT)
    if isSAT:
        #enumerate models
        #TODO: add assumptions based on action failure ..

        #TODO:collect models consistent with assumptions
        #get a model
        m = next(solver.enum_models())
        print(m)
        #TODO:extract init states from models
        #
    
    outfile = 'formula.txt'
    dump_formula(fml, outfile)
    interpret_formula(fml,m,vars_to_numbers, num_to_vars)

    goal_freed_formula = free_goal_state(["(at f5-4f)-9"],cnf.clauses,vars_to_numbers)
    open("translated_goal_free.txt",'w').write(translate_dimacs(goal_freed_formula, num_to_vars))

    init_and_goal_freed_formula = free_initial_state(goal_freed_formula,num_to_vars)
    open("translated_init_and_goal_free.txt",'w').write(translate_dimacs(init_and_goal_freed_formula, num_to_vars))
    
    
    #failed action: move-left at time 3

    formula,cnf,vars_to_nums,nums_to_vars = generate_formula(task,3) 
    # only need horizon 3 since that's where the failure happens

    freed_cnf_formula = free_goal_state(["(at f5-4f)-3"],cnf.clauses,vars_to_nums) 
    # free the goal state

    
    freed_cnf_formula = free_initial_state(freed_cnf_formula,nums_to_vars) 
    # free the initial state

    

    operators = task.operators

    possible_precondition_failures = []

    for op in operators:
        if op.name.startswith("(move_left"):
            
            for pre in op.preconditions:
                possible_precondition_failures.append(pre)

    #any of these preconditions could have failed
    print(possible_precondition_failures)
    #add step number of failure
    possible_precondition_failures = [p+"-3" for p in possible_precondition_failures]
    #translate to numbers and **negate** them
    possible_precondition_failures_nums = [-vars_to_nums[p] for p in possible_precondition_failures]

    print(possible_precondition_failures_nums)
    #This is now a clause that says that at least one of the possible precondition failures must have happened
    #add this to the formula
    freed_cnf_formula.append(possible_precondition_failures_nums)

    #Now lets get models that are consistent with this formula
    solver = Glucose3(bootstrap_with=freed_cnf_formula)
    for m in solver.enum_models():

        init_state = get_init_state_from_model(m,nums_to_vars)
        #keep_positives
        init_state = [s for s in init_state if not s.startswith("(not ")]

        print(init_state)
        


