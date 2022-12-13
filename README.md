# dnns_project

A framework for an agent to solve a task using symbolic planning without knowing any details of its initial state.

## Installation

Clone this repository to your local machine. <br />

Create a new virtual conda environment and install the required packages: <br />

`conda create -n <environment_name>` <br />
`conda activate <environment_name>` <br />
`pip install pyperplan`
`pip install python_sat`

Clone the ppdlgym repository (https://github.com/bkesari1998/pddlgym) and the pddlgym_planners respository (https://github.com/bkesari1998/pddlgym_planners) to your local machine at the same directory level as this repository. <br />

### Install the pddlgym repository:
`cd pddlgym` <br />
`pip install --editable .` <br />

### Install cmsgen:
cmsgen is a SATSolver that can give out solutions to a SAT problem in a 
nearly uniformly random fashion.

Clone the patched minisat repository (https://github.com/meelgroup/cmsgen) to your local machine. 
Follow the installation instruction on their readme page.

After compiling cmsgen, make sure to add the cmsgen binary to your system PATH.
