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

Install the pddlgym repository: <br />
`cd pddlgym` <br />
`pip install --editable .` <br />

Install minisat: <br />
Clone the patched minisat repository (https://github.com/agurfinkel/minisat) to your local machine. Switch to the directory: <br />
`cd minisat`

Set the location of installation: <br />
`export PREFIX=/usr` <br />
`make config prefix=$PREFIX`<br />

Compile and install: <br />
`make install`
