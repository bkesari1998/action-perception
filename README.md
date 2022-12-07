# dnns_project

A framework for an agent to solve a task using symbolic planning without knowing any details of its initial state.

## Installation

Clone this repository to your local machine. <br />

Create a new virtual conda environment and install the required packages: <br />

`conda env create --file=environment.yaml` <br />

Activate the environment: <br />
`conda activate dnns_project`

Clone the ppdlgym repository (https://github.com/bkesari1998/pddlgym) to your local machine at the same directory level as this repository. <br />

Install the pddlgym repository: <br />
`cd pddlgym` <br />
`pip install --editable .` <br />