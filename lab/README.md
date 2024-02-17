# Structure

The lab consists of three parts, each with two group tutorial sessions to be given by the TAs and one individual homework to be solved by students.

- Part 1: Horizontal Federated Learning
    - [Tutorial 1A - Centralized Learning Recap & Federated Learning Setup](lab-hfl.ipynb)
    - [Tutorial 1B - FedSGD/FedAvg & Result Overview](lab-hfl.ipynb)
    - [Homework 1 - HFL](hw-hfl.ipynb)
- Part 2: Vertical Federated Learning
    - Tutorial 2A - Classification
    - Tutorial 2B - Autoencoder
    - Homework 2 - VFL
- Part 3: Federated Learning Attacks & Defenses
    - Tutorial 3A - Attacks
    - Tutorial 3B - Defenses
    - Homework 3 - Security in FL

# Setup

All the tutorial and homework content is within the current directory of the repository. Homework submission happens via ILIAS, as a PDF answering the assignment question and a Jupyter Notebook containing the code needed to produce the answers.

All tutorials and assignments use the following structure and dependencies.

Steps to get up and running:
- Ensure [Python 3](https://www.python.org/downloads/) is installed (preferably 3.10 or higher, but older versions should work with at most small changes).
- Have a [VS Code](https://code.visualstudio.com/Download) installation with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extension packs (other IDEs like [PyCharm](https://www.jetbrains.com/pycharm/download/) also work, but this guide does not contain specific configuration files/steps for them).
- From the root directory of this repository, execute `python -m venv lab/.venv`, replacing `python` with `python3` if the initial command does not work.
- Open the repository in VS Code, create a new terminal (checking that the beginning of the line starts with `(.venv)` to signal the correct Python environment got activated), and execute `pip install -r lab/requirements.txt`.
