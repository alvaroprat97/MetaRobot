Probabilistic Embeddings for hybrid meta-Reinforcement and Imitation Learning

## Requirements

This project requires Python v3.6.x and pip3 to run.

## Setup

Requirements:
- Python 3
- pip3

1. Install dependencies

Dependency management is done through pip.

`pip3 install -r requirements.txt`

## Conditions
Before running PERIL, you must select the task families you want to meta-train on. Access this in `experiment_demos`.

Configurations for meta-training and meta-testing can be modified in `Configs/default_configs`

## Available Scripts
From the root directory, you can run:

`python3 -m experiment_demos` to run the PERIL training pipeline.

`python3 -m demos_tester` to test the algorithm on unseeen tasks.

## Documents
Please read our works on PERIL in the provided pdf document (XXXX). This document will be available after assessment of the project.


