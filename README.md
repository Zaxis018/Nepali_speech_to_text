# Nepali Speech to Text Translation (ASR) System

## Introduction
## Goals
## Contributors
## Project Architecture


# Status
## Known Issue
## High Level Next Steps


# Usage
### Inference
1) clone the repository
- `https://github.com/fuseai-fellowship/Nepali-Speech-to-Text-Translation.git`
2) change to source directory
- `cd src/`
3) run
-`streamlit run app.py`

## Installation
To begin this project, use the included `Makefile`

#### Creating Virtual Environment

This package is built using `python-3.8`. 
We recommend creating a virtual environment and using a matching version to ensure compatibility.


#### pip-tools

The method of managing dependencies in this package is using `pip-tools`. To begin, run `make use-pip-tools` to install. 

Then when adding a new package requirement, update the `requirements.in` file with 
the package name. You can include a specific version if desired but it is not necessary. 

To install and use the new dependency you can run `make deps-install` or equivalently `make`

If you have other packages installed in the environment that are no longer needed, you can you `make deps-sync` to ensure that your current development environment matches the `requirements` files. 

## Usage Instructions


# Data Source
Refere to the [dataset readme](./dataset/README.md) for details on the dataset, sources, usablility and the link to data.

## Code Structure
## Artifacts Location
HuggingFace Demo: https://huggingface.co/spaces/kshitizzzzzzz/NEPALI_ASR_Whisper_Small

# Results
## Metrics Used
## Evaluation Results
