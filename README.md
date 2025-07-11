# Langchain + Langgraph implementation

> [!NOTE]  
> Last update 7-10-2025

Different agent and LLM aplications using lang libraries and tools to create workflows and agent tools

## Features

- [agents](agents) different agent / LLM aplication files with example from tutorial and custom implementation
- Terminal based UI
- In-core build tools, agents and workflows (langgraph graphs)
- Checkpointer memory
- Supervisor agent architecture

## Setup

1. Get the necessary dependencies (use python venv / toml)
2. Create .env file to set the environment variables for OCI setup (also mutable to other LLM providers given API key)
    - Ensure to modify the file [yaml](agents/src/config/hw_agent.yaml) to add routes and variables
3. Find agent to test in the ```agents``` folder, copy the code and run from ```test.py```
4. Latest and best version in ```chain_app.py```

## Basic walkthrough

- [Best version](chain_app.py)
- [Architecture](walkthrough/Agent_architecture_FULL.png)
- [Demo video](walkthrough/Supervisor_Demo.mp4)