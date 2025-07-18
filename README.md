# Langchain + Langgraph implementation

> [!NOTE]  
> Last update 7-18-2025

AI agent with supervisor architecture traced using lagfuse on VM deployment

## Features

- [supervisor](main.py) main file to run the latest supervisor model with two worker agents
- [agents](test) different agent / LLM aplication files with example from tutorial and custom implementation
- Terminal based UI
- In-core build tools, agents and workflows (langgraph)
- Checkpointer memory

## Setup

1. Get the necessary dependencies (use python venv / toml)
2. Create .env file to set the environment variables for OCI setup (also mutable to other LLM providers given API key)
    - Ensure to modify the file [yaml](agents/src/config/hw_agent.yaml) to add routes and variables
3. Connect the langfuse services in [fuse_config](src/modules/fuse_config.py) for real-time trace
4. Ensure the ```stream``` method is running the right configurable form the fuse_config
5. Interact in Command line and read tracing in langfuse server (Cloud, local, VM services options)

> [!IMPORTANT]
> To run tracing example, switch to cloud mode in lagfuse. 
> Create account, project and API keys in the oficial langfuse documentation
> Add the keys in the .env file and change the fuse_configuration constructor
> Possible to connect with a local server / VM server using compartiments

## Basic walkthrough

- [Supervisor tracing](main.py)
- [Architecture](walkthrough/Agent_architecture_FULL.png)