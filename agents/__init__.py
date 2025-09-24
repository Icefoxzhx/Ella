from vico.agents import AgentProcess

def get_agent_cls(agent_type):
    if agent_type == 'ella':
        from .ella import EllaAgent
        return EllaAgent
    if agent_type == 'generative_agent':
        from .gen_agent import GenAgent
        return GenAgent
    else:
        raise NotImplementedError(f"agent type {agent_type} is not supported")