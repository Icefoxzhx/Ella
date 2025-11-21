from vico.agents import AgentProcess

def get_agent_cls(agent_type):
    if agent_type == 'ella' or agent_type == 'ella_seg':
        from .ella import EllaAgent
        return EllaAgent
    if agent_type == 'generative_agent' or agent_type == 'generative_agent_seg':
        from .gen_agent import GenAgent
        return GenAgent
    if agent_type == 'no_proximity':
        from .ella_no_proximity import EllaNoProximityAgent
        return EllaNoProximityAgent
    if agent_type == 'no_image':
        from .ella_no_image import EllaNoImageAgent
        return EllaNoImageAgent
    if agent_type == 'no_recency':
        from .ella_no_recency import EllaNoRecencyAgent
        return EllaNoRecencyAgent
    else:
        raise NotImplementedError(f"agent type {agent_type} is not supported")