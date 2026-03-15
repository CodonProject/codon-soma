import torch
from typing import List, Tuple

from soma.core.base import BasicAgent, RigidBody
from soma.envs.objects import ExplicitState


def get_nearby_entities_with_states(
    agent: BasicAgent,
    entities: List[RigidBody],
    observation_range: float
) -> Tuple[List[RigidBody], List[dict], List[torch.Tensor]]:
    '''
    Returns entities within observation range along with their states and positions.

    This is an optimized function that calculates distance only once per entity,
    combining the functionality of get_nearby_entities and collect_explicit_states.

    Args:
        agent (BasicAgent): The agent performing the observation.
        entities (List[RigidBody]): List of entities to check.
        observation_range (float): Maximum distance for sensing.

    Returns:
        Tuple[List[RigidBody], List[dict], List[torch.Tensor]]: A tuple containing:
            - List of entities within observation range
            - List of explicit state data dictionaries
            - List of relative position tensors
    '''
    nearby_entities = []
    explicit_states = []
    positions = []

    for entity in entities:
        delta = entity._position - agent._position
        distance = torch.norm(delta).item()
        if distance <= observation_range:
            nearby_entities.append(entity)
            if hasattr(entity, 'get_explicit_state'):
                entity_state = entity.get_explicit_state()
                explicit_states.append(entity_state.data)
            positions.append(delta.clone())

    return nearby_entities, explicit_states, positions
