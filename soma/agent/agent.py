import torch
from typing import Optional, List, TYPE_CHECKING

from soma.core.base import BasicAgent, BasicMind, Observation, Desire, Action
from soma.envs.objects import ExplicitState, ImplicitState
from soma.agent.sensors import get_nearby_entities_with_states

if TYPE_CHECKING:
    from soma.envs.space2d import Space2D


class Space2DAgent(BasicAgent):
    '''
    An agent for the Space2D environment with observation and action capabilities.

    This agent can observe nearby obstacles and agents within its observation range
    and convert desires into actions for movement in the 2D space.

    Attributes:
        observation_range (float): The maximum distance for sensing nearby entities.
        max_speed (float): Maximum speed for velocity output.
        explicit_state (ExplicitState): External observable state of the agent.
        implicit_state (ImplicitState): Internal hidden state of the agent.
        Inherits all attributes from BasicAgent.
    '''

    def __init__(
        self,
        mind: BasicMind,
        observation_range: float = 10.0,
        max_speed: float = 1.0,
        uid: Optional[str] = None,
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        mass: float = 1.0,
        collision_radius: float = 1.0
    ) -> None:
        '''
        Initializes the Space2DAgent with mind and observation range.

        Args:
            mind (BasicMind): The mind component for decision making.
            observation_range (float): The maximum distance for sensing nearby entities.
            max_speed (float): Maximum speed for velocity output.
            uid (Optional[str]): Unique identifier for the agent.
            position (Optional[torch.Tensor]): Initial position.
            velocity (Optional[torch.Tensor]): Initial velocity.
            mass (float): Mass of the agent.
            collision_radius (float): Radius for collision detection.
        '''
        super().__init__(
            mind=mind,
            uid=uid,
            position=position,
            velocity=velocity,
            mass=mass,
            collision_radius=collision_radius
        )
        self.observation_range = observation_range
        self.max_speed = max_speed
        self.explicit_state = ExplicitState(data={
            'position': self._position.clone(),
            'velocity': self._velocity.clone(),
            'collision_radius': torch.tensor([self._collision_radius])
        })
        self.implicit_state = ImplicitState(data={})

    def update_explicit_state(self) -> None:
        '''
        Updates the explicit state based on current agent properties.
        '''
        self.explicit_state.data['position'] = self._position.clone()
        self.explicit_state.data['velocity'] = self._velocity.clone()
        self.explicit_state.data['collision_radius'] = torch.tensor([self._collision_radius])

    def get_explicit_state(self) -> ExplicitState:
        '''
        Returns the current explicit state of the agent.

        Returns:
            ExplicitState: The observable state of the agent.
        '''
        self.update_explicit_state()
        return self.explicit_state

    def observe(self, env: 'Space2D') -> Observation:
        '''
        Gathers observations from the Space2D environment.

        Collects explicit states from the environment, nearby obstacles,
        and nearby agents within observation range.

        Args:
            env (Space2D): The environment to observe.

        Returns:
            Observation: The gathered observation containing environment explicit state,
                nearby obstacles' explicit states, and nearby agents' explicit states.
        '''
        data = {}

        data['self_position'] = self._position.clone()
        data['self_velocity'] = self._velocity.clone()

        env_explicit = env.get_explicit_state()
        for key, value in env_explicit.data.items():
            data[f'env_{key}'] = value.clone() if isinstance(value, torch.Tensor) else value

        _, nearby_obstacles_states, nearby_obstacles_positions = get_nearby_entities_with_states(
            self, env._obstacles, self.observation_range
        )
        if nearby_obstacles_positions:
            data['nearby_obstacles_positions'] = torch.stack(nearby_obstacles_positions)
        else:
            data['nearby_obstacles_positions'] = torch.zeros(0, 2)
        if nearby_obstacles_states:
            data['nearby_obstacles_states'] = nearby_obstacles_states

        nearby_agents = [
            agent for agent in env._agents
            if agent._uid != self._uid
        ]
        _, nearby_agents_states, nearby_agents_positions = get_nearby_entities_with_states(
            self, nearby_agents, self.observation_range
        )
        if nearby_agents_positions:
            data['nearby_agents_positions'] = torch.stack(nearby_agents_positions)
        else:
            data['nearby_agents_positions'] = torch.zeros(0, 2)
        if nearby_agents_states:
            data['nearby_agents_states'] = nearby_agents_states

        return Observation(data=data)

    def act(self, des: Desire) -> Action:
        '''
        Converts a desire into an executable action.

        Processes the desire data to generate an action for movement.
        Supports 'target_direction', 'force', and 'velocity' keys in desire.

        Args:
            des (Desire): The desire to convert into an action.

        Returns:
            Action: The generated action for environment execution.
        '''
        action_data = {}

        if 'target_direction' in des.data:
            target_direction = des.data['target_direction']
            direction_norm = torch.norm(target_direction)
            if direction_norm > 1e-8:
                normalized_direction = target_direction / direction_norm
                action_data['velocity'] = normalized_direction * self.max_speed
            else:
                action_data['velocity'] = torch.zeros(2)
        elif 'force' in des.data:
            action_data['force'] = des.data['force']
        elif 'velocity' in des.data:
            action_data['velocity'] = des.data['velocity']

        return Action(data=action_data)
