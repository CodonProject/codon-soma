import math
import torch
import torch.nn as nn
from typing import Optional

from soma.core.base import BasicMind, Observation, Desire


class Space2DMind(BasicMind):
    '''
    A mind component for agents in the Space2D environment.

    This class processes observations and generates desires for movement
    in a 2D space, with obstacle avoidance behavior.

    Attributes:
        _model (nn.Module): The neural network model for decision making.
        max_speed (float): Maximum speed for velocity output.
    '''

    def __init__(self, model: Optional[nn.Module] = None, max_speed: float = 1.0) -> None:
        '''
        Initializes the Space2DMind with optional model and max speed.

        Args:
            model (Optional[nn.Module]): The neural network model to use for decision making.
            max_speed (float): Maximum speed for velocity output.
        '''
        super().__init__(model=model if model is not None else nn.Identity())
        self.max_speed = max_speed

    def forward(self, obs: Observation) -> Desire:
        '''
        Processes an observation to generate a desire for movement.

        If model is None or Identity, uses simple heuristic to avoid obstacles
        by moving away from them. Otherwise, uses the model to process observation.

        Args:
            obs (Observation): The observation containing obstacle and position data.

        Returns:
            Desire: The generated desire with 'velocity' key for movement direction.
        '''
        if isinstance(self._model, nn.Identity):
            velocity = self._heuristic_forward(obs)
        else:
            velocity = self._model_forward(obs)

        return Desire(data={'velocity': velocity})

    def _heuristic_forward(self, obs: Observation) -> torch.Tensor:
        '''
        Generates velocity using simple obstacle avoidance heuristic.

        Moves away from nearby obstacles or in a random direction if no obstacles.

        Args:
            obs (Observation): The observation containing obstacle data.

        Returns:
            torch.Tensor: The velocity vector with shape (2,).
        '''
        agent_position = obs.data.get('self_position', torch.zeros(2))

        if 'nearby_obstacles_positions' in obs.data:
            obstacle_positions = obs.data['nearby_obstacles_positions']

            if obstacle_positions.numel() > 0:
                avoidance_direction = torch.zeros(2)
                for i in range(obstacle_positions.shape[0]):
                    relative_pos = obstacle_positions[i]
                    obstacle_pos = agent_position + relative_pos
                    direction = agent_position - obstacle_pos
                    distance = torch.norm(relative_pos)
                    if distance > 1e-8:
                        weight = 1.0 / (distance + 1e-6)
                        avoidance_direction = avoidance_direction + direction / distance * weight

                if torch.norm(avoidance_direction) > 1e-8:
                    velocity = avoidance_direction / torch.norm(avoidance_direction) * self.max_speed
                else:
                    velocity = self._random_direction() * self.max_speed
            else:
                velocity = self._random_direction() * self.max_speed
        else:
            velocity = self._random_direction() * self.max_speed

        return velocity

    def _model_forward(self, obs: Observation) -> torch.Tensor:
        '''
        Generates velocity using the neural network model.

        Args:
            obs (Observation): The observation to process.

        Returns:
            torch.Tensor: The velocity vector with shape (2,).
        '''
        obs_tensor = self._prepare_observation(obs)
        output = self._model(obs_tensor)
        velocity = output[:2] * self.max_speed
        return velocity

    def _prepare_observation(self, obs: Observation) -> torch.Tensor:
        '''
        Prepares observation data as input tensor for the model.

        Args:
            obs (Observation): The observation to prepare.

        Returns:
            torch.Tensor: Flattened observation tensor.
        '''
        tensors = []
        for key in sorted(obs.data.keys()):
            value = obs.data[key]
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    tensors.append(value.unsqueeze(0))
                else:
                    tensors.append(value.flatten())
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    for state_dict in value:
                        for state_key in sorted(state_dict.keys()):
                            state_value = state_dict[state_key]
                            if isinstance(state_value, torch.Tensor):
                                if state_value.dim() == 0:
                                    tensors.append(state_value.unsqueeze(0))
                                else:
                                    tensors.append(state_value.flatten())
        return torch.cat(tensors) if tensors else torch.zeros(1)

    def _random_direction(self) -> torch.Tensor:
        '''
        Generates a random unit direction vector.

        Returns:
            torch.Tensor: A random unit vector with shape (2,).
        '''
        angle = torch.rand(1).item() * 2 * math.pi
        return torch.tensor([math.cos(angle), math.sin(angle)])
