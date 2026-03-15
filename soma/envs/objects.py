import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from soma.core.base import Obstacle


@dataclass
class ExplicitState:
    '''
    External observable state that can be perceived by other entities.

    This class encapsulates the observable properties of an entity
    that other agents can sense and react to.

    Attributes:
        data (Dict[str, torch.Tensor]): Dictionary mapping state keys to tensor values.
    '''
    data: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        '''
        Converts the explicit state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing state values as lists.
        '''
        return {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in self.data.items()}


@dataclass
class ImplicitState:
    '''
    Internal hidden state that is private to the entity.

    This class encapsulates the internal properties of an entity
    that are not directly observable by other agents.

    Attributes:
        data (Dict[str, torch.Tensor]): Dictionary mapping state keys to tensor values.
    '''
    data: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        '''
        Converts the implicit state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing state values as lists.
        '''
        return {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in self.data.items()}


class Space2DObstacle(Obstacle):
    '''
    A static obstacle in the Space2D environment with explicit state.

    This class represents immovable obstacles that agents must avoid,
    with support for external observable state.

    Attributes:
        explicit_state (ExplicitState): External observable state of the obstacle.
        implicit_state (ImplicitState): Internal hidden state of the obstacle.
        Inherits all attributes from Obstacle.
    '''

    def __init__(
        self,
        position: Optional[torch.Tensor] = None,
        collision_radius: float = 1.0,
        obstacle_type: str = 'default'
    ) -> None:
        '''
        Initializes the Space2DObstacle with position, size, and type.

        Args:
            position (Optional[torch.Tensor]): Position of the obstacle.
            collision_radius (float): Radius for collision detection.
            obstacle_type (str): Type identifier for the obstacle.
        '''
        super().__init__(
            position=position,
            collision_radius=collision_radius
        )
        self.obstacle_type = obstacle_type
        self.explicit_state = ExplicitState(data={
            'position': self._position.clone(),
            'collision_radius': torch.tensor([self._collision_radius]),
            'obstacle_type': torch.tensor([hash(obstacle_type) % 1000])
        })
        self.implicit_state = ImplicitState(data={})

    def update_explicit_state(self) -> None:
        '''
        Updates the explicit state based on current obstacle properties.
        '''
        self.explicit_state.data['position'] = self._position.clone()
        self.explicit_state.data['collision_radius'] = torch.tensor([self._collision_radius])
        self.explicit_state.data['obstacle_type'] = torch.tensor([hash(self.obstacle_type) % 1000])

    def get_explicit_state(self) -> ExplicitState:
        '''
        Returns the current explicit state of the obstacle.

        Returns:
            ExplicitState: The observable state of the obstacle.
        '''
        self.update_explicit_state()
        return self.explicit_state
