import torch
from typing import Dict, Any

from soma.core.base import BasicEnvironment, BasicAgent, Action
from soma.envs.physics import (
    circle_circle_collision,
    resolve_collision,
    apply_bounce_boundary,
    apply_clamp_boundary,
    apply_wrap_boundary
)
from soma.envs.objects import ExplicitState, ImplicitState, Space2DObstacle


class Space2D(BasicEnvironment):
    '''
    A 2D space environment with physics simulation.

    This environment simulates agents and obstacles in a bounded 2D space
    with collision detection and response, and boundary enforcement.

    Attributes:
        width (float): The width of the environment.
        height (float): The height of the environment.
        boundary_mode (str): The boundary handling mode ('bounce', 'clamp', 'wrap').
        dt (float): The time step for physics simulation.
        explicit_state (ExplicitState): External observable state of the environment.
        implicit_state (ImplicitState): Internal hidden state of the environment.
    '''

    _BOUNDARY_MODE_MAP = {'bounce': 0.0, 'clamp': 1.0, 'wrap': 2.0}

    def __init__(
        self,
        width: float,
        height: float,
        boundary_mode: str = 'bounce',
        dt: float = 0.1
    ) -> None:
        '''
        Initializes the Space2D environment.

        Args:
            width (float): The width of the environment.
            height (float): The height of the environment.
            boundary_mode (str): The boundary handling mode ('bounce', 'clamp', 'wrap').
            dt (float): The time step for physics simulation.
        '''
        super().__init__()
        self.width = width
        self.height = height
        self.boundary_mode = boundary_mode
        self.dt = dt
        self.explicit_state = ExplicitState(data={
            'bounds': torch.tensor([width, height]),
            'boundary_mode': torch.tensor([0.0])
        })
        self.implicit_state = ImplicitState(data={})

    def update_explicit_state(self) -> None:
        '''
        Updates the explicit state based on current environment properties.
        '''
        self.explicit_state.data['bounds'] = torch.tensor([self.width, self.height])
        self.explicit_state.data['boundary_mode'] = torch.tensor(
            [self._BOUNDARY_MODE_MAP.get(self.boundary_mode, 0.0)]
        )

    def get_explicit_state(self) -> ExplicitState:
        '''
        Returns the current explicit state of the environment.

        Returns:
            ExplicitState: The observable state of the environment.
        '''
        self.update_explicit_state()
        return self.explicit_state

    def _physics_step(self) -> None:
        '''
        Performs physics simulation for one time step.

        This method updates positions, handles collisions between agents and obstacles,
        collisions between agents, and enforces boundary conditions.
        '''
        for agent in self._agents:
            if agent._is_movable:
                new_position = agent._position + agent._velocity * self.dt
                agent.update_position(new_position)

        self._handle_agent_obstacle_collisions()
        self._handle_agent_agent_collisions()
        self._enforce_boundaries()

    def _handle_agent_obstacle_collisions(self) -> None:
        '''
        Handles collisions between agents and obstacles.

        Note:
            Current implementation uses O(n*m) brute-force collision detection.
            For large numbers of obstacles, consider implementing spatial hashing
            or grid-based partitioning for O(n*log(m)) performance.
        '''
        for agent in self._agents:
            for obstacle in self._obstacles:
                is_colliding, normal, penetration = circle_circle_collision(agent, obstacle)
                if is_colliding:
                    resolve_collision(agent, obstacle, normal, penetration)

    def _handle_agent_agent_collisions(self) -> None:
        '''
        Handles collisions between agents.

        Note:
            Current implementation uses O(n²) brute-force collision detection.
            For large numbers of agents (> 50), consider implementing spatial hashing
            or grid-based partitioning for better performance.
        '''
        for i in range(len(self._agents)):
            for j in range(i + 1, len(self._agents)):
                agent1 = self._agents[i]
                agent2 = self._agents[j]
                is_colliding, normal, penetration = circle_circle_collision(agent1, agent2)
                if is_colliding:
                    resolve_collision(agent1, agent2, normal, penetration)

    def _enforce_boundaries(self) -> None:
        '''
        Enforces boundary conditions for all agents based on the boundary mode.
        '''
        for agent in self._agents:
            if not agent._is_movable:
                continue

            if self.boundary_mode == 'bounce':
                apply_bounce_boundary(agent, self.width, self.height)
            elif self.boundary_mode == 'clamp':
                apply_clamp_boundary(agent, self.width, self.height)
            elif self.boundary_mode == 'wrap':
                apply_wrap_boundary(agent, self.width, self.height)

    def _apply_action(self, agent: BasicAgent, action: Action) -> None:
        '''
        Applies an agent's action to the environment.

        Action data can contain 'force' or 'velocity' keys:
            - 'force': Applies force to the agent (velocity += force / mass * dt).
            - 'velocity': Directly sets the agent's velocity.

        Args:
            agent (BasicAgent): The agent performing the action.
            action (Action): The action to apply.
        '''
        if not agent._is_movable:
            return

        if 'force' in action.data:
            force = action.data['force']
            delta_velocity = force / agent._mass * self.dt
            new_velocity = agent._velocity + delta_velocity
            agent.update_velocity(new_velocity)

        if 'velocity' in action.data:
            new_velocity = action.data['velocity']
            agent.update_velocity(new_velocity)

    def to_dict(self) -> Dict[str, Any]:
        '''
        Converts the environment state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing environment state including dimensions,
                boundary mode, time step, and base environment state.
        '''
        base_dict = super().to_dict()
        base_dict['width'] = self.width
        base_dict['height'] = self.height
        base_dict['boundary_mode'] = self.boundary_mode
        base_dict['dt'] = self.dt
        return base_dict
