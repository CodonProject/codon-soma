import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from soma.core.base import *


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


def circle_circle_collision(
    body1: 'RigidBody',
    body2: 'RigidBody'
) -> Tuple[bool, torch.Tensor, float]:
    '''
    Checks for collision between two circular rigid bodies.

    Args:
        body1 (RigidBody): The first rigid body.
        body2 (RigidBody): The second rigid body.

    Returns:
        Tuple[bool, torch.Tensor, float]: A tuple containing:
            - is_colliding (bool): Whether the bodies are colliding.
            - normal_vector (torch.Tensor): The collision normal pointing from body1 to body2.
            - penetration_depth (float): The depth of penetration (positive if colliding).
    '''
    delta = body2._position - body1._position
    distance = torch.norm(delta)
    min_distance = body1._collision_radius + body2._collision_radius

    if distance < 1e-8:
        normal = torch.tensor([1.0, 0.0])
        penetration = min_distance
        return True, normal, penetration

    normal = delta / distance
    penetration = min_distance - distance.item()
    is_colliding = penetration > 0

    return is_colliding, normal, penetration


def resolve_collision(
    body1: 'RigidBody',
    body2: 'RigidBody',
    normal: torch.Tensor,
    penetration: float
) -> None:
    '''
    Resolves collision between two rigid bodies by separating them and updating velocities.

    This implements elastic collision with mass-based velocity exchange.

    Args:
        body1 (RigidBody): The first rigid body.
        body2 (RigidBody): The second rigid body.
        normal (torch.Tensor): The collision normal pointing from body1 to body2.
        penetration (float): The depth of penetration.
    '''
    total_mass = body1._mass + body2._mass

    if body1._is_movable and body2._is_movable:
        ratio1 = body2._mass / total_mass
        ratio2 = body1._mass / total_mass
        body1._position = body1._position - normal * penetration * ratio1
        body2._position = body2._position + normal * penetration * ratio2
    elif body1._is_movable:
        body1._position = body1._position - normal * penetration
    elif body2._is_movable:
        body2._position = body2._position + normal * penetration
    else:
        return

    relative_velocity = body1._velocity - body2._velocity
    velocity_along_normal = torch.dot(relative_velocity, normal)

    if velocity_along_normal > 0:
        return

    restitution = 1.0
    impulse_magnitude = -(1 + restitution) * velocity_along_normal
    impulse_magnitude /= (1.0 / body1._mass + 1.0 / body2._mass)

    impulse = impulse_magnitude * normal

    if body1._is_movable:
        body1._velocity = body1._velocity + impulse / body1._mass
    if body2._is_movable:
        body2._velocity = body2._velocity - impulse / body2._mass


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
        mode_map = {'bounce': 0.0, 'clamp': 1.0, 'wrap': 2.0}
        self.explicit_state.data['boundary_mode'] = torch.tensor([mode_map.get(self.boundary_mode, 0.0)])

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
        '''
        for agent in self._agents:
            for obstacle in self._obstacles:
                is_colliding, normal, penetration = circle_circle_collision(agent, obstacle)
                if is_colliding:
                    resolve_collision(agent, obstacle, normal, penetration)

    def _handle_agent_agent_collisions(self) -> None:
        '''
        Handles collisions between agents.
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
                self._apply_bounce_boundary(agent)
            elif self.boundary_mode == 'clamp':
                self._apply_clamp_boundary(agent)
            elif self.boundary_mode == 'wrap':
                self._apply_wrap_boundary(agent)

    def _apply_bounce_boundary(self, agent: BasicAgent) -> None:
        '''
        Applies bounce boundary conditions to an agent.

        When an agent hits a boundary, its velocity component is reflected.

        Args:
            agent (BasicAgent): The agent to apply boundary conditions to.
        '''
        min_x = agent._collision_radius
        max_x = self.width - agent._collision_radius
        min_y = agent._collision_radius
        max_y = self.height - agent._collision_radius

        position = agent._position.clone()
        velocity = agent._velocity.clone()

        if position[0] < min_x:
            position[0] = min_x
            velocity[0] = abs(velocity[0])
        elif position[0] > max_x:
            position[0] = max_x
            velocity[0] = -abs(velocity[0])

        if position[1] < min_y:
            position[1] = min_y
            velocity[1] = abs(velocity[1])
        elif position[1] > max_y:
            position[1] = max_y
            velocity[1] = -abs(velocity[1])

        agent.update_position(position)
        agent.update_velocity(velocity)

    def _apply_clamp_boundary(self, agent: BasicAgent) -> None:
        '''
        Applies clamp boundary conditions to an agent.

        When an agent hits a boundary, it stops at the boundary and
        the velocity component in that direction is set to zero.

        Args:
            agent (BasicAgent): The agent to apply boundary conditions to.
        '''
        min_x = agent._collision_radius
        max_x = self.width - agent._collision_radius
        min_y = agent._collision_radius
        max_y = self.height - agent._collision_radius

        position = agent._position.clone()
        velocity = agent._velocity.clone()

        if position[0] < min_x:
            position[0] = min_x
            velocity[0] = 0.0
        elif position[0] > max_x:
            position[0] = max_x
            velocity[0] = 0.0

        if position[1] < min_y:
            position[1] = min_y
            velocity[1] = 0.0
        elif position[1] > max_y:
            position[1] = max_y
            velocity[1] = 0.0

        agent.update_position(position)
        agent.update_velocity(velocity)

    def _apply_wrap_boundary(self, agent: BasicAgent) -> None:
        '''
        Applies wrap boundary conditions to an agent.

        When an agent crosses a boundary, it teleports to the opposite side.

        Args:
            agent (BasicAgent): The agent to apply boundary conditions to.
        '''
        position = agent._position.clone()

        if position[0] < 0:
            position[0] = self.width
        elif position[0] > self.width:
            position[0] = 0.0

        if position[1] < 0:
            position[1] = self.height
        elif position[1] > self.height:
            position[1] = 0.0

        agent.update_position(position)

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

    def to_dict(self) -> dict:
        '''
        Converts the environment state to a dictionary.

        Returns:
            dict: Dictionary containing environment state including dimensions,
                boundary mode, time step, and base environment state.
        '''
        base_dict = super().to_dict()
        base_dict['width'] = self.width
        base_dict['height'] = self.height
        base_dict['boundary_mode'] = self.boundary_mode
        base_dict['dt'] = self.dt
        return base_dict


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
                        weight = 1.0 / (distance.item() + 1e-6)
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
        angle = torch.rand(1).item() * 2 * 3.141592653589793
        return torch.tensor([torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))])


class Space2DAgent(BasicAgent):
    '''
    An agent for the Space2D environment with observation and action capabilities.

    This agent can observe nearby obstacles and agents within its observation range
    and convert desires into actions for movement in the 2D space.

    Attributes:
        observation_range (float): The maximum distance for sensing nearby entities.
        explicit_state (ExplicitState): External observable state of the agent.
        implicit_state (ImplicitState): Internal hidden state of the agent.
        Inherits all attributes from BasicAgent.
    '''

    def __init__(
        self,
        mind: BasicMind,
        observation_range: float = 10.0,
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

    def observe(self, env: Space2D) -> Observation:
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

        nearby_obstacles_states = []
        nearby_obstacles_positions = []
        for obstacle in env._obstacles:
            delta = obstacle._position - self._position
            distance = torch.norm(delta).item()
            if distance <= self.observation_range:
                if hasattr(obstacle, 'get_explicit_state'):
                    obs_state = obstacle.get_explicit_state()
                    nearby_obstacles_states.append(obs_state.data)
                nearby_obstacles_positions.append(delta.clone())
        if nearby_obstacles_positions:
            data['nearby_obstacles_positions'] = torch.stack(nearby_obstacles_positions)
        else:
            data['nearby_obstacles_positions'] = torch.zeros(0, 2)
        if nearby_obstacles_states:
            data['nearby_obstacles_states'] = nearby_obstacles_states

        nearby_agents_states = []
        nearby_agents_positions = []
        for agent in env._agents:
            if agent._uid == self._uid:
                continue
            delta = agent._position - self._position
            distance = torch.norm(delta).item()
            if distance <= self.observation_range:
                if hasattr(agent, 'get_explicit_state'):
                    agent_state = agent.get_explicit_state()
                    nearby_agents_states.append(agent_state.data)
                nearby_agents_positions.append(delta.clone())
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
                max_speed = torch.norm(self._velocity).item() if torch.norm(self._velocity) > 0 else 1.0
                action_data['velocity'] = normalized_direction * max_speed
            else:
                action_data['velocity'] = torch.zeros(2)
        elif 'force' in des.data:
            action_data['force'] = des.data['force']
        elif 'velocity' in des.data:
            action_data['velocity'] = des.data['velocity']

        return Action(data=action_data)
