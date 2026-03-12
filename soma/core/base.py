import torch
import torch.nn as nn

from .event import EventHook, Event
from typing import Union, Any, Optional, Dict, List

from dataclasses import dataclass, field


@dataclass
class Observation:
    '''
    Data class representing an agent's observation of the environment.

    This class encapsulates sensor data and other information gathered
    by an agent during the observe phase.

    Attributes:
        data (Dict[str, torch.Tensor]): Dictionary mapping observation keys to tensor values.
    '''
    data: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Desire:
    '''
    Data class representing an agent's internal desire or goal state.

    This class encapsulates the agent's internal motivations and goals
    generated during the think phase.

    Attributes:
        data (Dict[str, torch.Tensor]): Dictionary mapping desire keys to tensor values.
    '''
    data: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Action:
    '''
    Data class representing an agent's action to be executed.

    This class encapsulates the action commands generated during
    the act phase to be applied to the environment.

    Attributes:
        data (Dict[str, torch.Tensor]): Dictionary mapping action keys to tensor values.
    '''
    data: Dict[str, torch.Tensor] = field(default_factory=dict)


class RigidBody:
    '''
    Base class for all physical entities in the environment.

    This class encapsulates common physical properties such as position,
    velocity, mass, and collision radius for physics simulation.

    Attributes:
        _position (torch.Tensor): The position of the rigid body.
        _velocity (torch.Tensor): The velocity of the rigid body.
        _mass (float): The mass of the rigid body.
        _collision_radius (float): The collision radius for detection.
        _is_movable (bool): Whether the rigid body can be moved.
    '''

    def __init__(
        self,
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        mass: float = 1.0,
        collision_radius: float = 1.0,
        is_movable: bool = True
    ) -> None:
        '''
        Initializes the RigidBody with physical properties.

        Args:
            position (Optional[torch.Tensor]): Initial position, defaults to origin.
            velocity (Optional[torch.Tensor]): Initial velocity, defaults to zero.
            mass (float): Mass of the rigid body.
            collision_radius (float): Radius for collision detection.
            is_movable (bool): Whether the body can be moved by physics.
        '''
        self._position = position if position is not None else torch.zeros(2)
        self._velocity = velocity if velocity is not None else torch.zeros(2)
        self._mass = mass
        self._collision_radius = collision_radius
        self._is_movable = is_movable

    def update_position(self, new_position: torch.Tensor) -> None:
        '''
        Updates the position of the rigid body.

        Args:
            new_position (torch.Tensor): The new position value.
        '''
        if self._is_movable:
            self._position = new_position

    def update_velocity(self, new_velocity: torch.Tensor) -> None:
        '''
        Updates the velocity of the rigid body.

        Args:
            new_velocity (torch.Tensor): The new velocity value.
        '''
        if self._is_movable:
            self._velocity = new_velocity

    def to_dict(self) -> Dict[str, Any]:
        '''
        Converts the rigid body properties to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing rigid body properties.
        '''
        return {
            'position': self._position.tolist(),
            'velocity': self._velocity.tolist(),
            'mass': self._mass,
            'collision_radius': self._collision_radius,
            'is_movable': self._is_movable
        }


class Obstacle(RigidBody):
    '''
    A static obstacle in the environment.

    This class represents immovable obstacles that agents must avoid.
    It inherits from RigidBody with zero velocity and is_movable set to False.

    Attributes:
        Inherits all attributes from RigidBody.
    '''

    def __init__(
        self,
        position: Optional[torch.Tensor] = None,
        mass: float = float('inf'),
        collision_radius: float = 1.0
    ) -> None:
        '''
        Initializes the Obstacle with position and size.

        Args:
            position (Optional[torch.Tensor]): Position of the obstacle.
            mass (float): Mass of the obstacle, defaults to infinity.
            collision_radius (float): Radius for collision detection.
        '''
        super().__init__(
            position=position,
            velocity=torch.zeros(2),
            mass=mass,
            collision_radius=collision_radius,
            is_movable=False
        )


class BasicEnvironment:
    '''
    Base class for all environments.

    This class manages agents, obstacles, and hooks for event handling.
    It provides the main simulation loop with physics integration.

    Attributes:
        _hooks (list[EventHook]): List of event hooks.
        _agents (list[BasicAgent]): List of agents in the environment.
        _obstacles (list[Obstacle]): List of obstacles in the environment.
        _active_agent (Optional[BasicAgent]): Currently active agent.
        _step_count (int): Current simulation step count.
    '''

    def __init__(self) -> None:
        '''
        Initializes the BasicEnvironment.
        '''
        self._hooks: List['EventHook'] = []
        self._agents: List['BasicAgent'] = []
        self._obstacles: List['Obstacle'] = []
        self._active_agent: Optional['BasicAgent'] = None
        self._step_count = 0

    def setup(self) -> None:
        '''
        Sets up the environment by firing the setup event hook.

        This method should be called after all agents and hooks are added
        to initialize the environment state.
        '''
        self._fire_hook('setup')

    def _fire_hook(self, event_name: str, agent: Optional['BasicAgent'] = None) -> None:
        '''
        Fires an event hook by calling the corresponding method on all registered hooks.

        Args:
            event_name (str): The name of the event to fire (e.g., 'setup', 'step_start').
            agent (Optional[BasicAgent]): The agent associated with the event, if any.
        '''
        method_name = f'on_{event_name}'
        for hook in self._hooks:
            method = getattr(hook, method_name, None)
            if method and callable(method):
                method(Event(
                    env=self,
                    agent=agent
                ))

    def add_hook(self, hook: Union['EventHook', List['EventHook']]) -> 'BasicEnvironment':
        '''
        Adds event hook(s) to the environment.

        Args:
            hook (Union[EventHook, List[EventHook]]): Event hook or list of hooks to add.

        Returns:
            BasicEnvironment: Self for method chaining.
        '''
        if isinstance(hook, list):
            self._hooks.extend(hook)
        else:
            self._hooks.append(hook)
        return self

    def add_agent(self, agent: Union['BasicAgent', List['BasicAgent']]) -> 'BasicEnvironment':
        '''
        Adds agent(s) to the environment.

        Args:
            agent (Union[BasicAgent, List[BasicAgent]]): Agent or list of agents to add.

        Returns:
            BasicEnvironment: Self for method chaining.
        '''
        if isinstance(agent, list):
            self._agents.extend(agent)
        else:
            self._agents.append(agent)
        return self

    def add_obstacle(self, obstacle: Union['Obstacle', List['Obstacle']]) -> 'BasicEnvironment':
        '''
        Adds obstacle(s) to the environment.

        Args:
            obstacle (Union[Obstacle, List[Obstacle]]): Obstacle or list of obstacles to add.

        Returns:
            BasicEnvironment: Self for method chaining.
        '''
        if isinstance(obstacle, list):
            self._obstacles.extend(obstacle)
        else:
            self._obstacles.append(obstacle)
        return self

    def step(self) -> None:
        '''
        Executes one simulation step in the environment.

        This method iterates through all agents, allowing each to observe,
        think, and act. After all agents have acted, a physics step is
        performed and the step counter is incremented.
        '''
        self._fire_hook('step_start')

        for agent in self._agents:
            self._active_agent = agent
            action = agent.step(self)
            self._apply_action(agent, action)
            self._active_agent = None

        self._physics_step()
        self._step_count += 1

        self._fire_hook('step_end')

    def _physics_step(self) -> None:
        '''
        Performs physics simulation for one time step.

        This method should be overridden by subclasses to implement
        custom physics behavior such as collision detection and response.
        '''
        pass

    def _apply_action(self, agent: 'BasicAgent', action: 'Action') -> None:
        '''
        Applies an agent's action to the environment.

        This method should be overridden by subclasses to implement
        action execution logic specific to the environment.

        Args:
            agent (BasicAgent): The agent performing the action.
            action (Action): The action to apply.
        '''
        pass

    def to_dict(self) -> Dict[str, Any]:
        '''
        Converts the environment state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing environment state including
                step count and obstacle information.
        '''
        return {
            'step_count': self._step_count,
            'obstacles': [obs.to_dict() for obs in self._obstacles]
        }


class BasicAgent(RigidBody):
    '''
    Base class for all agents in the environment.

    This class inherits from RigidBody, providing physical properties
    along with cognitive capabilities (observe, think, act).

    Attributes:
        _mind (BasicMind): The mind component for decision making.
        _uid (str): Unique identifier for the agent.
        Inherits all attributes from RigidBody.
    '''

    def __init__(
        self,
        mind: 'BasicMind',
        uid: Optional[str] = None,
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        mass: float = 1.0,
        collision_radius: float = 1.0
    ) -> None:
        '''
        Initializes the BasicAgent with mind and physical properties.

        Args:
            mind (BasicMind): The mind component for decision making.
            uid (Optional[str]): Unique identifier for the agent.
            position (Optional[torch.Tensor]): Initial position.
            velocity (Optional[torch.Tensor]): Initial velocity.
            mass (float): Mass of the agent.
            collision_radius (float): Radius for collision detection.
        '''
        super().__init__(
            position=position,
            velocity=velocity,
            mass=mass,
            collision_radius=collision_radius,
            is_movable=True
        )
        self._mind: 'BasicMind' = mind
        self._uid = uid

    def step(self, env: 'BasicEnvironment') -> Action:
        '''
        Executes one cognitive cycle: observe, think, and act.

        This method orchestrates the agent's decision-making process by
        gathering observations, processing them through the mind, and
        generating an action.

        Args:
            env (BasicEnvironment): The environment the agent is operating in.

        Returns:
            Action: The action to be executed in the environment.
        '''
        env._fire_hook('observe_start', self)
        obs = self.observe(env)
        env._fire_hook('observe_end', self)

        env._fire_hook('think_start', self)
        desire = self.think(obs)
        env._fire_hook('think_end', self)

        env._fire_hook('act_start', self)
        action = self.act(desire)
        env._fire_hook('act_end', self)
        return action

    def observe(self, env: BasicEnvironment) -> Observation:
        '''
        Gathers observations from the environment.

        This method should be overridden by subclasses to implement
        specific observation logic based on the agent's sensors.

        Args:
            env (BasicEnvironment): The environment to observe.

        Returns:
            Observation: The gathered observation data.

        Raises:
            NotImplementedError: If not overridden by subclass.
        '''
        raise NotImplementedError()

    def think(self, obs: Observation) -> Desire:
        '''
        Processes observations to generate internal desires.

        This method delegates to the mind component for decision making.

        Args:
            obs (Observation): The observation to process.

        Returns:
            Desire: The generated desire or goal state.
        '''
        return self._mind.forward(obs=obs)

    def act(self, des: Desire) -> Action:
        '''
        Converts desires into executable actions.

        This method should be overridden by subclasses to implement
        specific action generation logic.

        Args:
            des (Desire): The desire to convert into an action.

        Returns:
            Action: The generated action.

        Raises:
            NotImplementedError: If not overridden by subclass.
        '''
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        '''
        Converts the agent state to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing agent state including
                rigid body properties and unique identifier.
        '''
        rigid_body_dict = super().to_dict()
        rigid_body_dict['uid'] = self._uid
        return rigid_body_dict


class BasicMind:
    '''
    Base class for agent decision-making components.

    This class encapsulates the neural network model and provides
    the interface for processing observations into desires.

    Attributes:
        _model (nn.Module): The neural network model for decision making.
    '''

    def __init__(self, model: nn.Module) -> None:
        '''
        Initializes the BasicMind with a neural network model.

        Args:
            model (nn.Module): The neural network model to use for decision making.
        '''
        self._model = model

    def forward(self, obs: Observation) -> Desire:
        '''
        Processes an observation to generate a desire.

        This method should be overridden by subclasses to implement
        specific decision-making logic.

        Args:
            obs (Observation): The observation to process.

        Returns:
            Desire: The generated desire.

        Raises:
            NotImplementedError: If not overridden by subclass.
        '''
        raise NotImplementedError()