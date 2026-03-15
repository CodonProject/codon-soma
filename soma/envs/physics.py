import torch
from typing import Tuple

from soma.core.base import RigidBody


_DEFAULT_NORMAL = torch.tensor([1.0, 0.0])


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
        normal = _DEFAULT_NORMAL.clone()
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


def apply_bounce_boundary(
    agent: 'BasicAgent',
    width: float,
    height: float
) -> None:
    '''
    Applies bounce boundary conditions to an agent.

    When an agent hits a boundary, its velocity component is reflected.

    Args:
        agent (BasicAgent): The agent to apply boundary conditions to.
        width (float): The width of the environment.
        height (float): The height of the environment.
    '''
    min_x = agent._collision_radius
    max_x = width - agent._collision_radius
    min_y = agent._collision_radius
    max_y = height - agent._collision_radius

    position = agent._position.clone()
    velocity = agent._velocity.clone()

    if position[0] < min_x:
        position[0] = min_x
        velocity[0] = torch.abs(velocity[0])
    elif position[0] > max_x:
        position[0] = max_x
        velocity[0] = -torch.abs(velocity[0])

    if position[1] < min_y:
        position[1] = min_y
        velocity[1] = torch.abs(velocity[1])
    elif position[1] > max_y:
        position[1] = max_y
        velocity[1] = -torch.abs(velocity[1])

    agent.update_position(position)
    agent.update_velocity(velocity)


def apply_clamp_boundary(
    agent: 'BasicAgent',
    width: float,
    height: float
) -> None:
    '''
    Applies clamp boundary conditions to an agent.

    When an agent hits a boundary, it stops at the boundary and
    the velocity component in that direction is set to zero.

    Args:
        agent (BasicAgent): The agent to apply boundary conditions to.
        width (float): The width of the environment.
        height (float): The height of the environment.
    '''
    min_x = agent._collision_radius
    max_x = width - agent._collision_radius
    min_y = agent._collision_radius
    max_y = height - agent._collision_radius

    position = agent._position.clone()
    velocity = agent._velocity.clone()

    if position[0] < min_x:
        position[0] = min_x
        velocity[0] = torch.tensor(0.0)
    elif position[0] > max_x:
        position[0] = max_x
        velocity[0] = torch.tensor(0.0)

    if position[1] < min_y:
        position[1] = min_y
        velocity[1] = torch.tensor(0.0)
    elif position[1] > max_y:
        position[1] = max_y
        velocity[1] = torch.tensor(0.0)

    agent.update_position(position)
    agent.update_velocity(velocity)


def apply_wrap_boundary(
    agent: 'BasicAgent',
    width: float,
    height: float
) -> None:
    '''
    Applies wrap boundary conditions to an agent.

    When an agent crosses a boundary, it teleports to the opposite side.

    Args:
        agent (BasicAgent): The agent to apply boundary conditions to.
        width (float): The width of the environment.
        height (float): The height of the environment.
    '''
    position = agent._position.clone()

    if position[0] < 0:
        position[0] = width
    elif position[0] > width:
        position[0] = 0.0

    if position[1] < 0:
        position[1] = height
    elif position[1] > height:
        position[1] = 0.0

    agent.update_position(position)
