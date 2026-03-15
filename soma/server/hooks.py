import asyncio
from typing import Dict, Any

from soma.core.event import Event, EventHook
from soma.server.app import broadcast_state


class VisualizerHook(EventHook):
    '''
    Event hook that bridges the environment with the visualization server.

    This hook captures environment state at key events and broadcasts
    the state to all connected WebSocket clients for real-time visualization.

    Attributes:
        _initialized (bool): Whether the hook has been initialized.
    '''

    def __init__(self) -> None:
        '''
        Initializes the VisualizerHook.
        '''
        self._initialized = False

    def on_setup(self, event: Event) -> None:
        '''
        Handles the setup event for initialization.

        Broadcasts initial environment state when the environment is set up.

        Args:
            event (Event): The setup event containing environment reference.
        '''
        self._initialized = True
        state = self._collect_state(event)
        self._try_broadcast(state)

    def on_step_end(self, event: Event) -> None:
        '''
        Handles the step end event to broadcast updated state.

        Collects and broadcasts the current environment state after each step.

        Args:
            event (Event): The step end event containing environment reference.
        '''
        state = self._collect_state(event)
        self._try_broadcast(state)

    def _try_broadcast(self, state: Dict[str, Any]) -> None:
        '''
        Attempts to broadcast state to connected clients.

        Safely handles the case when no event loop is running.

        Args:
            state (Dict[str, Any]): The state dictionary to broadcast.
        '''
        try:
            asyncio.get_running_loop()
            asyncio.create_task(broadcast_state(state))
        except RuntimeError:
            pass

    def _collect_state(self, event: Event) -> Dict[str, Any]:
        '''
        Collects environment state for broadcasting.

        Gathers agent positions, obstacle positions, and general environment
        information into a dictionary suitable for JSON serialization.

        Args:
            event (Event): The event containing environment reference.

        Returns:
            Dict[str, Any]: Dictionary containing the collected state including
                agents, obstacles, and environment metadata.
        '''
        env = event.env

        agents_data = []
        for agent in env._agents:
            agent_dict = agent.to_dict()
            agents_data.append(agent_dict)

        obstacles_data = []
        for obstacle in env._obstacles:
            obstacle_dict = obstacle.to_dict()
            obstacles_data.append(obstacle_dict)

        env_dict = env.to_dict()

        state = {
            'environment': env_dict,
            'agents': agents_data,
            'obstacles': obstacles_data
        }

        return state
