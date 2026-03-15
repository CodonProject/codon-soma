from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING: from .base import BasicEnvironment, BasicAgent

from dataclasses import dataclass


@dataclass
class Event:
    '''
    Data class representing an event in the environment.

    This class encapsulates the context of an event, including the environment
    and optionally the agent associated with the event.

    Attributes:
        env (BasicEnvironment): The environment where the event occurred.
        agent (Optional[BasicAgent]): The agent associated with the event, if any.
    '''
    env: 'BasicEnvironment'
    agent: Optional['BasicAgent']


class EventHook:
    '''
    Base class for event hooks that respond to environment events.

    Subclasses can override specific methods to react to different events
    in the simulation lifecycle.

    Methods:
        on_setup: Called when the environment is set up.
        on_reset: Called when the environment is reset.
        on_step_start: Called at the beginning of each simulation step.
        on_step_end: Called at the end of each simulation step.
        on_observe_start: Called when an agent starts observing.
        on_observe_end: Called when an agent finishes observing.
        on_think_start: Called when an agent starts thinking.
        on_think_end: Called when an agent finishes thinking.
        on_act_start: Called when an agent starts acting.
        on_act_end: Called when an agent finishes acting.
    '''

    def on_setup(self, event: Event) -> None:
        '''
        Handles the setup event.

        Args:
            event (Event): The setup event containing environment reference.
        '''
        pass

    def on_reset(self, event: Event) -> None:
        '''
        Handles the reset event.

        Args:
            event (Event): The reset event containing environment reference.
        '''
        pass

    def on_step_start(self, event: Event) -> None:
        '''
        Handles the step start event.

        Args:
            event (Event): The step start event containing environment reference.
        '''
        pass

    def on_step_end(self, event: Event) -> None:
        '''
        Handles the step end event.

        Args:
            event (Event): The step end event containing environment reference.
        '''
        pass

    def on_observe_start(self, event: Event) -> None:
        '''
        Handles the observe start event.

        Args:
            event (Event): The observe start event containing agent reference.
        '''
        pass

    def on_observe_end(self, event: Event) -> None:
        '''
        Handles the observe end event.

        Args:
            event (Event): The observe end event containing agent reference.
        '''
        pass

    def on_think_start(self, event: Event) -> None:
        '''
        Handles the think start event.

        Args:
            event (Event): The think start event containing agent reference.
        '''
        pass

    def on_think_end(self, event: Event) -> None:
        '''
        Handles the think end event.

        Args:
            event (Event): The think end event containing agent reference.
        '''
        pass

    def on_act_start(self, event: Event) -> None:
        '''
        Handles the act start event.

        Args:
            event (Event): The act start event containing agent reference.
        '''
        pass

    def on_act_end(self, event: Event) -> None:
        '''
        Handles the act end event.

        Args:
            event (Event): The act end event containing agent reference.
        '''
        pass
