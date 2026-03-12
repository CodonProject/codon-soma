from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING: from .base import BasicEnvironment, BasicAgent

from dataclasses import dataclass

@dataclass
class Event:
    env: 'BasicEnvironment'
    agent: Optional['BasicAgent']


class EventHook:
    def on_setup(self, event: Event): pass
    def on_reset(self, event: Event): pass

    def on_step_start(self, event: Event): pass
    def on_step_end(self, event: Event): pass
    
    def on_observe_start(self, event: Event): pass
    def on_observe_end(self, event: Event): pass
    
    def on_think_start(self, event: Event): pass
    def on_think_end(self, event: Event): pass
    
    def on_act_start(self, event: Event): pass
    def on_act_end(self, event: Event): pass