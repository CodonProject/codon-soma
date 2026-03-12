import torch
import torch.nn as nn

from .event import EventHook, Event
from typing import Union, Any, Optional, Dict

from dataclasses import dataclass, field

@dataclass
class Observation:
    data: Dict[str, torch.Tensor] = field(default_factory=dict)

@dataclass
class Desire:
    data: Dict[str, torch.Tensor] = field(default_factory=dict)

@dataclass
class Action:
    data: Dict[str, torch.Tensor] = field(default_factory=dict)


class BasicEnvironment:
    def __init__(self):
        self._hooks:  list['EventHook']  = []
        self._agents: list['BasicAgent'] = []
        # state
        self._active_agent: Optional['BasicAgent'] = None
        self._step_count = 0
    
    def setup(self) -> None:
        self._fire_hook('setup')

    def _fire_hook(self, event_name: str, agent: Optional['BasicAgent']=None) -> None:
        method_name = f"on_{event_name}"
        for hook in self._hooks:
            method = getattr(hook, method_name, None)
            if method and callable(method):
                method(Event(
                    env=self,
                    agent=agent
                ))

    def add_hook(self, hook: Union[EventHook, list[EventHook]]) -> 'BasicEnvironment':
        if isinstance(hook, list):
            self._hooks.extend(hook)
        else:
            self._hooks.append(hook)
        return self
    
    def add_agent(self, agent: Union['BasicAgent', list['BasicAgent']]) -> 'BasicEnvironment':
        if isinstance(agent, list):
            self._agents.extend(agent)
        else:
            self._agents.append(agent)
        return self

    def step(self) -> None:
        self._fire_hook('step_start')

        for agent in self._agents:
            self._active_agent = agent
            action = agent.step(self)
            self._apply_action(agent, action)
            self._active_agent = None

        self._physics_step()
        self._step_count += 1

        self._fire_hook('step_end')
    
    def _physics_step(self): pass

    def _apply_action(self, agent: 'BasicAgent', action: Action): pass

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_count': self._step_count
        }


class BasicAgent:
    def __init__(self, mind: 'BasicMind', uid: str=None):
        self._mind: 'BasicMind' = mind
        self._uid = uid
    
    def step(self, env: 'BasicEnvironment') -> Action:
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
        raise NotImplementedError()
    
    def think(self, obs: Observation) -> Desire:
        return self._mind.forward(obs=obs)

    def act(self, des: Desire) -> Action:
        raise NotImplementedError()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'uid': self._uid
        }


class BasicMind:
    def __init__(self, model: nn.Module):
        self._model = model

    def forward(self, obs: Observation) -> Desire:
        raise NotImplementedError()