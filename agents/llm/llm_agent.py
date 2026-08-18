"""Generic LLM agent; economic role logic lives in adapters."""

import json
import os

import numpy as np
import torch

from .econ_adapter import EconAdapter
from .json_io import parse_json
from .llm_client import create_llm_client
from .monetary_policy import validate_fast_policy, validate_slow_policy


class llm_agent:
    def __init__(self, envs, args, type=None, agent_name="government"):
        self.adapter = EconAdapter(envs, agent_name, type)
        self.agent_name, self.agent_type = agent_name, type
        self.llm_name, self.on_policy = args.llm_name, True
        self.action_dim = self.adapter.entity.action_space.shape[-1]
        self.last_output = self.last_decision = self.last_error = None
        self.decision_interval = max(1, int(getattr(args, "llm_decision_interval", 1)))
        self.log_decisions = bool(getattr(args, "llm_log_decisions", False))
        self._step, self._cached_action = 0, None
        self.slow_policy = self.previous_obs = self.previous_decision = None
        self._next_slow_review = None
        self.mock_response = os.environ.get("LLM_MOCK_RESPONSE")
        self.mock_slow_response = os.environ.get("LLM_MOCK_SLOW_RESPONSE")

        self.client = None if self.mock_response else create_llm_client(args)

    def get_action(self, obs_tensor):
        obs = obs_tensor.cpu().numpy() if isinstance(obs_tensor, torch.Tensor) else np.asarray(obs_tensor)
        direct = self.adapter.direct_action(obs)
        if direct is not None:
            return direct
        if self.adapter.uses_slow_policy and (
            self.slow_policy is None or self._step == self._next_slow_review
        ):
            self.slow_policy = self._generate_slow_policy(obs)
            self._next_slow_review = self._step + self.slow_policy["review_interval"]
            self._cached_action = None
            if self.log_decisions:
                print("\nLLM slow policy:")
                print(json.dumps(self.slow_policy, ensure_ascii=False, indent=2))
        if self._cached_action is not None and self._step % self.decision_interval:
            self._step += 1
            return self._cached_action.copy()

        try:
            if self.mock_response:
                output = self.mock_response
            else:
                prompt = self.adapter.prompt(
                    obs, self.slow_policy, self.previous_obs, self.previous_decision
                )
                request = {
                    "model": self.llm_name,
                    "messages": [{"role": "user", "content": prompt}],
                    **self.adapter.request_options(self.llm_name, stage="fast"),
                }
                output = self.client.chat.completions.create(**request).choices[0].message.content
            self.last_output, self.last_decision = output, parse_json(output)
            try:
                self._cached_action = self._validate_and_map(self.last_decision, obs)
            except ValueError as error:
                if self.mock_response:
                    raise
                request["messages"] += [
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": self.adapter.repair_prompt(error)},
                ]
                output = self.client.chat.completions.create(**request).choices[0].message.content
                self.last_output, self.last_decision = output, parse_json(output)
                self._cached_action = self._validate_and_map(self.last_decision, obs)
            self.previous_obs, self.previous_decision = obs.copy(), self.last_decision
            if self.log_decisions:
                print(f"\nLLM decision [{self.agent_name}.{self.agent_type}]:")
                print(json.dumps(self.last_decision, ensure_ascii=False, indent=2))
                print(f"EconGym action: {self._cached_action.tolist()}")
        except Exception as error:
            self.last_error = error
            print(f"LLM {self.agent_name} failed; using fallback: {error}")
            self._cached_action = self.adapter.fallback(obs)
        self._step += 1
        return self._cached_action.copy()

    def _validate_and_map(self, decision, obs):
        if self.adapter.uses_slow_policy:
            validate_fast_policy(decision, self.slow_policy, obs)
        return self.adapter.action(decision, obs)

    def _generate_slow_policy(self, obs):
        try:
            if self.mock_slow_response:
                return validate_slow_policy(parse_json(self.mock_slow_response))
            if self.mock_response:
                return self.adapter.slow_fallback()
            request = {
                "model": self.llm_name,
                "messages": [{"role": "user", "content": self.adapter.slow_prompt(obs)}],
                **self.adapter.request_options(self.llm_name, stage="slow"),
            }
            output = self.client.chat.completions.create(**request).choices[0].message.content
            try:
                return validate_slow_policy(parse_json(output))
            except ValueError as error:
                request["messages"] += [
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": self.adapter.slow_repair_prompt(error)},
                ]
                output = self.client.chat.completions.create(**request).choices[0].message.content
                return validate_slow_policy(parse_json(output))
        except Exception as error:
            self.last_error = error
            print(f"LLM slow policy failed; using fallback: {error}")
            return self.adapter.slow_fallback()

    def reset_episode(self):
        self._step, self._cached_action = 0, None
        self.slow_policy = self.previous_obs = self.previous_decision = None
        self._next_slow_review = None

    def train(self, transition_dict):
        return 0.0, 0.0

    def save(self, dir_path):
        pass
