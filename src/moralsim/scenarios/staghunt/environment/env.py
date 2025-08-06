from omegaconf import DictConfig
from ..persona.cognition import utils as cognition_utils
from .utils import (
    prompt_round_payoffs_base,
    prompt_universalization_base,
)
from moralsim.scenarios.common.environment import MoralityPerturbationEnv
from moralsim.scenarios.common.persona.persona import ActionObs
from moralsim.persona.common import PersonaEvent
from dateutil.relativedelta import relativedelta
import numpy as np

import logging

logger = logging.getLogger(__name__)


class StagHuntPerturbationEnv(MoralityPerturbationEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str], seed: int,
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name, seed)
        self.suckers_payoff_factor = 0
        self.defect_base = 1 / self.cfg.defect_base

    def _prompt_round_payoffs(self, agent: str) -> str:
        chosen_action = self.internal_global_state["chosen_action"][agent]
        round_payoff = self.internal_global_state["round_payoff"][agent]
        both_defect = self.internal_global_state["both_defect"]
        input_total = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        if "sh_base" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_base
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_round_payoffs_fn(self.agents, self.agent_id_to_name, self.internal_global_state["chosen_action"], round_payoff, both_defect, input_total, agent, self.min_payoff_to_survive, self.defect_base)

    def _prompt_universalization(self, agent: str) -> str:
        round_input = self.internal_global_state["last_collected_resource"][agent]
        if "sh_base" in self.cfg.name.lower():
            prompt_universalization_fn = prompt_universalization_base
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_universalization_fn(self.cfg.inject_universalization, round_input, self.defect_base, self.suckers_payoff_factor)
    
    def _prompt_opponent_action_report(self):
        pass

    def _agent_no_survival(self, agent: str) -> bool:
        return self.min_payoff_to_survive is not None and self.internal_global_state["round_payoff"][agent] < self.min_payoff_to_survive

    def _apply_perturbations(self) -> None:
        if self.perturbation is None:
            return
        if self.num_round == self.perturbation.round:
            if self.perturbation.type == "low_inputs_season":
                self._generate_inputs = self._generate_inputs_low
                self.perturbation_round = self.perturbation.round
                cognition_utils.MIN_PAYOFF_TO_SURVIVE = self.perturbation.min_payoff_to_survive
                self.min_payoff_to_survive = self.perturbation.min_payoff_to_survive
            logger.info(f"Apply perturbation: {self.perturbation.type}")
    
    def _generate_inputs_stochastic(self, *, min_input=None, max_input=None, seed_dec=0) -> None:
        if min_input is None:
            min_input = self.cfg.min_input
            max_input = self.cfg.max_input + 1 # inclusive
        rs = np.random.RandomState(self.seed * 1000 + self.num_round - seed_dec) # different earnings for each seed
        resource_per_agent = rs.randint(min_input, max_input)

        for agent in self.agents:
            res = resource_per_agent
            self.internal_global_state["collected_resource"][agent] += res
            self.internal_global_state["last_collected_resource"][agent] = res
    
    def _generate_inputs_low(self) -> None:
        """Low earnings: If other agent defects and you cooperate, you don't survive."""
        self._generate_inputs_stochastic(min_input=self.cfg.min_input, max_input=int(1 / self.suckers_payoff_factor) * self.min_payoff_to_survive - 1, seed_dec=self.perturbation_round)

    def _assign_resource(self) -> None:
        chosen_actions = self.internal_global_state["chosen_action"]
        # 1: Cooperate, 2: Defect
        input = self.internal_global_state["last_collected_resource"][self.agents[0]]
        equal_dist = False
        sum_actions = sum(list(chosen_actions.values()))
        if (both_defect := sum_actions == 2 * len(self.agents)):
            equal_dist = True
            input = self.defect_base * input
        elif sum_actions == len(self.agents):
            equal_dist = True
        self.internal_global_state["both_defect"] = both_defect
        for agent in self.agents:
            res = 0
            if equal_dist:
                res = input / len(self.agents)
            elif chosen_actions[agent] == 1:
                res = self.suckers_payoff_factor * input
            else:
                res = (1 - self.suckers_payoff_factor) * input
            
            self.log_round(self.internal_global_state["action"][agent], self.internal_global_state["last_collected_resource"][agent], res)

            self.internal_global_state["acc_payoff"][agent] += res
            self.internal_global_state["round_payoff"][agent] = res

            self.rewards[agent] += res

    def _observe_pre_round(self, agent: str) -> ActionObs:
        if self._agent_selector.is_first():
            self._generate_inputs()
        obs = ActionObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=[],
            context=self._prompt_universalization(agent),
            agent_resource_num=self.internal_global_state["last_collected_resource"][agent],
        )
        return obs

    def _observe_office_after_round(self, agent: str) -> ActionObs:
        obs = ActionObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=[
                PersonaEvent(
                    self._prompt_round_payoffs(agent),
                    created=self.internal_global_state["next_time"][agent] + relativedelta(days=1),
                    expiration=self._get_expiration_month(
                        self.internal_global_state["next_time"][agent]
                    ),
                    always_include=True,
                )
            ],
            context="",
            agent_resource_num={agent: 0 for agent in self.agents},
        )
        return obs
