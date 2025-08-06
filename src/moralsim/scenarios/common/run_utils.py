import os
import sys
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from moralsim.persona import EmbeddingModel, PersonaAgent
from moralsim.persona.common import PersonaIdentity
from moralsim.utils import ModelWandbWrapper
from moralsim.scenarios.common import ActionObs
from .environment import MoralityPerturbationEnv
from .persona import MoralityDummyPersona
from .persona.cognition import utils as cognition_utils

def init_all_personas(
    personas: dict[str, MoralityDummyPersona | PersonaAgent],
    num_personas: int,
    cfg
) -> tuple[dict[str, str], dict[str, str]]:
    identities = {}
    for i in range(num_personas):
        persona_id = f"persona_{i}"
        identities[persona_id] = PersonaIdentity(
            agent_id=persona_id, **cfg.personas[persona_id]
        )

    # Standard setup
    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)

    for persona in personas:
        for other_persona in personas:
            # also add self reference, for conversation
            personas[persona].add_reference_to_other_persona(personas[other_persona])
    return agent_name_to_id, agent_id_to_name

def init_utils(cfg, mix_llm):
    cognition_utils.SYS_VERSION = cfg.agent.system_prompt
    
    if cfg.agent.cot_prompt == "think_step_by_step":
        cognition_utils.REASONING = "think_step_by_step"
    elif cfg.agent.cot_prompt == "deep_breath":
        cognition_utils.REASONING = "deep_breath"
    
    if cfg.env.min_payoff_to_survive is not None:
        cognition_utils.MIN_PAYOFF_TO_SURVIVE = cfg.env.min_payoff_to_survive
    cognition_utils.OTHER_PERSONAS = [cfg.personas[f"persona_{i}"].name for i in range(cfg.personas.num)]
    map_model_names = {
        "anthropic/claude-3.7-sonnet": "Claude-3.7-Sonnet",
        "deepseek/deepseek-r1": "DeepSeek-R1",
        "z-gpt-4o-2024-08-0": "GPT-4o",
        "z-gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B"
    }
    if cfg.agent.reveal_identity:
        print(mix_llm)
        cognition_utils.OTHER_MODELS = [map_model_names[llm.llm.path] for llm in mix_llm] if len(mix_llm) > 0 else []

def run_step(
    env: MoralityPerturbationEnv,
    agent: MoralityDummyPersona | PersonaAgent,
    obs: ActionObs,
    num_personas: int,
    wandb_logger: ModelWandbWrapper,
) -> tuple[bool, str, ActionObs]:
    has_next_step = True
    action = agent.loop(obs)

    (
        agent_id,
        obs,
        rewards,
        termination,
    ) = env.step(action)

    stats = {}
    STATS_KEYS = [
        "conversation_resource_limit",
        *[f"persona_{i}_input_value" for i in range(num_personas)],
        *[f"persona_{i}_chosen_action" for i in range(num_personas)],
    ]
    for s in STATS_KEYS:
        if s in action.stats:
            stats[s] = action.stats[s]

    if np.any(list(termination.values())):
        wandb_logger.log_game(
            {
                **stats,
            },
            last_log=True,
        )
        has_next_step = False
    else:
        wandb_logger.log_game(
            {
                **stats,
            }
        )
    return has_next_step, agent_id, obs
