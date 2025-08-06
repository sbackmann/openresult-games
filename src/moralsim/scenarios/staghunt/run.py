import os
from typing import List

from omegaconf import DictConfig, OmegaConf

from moralsim.persona import EmbeddingModel
from moralsim.utils import ModelWandbWrapper

from .environment import StagHuntPerturbationEnv
from .persona import StagHuntPersona, StagHuntDummyPersona
from .persona.cognition import utils as cognition_utils
from ..common.run_utils import init_all_personas, init_utils, run_step


def run(
    cfg: DictConfig,
    mix_llm: list[dict],
    wandb_logger: ModelWandbWrapper,
    wrappers: List[ModelWandbWrapper],
    framework_wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
    seed: int,
):
    default_options = {
        "model_path": ("model_name", "1")
    }
    init_utils(cfg, mix_llm)
    num_personas = cfg.personas.num

    personas = {}
    for i in range(num_personas):
        personas.update({
            f"persona_{i}": StagHuntPersona(
                cfg.agent,
                wrappers[i],
                framework_wrapper,
                embedding_model,
                os.path.join(experiment_storage, f"persona_{i}"),
                cfg.scenario,
            ) if (actions := cfg.personas[f"persona_{i}"].actions) is None
            else StagHuntDummyPersona(cfg.agent, wrappers[i], actions, cfg.env.max_num_rounds)
        })
        # cognition_utils.DEFAULT_OPTIONS.update({f"persona_{i}": default_options[wrappers[i]]})

    agent_name_to_id, agent_id_to_name = init_all_personas(personas, num_personas, cfg)

    env = StagHuntPerturbationEnv(cfg.env, experiment_storage, agent_id_to_name, seed=seed)

    agent_id, obs = env.reset(seed=seed)
    has_next_step = True
    while has_next_step:
        agent = personas[agent_id]
        has_next_step, agent_id, obs = run_step(env, agent, obs, num_personas, wandb_logger)

        if has_next_step:
            wandb_logger.save(experiment_storage, agent_name_to_id)

    env.save_log()
    for persona in personas:
        if isinstance(personas[persona], StagHuntPersona):
            personas[persona].memory.save()
