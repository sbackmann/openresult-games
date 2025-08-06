import logging

from moralsim.persona import (
    PerceiveComponent,
    RetrieveComponent,
)
from moralsim.persona.common import (
    PersonaAction,
    PersonaActionChoice,
)
from moralsim.persona.embedding_model import EmbeddingModel
from moralsim.persona.memory import AssociativeMemory
from moralsim.scenarios.common.environment import ActionObs
from moralsim.scenarios.common.persona import MoralityDummyPersona, MoralityPersona
from moralsim.scenarios.common.persona.cognition import (
    MoralityPlanComponent,
    MoralityReflectComponent,
    MoralityStoreComponent
)
from moralsim.utils import ModelWandbWrapper

from .cognition import (
    ChickenDummyActComponent,
    ChickenActComponent,
)
from .cognition.utils import (
    get_decision_prompt_base,
    get_system_prompt_base,
)

logger = logging.getLogger(__name__)


class ChickenDummyPersona(MoralityDummyPersona):
    last_collected_resource_num: int

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        actions: str | list,
        max_num_rounds: int
    ):
        super().__init__(cfg, model)
        self.act = ChickenDummyActComponent(model, cfg, actions, max_num_rounds)
        self.act.init_persona_ref(self)


class ChickenPersona(MoralityPersona):
    last_collected_resource_num: int
    other_personas: dict[str, "ChickenPersona"]

    act: ChickenActComponent

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        framework_model: ModelWandbWrapper,
        embedding_model: EmbeddingModel,
        base_path: str,
        scenario: str,
        memory_cls: type[AssociativeMemory] = AssociativeMemory,
        perceive_cls: type[PerceiveComponent] = PerceiveComponent,
        retrieve_cls: type[RetrieveComponent] = RetrieveComponent,
        store_cls: type[MoralityStoreComponent] = MoralityStoreComponent,
        reflect_cls: type[MoralityReflectComponent] = MoralityReflectComponent,
        plan_cls: type[MoralityPlanComponent] = MoralityPlanComponent,
        act_cls: type[ChickenActComponent] = ChickenActComponent,
    ) -> None:
        if scenario == "ch_base":
            system_prompt_fn = get_system_prompt_base
            decision_prompt_fn = get_decision_prompt_base
            self.reflect_focus = "payoff"
        else:
            raise ValueError(f"Unexpected scenario name: {scenario}.")
        super().__init__(
            cfg,
            model,
            framework_model,
            embedding_model,
            base_path,
            system_prompt_fn,
            decision_prompt_fn,
            memory_cls,
            perceive_cls,
            retrieve_cls,
            store_cls,
            reflect_cls,
            plan_cls,
            act_cls,
        )
        self.model_wrapper = model
        print(self.cfg)

    def loop(self, obs: ActionObs) -> PersonaAction:
        res = []
        self.current_time = obs.current_time 

        self.perceive.perceive(obs)
        if obs.current_location == "office" and obs.phase == "office":
            retrieved_memory = self.retrieve.retrieve([obs.current_location], 10)
            num_resource, html_interactions = self.act.choose_action(
                retrieved_memory,
                obs.current_location,
                obs.current_time,
                obs.context,
            )
            action = PersonaActionChoice(
                (self.agent_id, "agent"),
                "office",
                num_resource,
                stats={
                    f"{self.agent_id}_input_value": obs.agent_resource_num,
                    f"{self.agent_id}_chosen_action": num_resource,
                },
                html_interactions=html_interactions,
            )
        elif obs.current_location == "office" and obs.phase == "office_after_round":
            # dummy action to register observation
            action = PersonaAction((self.agent_id, "agent"), "office")
        elif obs.current_location == "home":
            self.reflect.run([self.reflect_focus], self.cfg.reflect.exclude_thoughts_from_prompt)
            action = PersonaAction((self.agent_id, "agent"), "home")

        self.memory.save()
        return action
