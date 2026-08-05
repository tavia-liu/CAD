"""Full-CAD detector plugin for AgentDojo.

Enable at benchmark time with:

    export AGENTDOJO_DEFENSE_PLUGINS=detector

Then run AgentDojo with `--defense full_cad`.
"""

from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig, register_defense

from detector.defense import FullCADDefense


def _build_full_cad_defense(config: PipelineConfig) -> FullCADDefense:
    return FullCADDefense()


register_defense("full_cad", _build_full_cad_defense)
