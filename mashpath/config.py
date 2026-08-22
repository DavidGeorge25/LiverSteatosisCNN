"""The root config object the CLI builds.

One class composes the shared core sections with every feature's own section.
That keeps a single YAML file able to describe a whole run -- which is what
`configs/tuned.yaml` and every `config_used.yaml` written so far already do --
while each feature module still only ever receives its own section
(`detect_fat` takes a `FatConfig`, never the root), so features stay decoupled
from each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.config import CoreConfig
from .features.ballooning.config import BallooningConfig
from .features.inflammation.config import InflammationConfig
from .features.steatosis.config import FatConfig


@dataclass
class MashConfig(CoreConfig):
    fat: FatConfig = field(default_factory=FatConfig)
    ballooning: BallooningConfig = field(default_factory=BallooningConfig)
    # `InflammationConfig` itself, not a hand-copied subset. The subset that
    # used to live here listed only nuclei/classify/nucleus_qc, so the
    # `portal:` and `foci:` blocks were invisible to the top-level CLI -- and
    # because `_apply_dict` rejects unknown keys, adding either to a YAML would
    # have hard-failed every command that loads this class.
    inflammation: InflammationConfig = field(default_factory=InflammationConfig)


# The pre-restructure name. Kept so `config_used.yaml` round-trips and so any
# notebook doing `from mashpath.config import Config` keeps working.
Config = MashConfig
