from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MapArea(object):
    xMin: float = field(default=0.0)
    xMax: float = field(default=0.0)
    yMin: float = field(default=0.0)
    yMax: float = field(default=0.0)
