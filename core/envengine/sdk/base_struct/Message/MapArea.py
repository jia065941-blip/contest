from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MapArea(object):
    lonMin: float = field(default=0.0)
    latMin: float = field(default=0.0)
    lonMax: float = field(default=0.0)
    latMax: float = field(default=0.0)
