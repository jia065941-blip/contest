"""One numerical inference session, with no domain observations or actions."""

from ..runtime import Runtime


class InferenceSkill(Runtime):
    """Public session API: open(), infer(named_arrays), close(), or a context manager.

    All lifecycle, ownership and validation behavior lives in Runtime. A skill
    instance owns one backend instance; this release does not share resources.
    """
