"""Structured errors that never interpolate caller-provided tensor data."""

from typing import Any


class SkillGenerationError(Exception):
    def __init__(self, code: str, field: str, expected: Any = None, actual: Any = None):
        self.code = code
        self.field = field
        self.expected = expected
        self.actual = actual
        # Callers pass structural metadata here, never tensor contents.
        super().__init__(f"{type(self).__name__}: {code}; field={field!r}; "
                         f"expected={repr(expected)[:240]}; actual={repr(actual)[:240]}")


class ManifestError(SkillGenerationError):
    pass


class ArtifactError(SkillGenerationError):
    pass


class BackendError(SkillGenerationError):
    pass


class DependencyCompatibilityError(BackendError):
    def __init__(self, package: str, required: str, installed: str | None):
        self.package = package
        self.required = required
        self.installed = installed
        super().__init__("dependency_mismatch", f"dependencies.{package}",
                         required, installed if installed is not None else "not installed")

    def __str__(self) -> str:
        return (f"DependencyCompatibilityError\n\nPackage: {self.package}\n"
                f"Required: {self.required}\nInstalled: {self.installed or 'not installed'}\n\n"
                "Model bundle validation failed.")


class ValidationError(SkillGenerationError):
    pass


class StateError(SkillGenerationError):
    pass
