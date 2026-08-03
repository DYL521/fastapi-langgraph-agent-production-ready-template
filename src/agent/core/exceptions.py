"""Domain exceptions for the application.

These are transport-agnostic — they carry no HTTP status codes or framework
dependencies. The API layer maps them to appropriate HTTP responses.
"""


class NotFoundError(Exception):
    """Raised when a requested entity does not exist."""

    def __init__(self, entity: str, identifier: str):  # noqa: D107
        self.entity = entity
        self.identifier = identifier
        super().__init__(f"{entity} '{identifier}' not found")
