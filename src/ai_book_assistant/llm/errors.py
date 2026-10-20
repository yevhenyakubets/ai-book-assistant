class LLMError(Exception):
    """The language model could not be reached or refused the request.

    status_code is the HTTP status when there was one (for example 401 for a
    bad key, 429 for rate limiting), and None for connection problems.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
