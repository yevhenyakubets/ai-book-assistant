import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(
    auto_error=False,
    description="The access token the API owner gave you.",
)


def require_access_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    """Reject requests without the configured token; do nothing if none is set."""
    expected = request.app.state.access_token
    if expected is None:
        return
    supplied = credentials.credentials if credentials else ""
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
