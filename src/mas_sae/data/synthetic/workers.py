"""Client ownership shared by the generation and dataset-audit runners."""
import threading
from typing import Any, Callable


class WorkerClients:
    """Lazily create one client per worker; an injected serial client is borrowed."""
    def __init__(self, factory: Callable[[], Any] | None, client: Any = None):
        self.factory = factory
        self.borrowed = client
        self.local = threading.local()
        self.owned: list[Any] = []
        self.lock = threading.Lock()

    def get(self):
        if self.borrowed is not None:
            return self.borrowed
        if not hasattr(self.local, "client"):
            self.local.client = self.factory()
            with self.lock:
                self.owned.append(self.local.client)
        return self.local.client

    def close(self) -> list[BaseException]:
        errors = []
        for client in self.owned:
            try:
                client.close()
            except BaseException as exc:
                errors.append(exc)
        return errors
