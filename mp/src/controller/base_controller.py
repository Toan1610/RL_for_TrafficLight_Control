from abc import ABC, abstractmethod

class BaseController:
    def __init__(self, tls_id: str, iface, **params):
        self.tls_id = tls_id
        self.cfg = params
        self.iface = iface
    # def on_reset(self, obs: TLSObservation) -> None: ...
    def on_close(self) -> None: ...

    @abstractmethod
    def action(self, t) -> float:
        pass

    @abstractmethod
    def start(self) -> None:
        pass