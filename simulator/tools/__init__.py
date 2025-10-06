from typing import Protocol


class Tool(Protocol):
    def main_gui(self):
        raise NotImplementedError

    def sidebar_gui(self, width: float):
        pass

    def switched_away(self):
        pass
