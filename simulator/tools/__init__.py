from typing import Protocol


class Tool(Protocol):
    def main_gui(self):
        raise NotImplementedError

    def sidebar_gui(self):
        pass

    def switched_away(self):
        pass
