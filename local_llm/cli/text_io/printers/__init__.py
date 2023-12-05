from abc import ABC, abstractmethod

class PrinterBase(ABC):

    @abstractmethod
    def print(self, text):
        pass