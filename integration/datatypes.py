import abc

from pydantic import BaseModel

from integration.abstract import Object, classproperty


class SimpleType(BaseModel):
    attr1: float
    attr2: str


class ComplexType(Object):
    @classproperty
    def class_method(cls) -> None:  # pylint: disable=no-self-argument
        pass

    @abc.abstractmethod
    def abstract_method(self) -> None:
        raise NotImplementedError()
