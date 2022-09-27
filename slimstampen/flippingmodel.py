"""flippingmodel"""
import slimstampen.spacingmodel as sp
from collections import namedtuple
from typing import Tuple

Fact = namedtuple("Fact",sp.Fact._fields + ("flipped",), defaults=(False,))
Response = sp.Response
Encounter = sp.Encounter

class FlippingModel(sp.SpacingModel):
    """
    An extension to the basic slimstampen spacingmodel that flips the question and answer of the individual facts in a regular order.
    """

    def get_next_fact(self, current_time: int) -> Tuple[Fact, bool]:
        """
        Returns a tuple containing the fact that needs to be repeated most urgently and a boolean indicating whether this fact is new (True) or has been presented before (False).
        If none of the previously studied facts needs to be repeated right now, return a new fact instead.
        When the flipping threshold of the chosen fact is reached before presenting it, it will be returned in reversed order.
        """
        next_fact, new = super().get_next_fact(current_time)

        if new:
            return next_fact, new

        new_fact = next_fact._replace(question=next_fact.answer, answer=next_fact.question, flipped=not next_fact.flipped)
        self.facts[self.facts.index(next_fact)] = new_fact

        return new_fact, new
