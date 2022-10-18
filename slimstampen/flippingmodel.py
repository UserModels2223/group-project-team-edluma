"""flippingmodel"""
from math import floor
import slimstampen.spacingmodel as sp
from collections import namedtuple
from typing import Tuple, List, Union
import pandas as pd
import io
import random

Fact = namedtuple("Fact", sp.Fact._fields + ("flipped",), defaults=(False,))
Response = sp.Response
Encounter = sp.Encounter


class FlippingModel(sp.SpacingModel):
    """
    An extension to the basic slimstampen spacingmodel that flips the question and answer of the individual facts in a regular order.
    """

    FLIPPING_THRESHOLD = -0.75
    FLIPPING_ALPHA = 0.3

    def get_next_fact(self, current_time: int, use_flipping: bool = True) -> Tuple[Fact, bool]:
        """
        Returns a tuple containing the fact that needs to be repeated most urgently and a boolean indicating whether this fact is new (True) or has been presented before (False).
        If none of the previously studied facts needs to be repeated right now, return a new fact instead.
        When the flipping threshold of the chosen fact is reached before presenting it, it will be returned in reversed order.
        """
        next_fact, new = super().get_next_fact(current_time)

        if use_flipping or not new:
            return self.flip_fact(next_fact, current_time), new
        else:
            return next_fact, new

    def flip_fact(self, fact: Fact, time: int) -> Fact:
        """decide if the fact needs to be flipped or not"""

        if self.calculate_flip_activation(time, fact) > self.FLIPPING_THRESHOLD:

            new_fact = fact._replace(
                question=fact.answer, answer=fact.question, flipped=not fact.flipped)
            self.facts[self.facts.index(fact)] = new_fact

            return new_fact
        else:
            return fact

    def get_test_questions(self, flip_ratio: float = 0.5, max_facts: int = None) -> list:
        """
        return randomly shuffled learned facts for testing
        If the `max` parameter is set, the return will be limited to that number of facts if it is lower than the full number of facts.
        The `flip_ratio` parameter determines how many of the returned facts are flipped.
        """

        dist_resp = set([r.fact.fact_id for r in self.responses])

        studied_facts = []

        if flip_ratio > 1:
            flip_ratio = 1

        if flip_ratio < 0:
            flip_ratio = 0

        for idx in dist_resp:
            fact = [f for f in self.facts if f.fact_id == idx][0]
            if fact.flipped:
                # normalize facts
                fact = fact._replace(question=fact.answer,
                              answer=fact.question, flipped=not fact.flipped)
            studied_facts.append(fact)

        random.shuffle(studied_facts)

        if max_facts:
            studied_facts = studied_facts[:max_facts]

        num_facts = len(studied_facts)

        for i in range(floor(num_facts*flip_ratio)):
            flip_fact = studied_facts[i]._replace(
                question=studied_facts[i].answer, answer=studied_facts[i].question, flipped=not studied_facts[i].flipped)
            studied_facts[i] = flip_fact

        random.shuffle(studied_facts)

        return studied_facts

    def calculate_flip_activation(self, time: int, fact: Fact) -> float:
        """calculate the flipping activation of a fact"""
        encounters = []

        responses_for_fact = [
            r for r in self.responses if r.fact.fact_id == fact.fact_id and r.start_time < time and r.fact.flipped == fact.flipped]

        alpha = self.FLIPPING_ALPHA

        # Calculate the activation by running through the sequence of previous responses
        for response in responses_for_fact:
            activation = self.calculate_activation_from_encounters(
                encounters, response.start_time)
            encounters.append(Encounter(activation, response.start_time,
                              self.normalise_reaction_time(response), self.FLIPPING_ALPHA))
            alpha = self.estimate_flip_alpha(
                encounters, activation, response, alpha)

            # Update decay estimates of previous encounters
            encounters = [encounter._replace(decay=self.calculate_decay(
                encounter.activation, alpha)) for encounter in encounters]

        return self.calculate_activation_from_encounters(encounters, time)

    def get_flip_alpha(self, time: int, fact: Fact) -> float:
        """
        Return the estimated flip alpha of the fact at the specified time
        """
        encounters = []

        responses_for_fact = [
            r for r in self.responses if r.fact.fact_id == fact.fact_id and r.start_time < time and r.fact.flipped == fact.flipped]
        alpha = self.FLIPPING_ALPHA

        # Calculate the activation by running through the sequence of previous responses
        for response in responses_for_fact:
            activation = self.calculate_activation_from_encounters(
                encounters, response.start_time)
            encounters.append(Encounter(activation, response.start_time,
                              self.normalise_reaction_time(response), self.FLIPPING_ALPHA))
            alpha = self.estimate_flip_alpha(
                encounters, activation, response, alpha)

            # Update decay estimates of previous encounters
            encounters = [encounter._replace(decay=self.calculate_decay(
                encounter.activation, alpha)) for encounter in encounters]

        return alpha

    def estimate_flip_alpha(self, encounters: List[Encounter], activation: float, response: Response, previous_alpha: float) -> float:
        """
        Estimate the alpha parameter for an item.
        """
        if len(encounters) < 3:
            return self.FLIPPING_ALPHA

        a_fit = previous_alpha
        reading_time = self.get_reading_time(response.fact.question)
        estimated_rt = self.estimate_reaction_time_from_activation(
            activation, reading_time)
        est_diff = estimated_rt - self.normalise_reaction_time(response)

        if est_diff < 0:
            # Estimated RT was too short (estimated activation too high), so actual decay was larger
            a0 = a_fit
            a1 = a_fit + 0.05

        else:
            # Estimated RT was too long (estimated activation too low), so actual decay was smaller
            a0 = a_fit - 0.05
            a1 = a_fit

        # Binary search between previous fit and proposed alpha
        for _ in range(6):
            # Adjust all decays to use the new alpha
            a0_diff = a0 - a_fit
            a1_diff = a1 - a_fit
            d_a0 = [e._replace(decay=e.decay + a0_diff) for e in encounters]
            d_a1 = [e._replace(decay=e.decay + a1_diff) for e in encounters]

            # Calculate the reaction times from activation and compare against observed RTs
            encounter_window = encounters[max(1, len(encounters) - 5):]
            total_a0_error = self.calculate_predicted_reaction_time_error(
                encounter_window, d_a0, reading_time)
            total_a1_error = self.calculate_predicted_reaction_time_error(
                encounter_window, d_a1, reading_time)

            # Adjust the search area based on the lowest total error
            ac = (a0 + a1) / 2
            if total_a0_error < total_a1_error:
                a1 = ac
            else:
                a0 = ac

        # The new alpha estimate is the average value in the remaining bracket
        return (a0 + a1) / 2

    def export_data(self, path: str = None) -> Union[pd.DataFrame, str]:
        """
        Save the response data to the specified csv file, and return a copy of the pandas DataFrame.
        If no path is specified, return a CSV-formatted copy of the data instead.
        """

        df_string = super().export_data()

        df = pd.read_csv(io.StringIO(df_string))

        def get_fact(fact_id):
            return [f for f in self.facts if f.fact_id == str(fact_id)][0]

        def calc_flip_rof(row):
            return self.get_flip_alpha(row["start_time"] + 1, get_fact(row["fact_id"]))

        def calc_flip_act(row):
            return self.calculate_flip_activation(row["start_time"], get_fact(row["fact_id"]))

        # Add column for rate of forgetting estimate after each observation
        df["flip_alpha"] = df.apply(calc_flip_rof, axis=1)
        df["flip_activation"] = df.apply(calc_flip_act, axis=1)

        # Save to CSV file if a path was specified, otherwise return the CSV-formatted output
        if path is not None:
            df.to_csv(path, encoding="UTF-8")
            return df

        return df.to_csv()
