import csv
from time import time
import slimstampen.flippingmodel as slim
import argparse

parser = argparse.ArgumentParser(prog="Slimstampen Flipping model",
                                 description="runs the slimstampen spacing model extended with a flipping functionality", epilog="Type `exit`at any time during the run to end immediatly")
parser.add_argument("--time", "-T", type=int, default=10,
                    help="The number of minutes the program should be run (default: 10)")
parser.add_argument("--file", "-F", type=str,
                    help="The csv file that will be used to generate the facts", required=True)

if __name__ == "__main__":

    options = parser.parse_args()

    m = slim.FlippingModel()

    with open(options.file, encoding="utf-8") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fact = slim.Fact(*row)
            m.add_fact(fact)

    start = time()
    end = start + (options.time*60)
    startMs = start*1000

    while start < end:

        fact, new = m.get_next_fact(int(time()*1000 - startMs))

        presTime = int(time()*1000 - startMs)
        if new:
            answer = input(f"New vocabulary: {fact.answer} means {fact.question}!\nPlease type what {fact.question} means below.\n").strip()
        else:
            answer = input(f"What is the translation of {fact.question}?\n").strip()

        if answer == "exit":
            break

        rt = int(time()*1000 - startMs) - presTime
        correct = False
        if answer == fact.answer:
            correct = True

        resp = slim.Response(fact, presTime, rt, correct)
        m.register_response(resp)

        start = time()

    m.export_data("data.csv")
