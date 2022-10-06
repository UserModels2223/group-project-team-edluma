import csv
from time import time
import slimstampen.flippingmodel as slim
import argparse

parser = argparse.ArgumentParser(prog="Slimstampen Flipping model",
                                 description="runs the slimstampen spacing model extended with a flipping functionality", epilog="Type `exit` at any time during the run to end immediatly")
parser.add_argument("--time", "-T", type=int, default=10,
                    help="The number of minutes the program should run (default: 10)")
parser.add_argument("--file", "-F", type=str,
                    help="The path to the csv file that will be used to generate the facts", required=True)
parser.add_argument("--limit", "-L", type=int,
                    help="Limits the maximal number of words used in the training session (default: no limit)")
parser.add_argument("--trials", "-R", type=int, 
                    help="Specifies the maximal number of trials for the run (default: no limit)\nWhen this option is specified together with a time limit, the session will end with the first parameter to be reach the limit.")

if __name__ == "__main__":

    options = parser.parse_args()

    m = slim.FlippingModel()

    with open(options.file, encoding="utf-8") as file:
        csvreader = csv.reader(file)
        count = 0
        for row in csvreader:
            if options.limit and count > options.limit:
                break
            fact = slim.Fact(*row)
            m.add_fact(fact)
            count+=1

    start = time()
    end = start + (options.time*60)
    startMs = start*1000

    count = 0
    while start < end:

        if options.trials and count > options.trials:
            break

        presTime = int(time()*1000 - startMs)
        fact, new = m.get_next_fact(presTime)

        if new:
            answer = input(
                f"New vocabulary: {fact.answer} means {fact.question}!\nPlease type what {fact.question} means below.\n").strip()
        else:
            answer = input(
                f"What is the translation of {fact.question}?\n").strip()

        if answer == "exit":
            break

        rt = int(time()*1000 - startMs) - presTime
        correct = False
        if answer == fact.answer:
            correct = True

        resp = slim.Response(fact, presTime, rt, correct)
        m.register_response(resp)

        start = time()
        count+=1

    m.export_data("data.csv")
