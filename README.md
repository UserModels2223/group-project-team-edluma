# User Models Project

We aim to optimize the base slimstampen activation-based algorithm by introducing the possibility of question-answer pairs to be flipped, such that the original question becomes the answer and the original answer the question.

We use the 1000 most common Swahili words as input for our experiments, but removed a few words that were in conjugated form as well as small words such as presonal pronouns from the list. The full list can be found in the `data` folder.

## Requirements

- [Python 3](https://www.python.org/downloads/) (recommended) or Python 2.7+
- [pandas](https://pandas.pydata.org/getting_started.html)

To open the tutorial, you need [jupyter](https://jupyter.org/install). The experiment example requires [OpenSesame](https://osdoc.cogsci.nl/3.3/download/).
If something is not working as expected, first make sure that you have up-to-date versions of all these software packages.

## How to Run

You can either run the notebook [flipping.ipynb](flipping.ipynb) or start the program over the commandline with the command `python main.py`. If you start through the commandline you will also have to specify the data file you want to use (`--file` or `-F` flag) as well as the duration of the session (`--time` or `-T` flag, default: 10 minutes). A good default command for testing would e.g. be `python main.py -F ./data/swahili.csv -T 5`.

The jupyter notebook [Tutorial.ipynb](Tutorial.ipynb) shows the basics of using the normal slimstampen spacing model.

## OpenSesame

An OpenSesame configuration is also included with the code ([OpenSesameExample.osexp](OpenSesameExample.osexp)). The spacing model code is embedded in the file, so that it is easier to share with participants.
