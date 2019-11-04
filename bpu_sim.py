import os
import sys
import gzip
import json
import itertools
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict

################################################################################
#                                 BPU EMULATOR                                 #
#                              BY DEREK RODRIGUEZ                              #
################################################################################


################################################################################
#                     CLASSES FOR BRANCH PREDICTING UNITS                      #
################################################################################


class DecisionMechanism:
    """ ABC for All Decision Mechanisms used to decide Taken/Untaken """

    def __init__(self, default_state):
        self.state = default_state

    def decide_and_update(self, bhr: int, actual_answer) -> bool:
        """ Do we take the current branch or no?"""
        pass


class OneBitDecisionMechanism(DecisionMechanism):
    """Simple one bit predictor that follows previous branch history"""

    def __init__(self, default_state):
        super().__init__(default_state)

    def decide_and_update(self, bhr: int, actual_answer: bool) -> bool:
        """ Do whatever was done on the previous matching branch sequence. """
        outcome = self.state
        state = actual_answer
        return outcome


class TwoBitCounterDecisionMechanism(DecisionMechanism):
    """ Two Bit counter that adds histeresis over the one bit. """

    def __init__(self, default_state):
        super().__init__(default_state)

    def decide_and_update(self, bhr: int, actual_answer: bool) -> bool:
        """ Prediction result is first of two bits, a taken branch increments 
        and an untaken branch decrements. """
        outcome = bool(self.state & 2)
        if (actual_answer and not outcome) and self.state < 3:
            self.state += 1
        if (not (actual_answer) and outcome) and self.state > 0:
            self.state -= 1
        return outcome


class PerceptronDecisionMechanism(DecisionMechanism):
    """ Perceptron predictor that finds linear boundary for prediction. 

        I went ahead and set everything up so that defaults to working for 
        an 8 entry PHT (3 bits in the BHR). Things I modified from the original
        paper were the training method for the bias + their formula for calculating
        the training threshold (generalized because the 1.92 is just log2 of bits per
        weight).
    """

    def __init__(self, default_state, bhr_bit_count: int):
        super().__init__(default_state)
        self.bias = 1e-6
        self.weights = np.zeros(bhr_bit_count, dtype=np.float_)
        self.training_threshold = (
            0.3678794412 * bhr_bit_count + 6
        )  # 1/e as training slope bc floating point, original paper used integers

    def decide_and_update(self, bhr: int, actual_answer: bool) -> bool:
        """ Activation function is just the identity function """
        activation = self.bias
        y = 1 if actual_answer else -1
        x = [(-1, 1)[(bhr >> i) & 1] for i in range(len(self.weights))]
        for i in range(len(self.weights)):
            activation += self.weights[i] * x[i]
        if ((activation / abs(activation)) != y) or (
            abs(activation) < self.training_threshold
        ):
            self.bias += x[0] * y
            for i in range(len(self.weights)):
                self.weights[i] += x[i] * y
        return bool(activation > 0)


class TwoLevelAdaptiveBranchPredictorUnit:
    """ BPU emulator that tracks state for the DecisionMechanism """

    def __init__(self, cfg={}, bhr_bit_count=3, decision_object=None):
        self.cfg = cfg
        self.bhr = 0
        self.max_bhr_val = 2 ** (bhr_bit_count) - 1
        self.pht = {k: deepcopy(decision_object) for k in range(2 ** bhr_bit_count)}
        self.branches_seen = 0
        self.branches_correct = 0

    def process_branch(self, actual_answer: bool):
        """ Return prediction for branch, update state """
        decision = self.pht[self.bhr & self.max_bhr_val].decide_and_update(
            self.bhr, actual_answer
        )  # get decision and update predictor
        self.bhr = (
            (self.bhr << 1) | int(actual_answer)
        ) & self.max_bhr_val  # update bhr for next branch
        return decision

    def parse_itrace(self, addr: np.ndarray) -> None:
        """ Parse the entire instruction trace """
        buffer_misses = 0
        seen = set()
        for i in tqdm(range(len(addr) - 1), desc="Instructions parsed", unit="instr"):
            if addr[i] in self.cfg.keys():
                buffer_misses += 1 if addr[i] not in seen else 0
                actual_answer = bool(self.cfg[addr[i]].index(addr[i + 1]))
                prediction = self.process_branch(actual_answer)
                self.branches_seen += 1
                if prediction == actual_answer:
                    self.branches_correct += 1
        sys.stderr.write("Buffer Misses: {}\n".format(buffer_misses))
        sys.stderr.write(
            "Total accuracy: {} / {} = {}%\n".format(
                self.branches_correct,
                self.branches_seen,
                (self.branches_correct / self.branches_seen) * 100.0,
            )
        )


################################################################################
#                         PARSING BRANCHES FROM ITRACE                         #
################################################################################


def generate_cfg(addr: np.ndarray, filename: str) -> Dict[np.uint64, List[np.uint64]]:
    """Generate adjacency list for conditional branches.
    
    I considered multiple different attempts at parsing branches, but at the
    end of the day I determined this was the most straightforward method. I
    first generate a table of every bigram in the itrace, and then use those
    bigrams to map all source instructions to their potential destination
    instructions.  That almost worked but I spent far too much time parsing it.
    Instead I decided to "filter out" my desired branches based on the
    following criteria:
        - The real branch address appears more than once.
        - All addresseses immediately following the true branch only take one
          of two unique values. These are the targets of the branch
        - the sum of the observations of all branch_targets sum up to the
          observations of the branch itself. 
    """
    cfg = {}
    vals, freqs = np.unique(addr, return_counts=True)
    sys.stderr.write(
        "Parsing trace for conditional branches, this will take a minute or two...\n"
    )
    for branch_candidate in tqdm(np.take(vals, np.where(freqs > 1)[0]), disable=False):
        at_indices = np.where(addr == branch_candidate)[0]
        at_indices = np.extract(at_indices < len(addr) - 1, at_indices)
        after_indices = at_indices + 1
        outcomes = np.unique(np.take(addr, after_indices))
        if len(outcomes) == 2:
            outcome, other_outcome = outcomes
            outcome_count = freqs[vals == outcome]
            other_outcome_count = freqs[vals == other_outcome]
            branch_count = freqs[vals == branch_candidate]
            if outcome_count + other_outcome_count == branch_count:
                cfg[branch_candidate] = sorted(
                    outcomes, key=lambda l: abs(int(l) - branch_candidate)
                )
    return cfg


def print_stats(addr: np.ndarray, cfg: Dict[np.uint64, List[np.uint64]]) -> None:
    """Dump Statistics for all found branches"""
    # dump sample frequency
    unique_addr, addr_frequency = np.unique(addr, return_counts=True)
    for branch, outcomes in cfg.items():
        untaken, taken = outcomes
        untaken_count = addr_frequency[unique_addr == untaken]
        taken_count = addr_frequency[unique_addr == taken]
        branch_count = addr_frequency[unique_addr == branch]
        print(
            "{} ? {} : {} | Taken = {} / {}".format(
                hex(branch),
                hex(untaken),
                hex(taken),
                taken_count,
                untaken_count + taken_count,
            )
        )

    # dump summary
    print(
        "Parser counted {} unique branches occurring a total of {} times.".format(
            len(cfg.keys()),
            np.take(
                addr_frequency,
                np.searchsorted(unique_addr, np.asarray(list(cfg.keys()))),
            ).sum(),
        )
    )


def dump_cfg(cfg: Dict[np.uint64, List[np.uint64]]) -> None:
    pprint({hex(k): list(map(hex, v)) for k, v in cfg.items()})


################################################################################
#                            INPUT AND SETUP HANDLING                          #
################################################################################


def load_gzipped_txt(filename: str) -> np.ndarray:
    """Opens the compressed file and returns them in numpy array"""
    with gzip.open(filename) as fh:
        addr = []
        for line in fh:
            addr.append(int(line, 16))
        return np.asarray(addr, dtype=np.uint64)


def load_regular_txt(filename: str) -> np.ndarray:
    """Opens the file and returns them in numpy array"""
    with open(filename) as fh:
        addr = []
        for line in fh:
            addr.append(int(line, 16))
        return np.asarray(addr, dtype=np.uint64)


def choose_load_function(filename: str) -> np.ndarray:
    """Parse tracefile extension to use correct loading method.

    By default, I tested my code with 1E6 PC samples from the given itrace 
    stored in a numpy (.npy) file.
    """
    extension = filename.split(".")[-1]
    if extension == "gz":
        return load_gzipped_txt(filename)
    elif extension == "out" or extension == "txt":
        return load_regular_txt(filename)
    elif extension == "npy":
        return np.fromfile(filename, dtype=np.uint64)[:1000000]


def fix_adjacent_repeated_pc(addr: np.ndarray) -> np.ndarray:
    """Remove adjacent repeats from trace."""
    sys.stderr.write("Fixing sample aliasing in trace...\n")
    return np.asarray(
        [a for a, b in tqdm(itertools.groupby(addr.tolist()), total=len(addr))]
    )


def main():
    parser = ArgumentParser(
        description="Parse PC Traces and Simulate Branch Predictors."
    )
    parser.add_argument(
        "--tracefile",
        type=str,
        default="addrs.npy",
        help="Name of the file containing i-trace. must be either numpy or "
        "ASCII text, or gzip-compressed ASCII text format. Defaults to file"
        " called addrs.npy in local folder",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show captured branch statistics for an instruction trace",
    )
    parser.add_argument(
        "--dot",
        action="store_true",
        help="instead of doing prediction, just generate dotfile to visualize "
        "control flow in GraphViz",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump the adjacency list of the control flow graph for the conditional branches",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bit", "counter", "perceptron", "homework"],
        default="homework",
        help="Select the decision model to use from the"
        "BPU. Default is 2-bit counter. Will default to configuring and running"
        "everything according to homework.",
    )
    parser.add_argument(
        "--no-predict",
        dest="no_pred",
        action="store_true",
        help="disable the prediction run",
    )
    parser.add_argument(
        "--bhr-bits",
        dest="bhr_bit_count",
        type=int,
        default=3,
        help="Amount of bits in the branch history register",
    )
    args = parser.parse_args()

    # Do the important stuff based on input
    addr = choose_load_function(args.tracefile)
    addr = fix_adjacent_repeated_pc(addr)
    cfg = generate_cfg(addr, args.tracefile)
    if args.stats:
        print_stats(addr, cfg)
    if args.dot:
        print_dotfile(addr, cfg)
    if args.dump:
        dump_cfg(cfg)
    if not args.no_pred and not args.dot:
        decision_obj = None
        if args.model == "bit":
            decision_obj = OneBitDecisionMechanism(default_state=False)
            bpu = TwoLevelAdaptiveBranchPredictorUnit(
                cfg=cfg, bhr_bit_count=args.bhr_bit_count, decision_object=decision_obj
            )
            bpu.parse_itrace(addr)
        elif args.model == "counter":
            decision_obj = TwoBitCounterDecisionMechanism(
                default_state=1
            )  # 1 = weakly not taken
            bpu = TwoLevelAdaptiveBranchPredictorUnit(
                cfg=cfg, bhr_bit_count=args.bhr_bit_count, decision_object=decision_obj
            )
            bpu.parse_itrace(addr)
        elif args.model == "perceptron":
            decision_obj = PerceptronDecisionMechanism(
                None, bhr_bit_count=args.bhr_bit_count
            )
            bpu = TwoLevelAdaptiveBranchPredictorUnit(
                cfg=cfg, bhr_bit_count=args.bhr_bit_count, decision_object=decision_obj
            )
            bpu.parse_itrace(addr)
        elif args.model == "homework":
            # first part with single bits
            sys.stderr.write("Doing run with 32 one-bit predictor entries\n")
            one_bit = OneBitDecisionMechanism(default_state=False)
            thirty_two_table = TwoLevelAdaptiveBranchPredictorUnit(
                cfg, bhr_bit_count=5, decision_object=one_bit
            )
            thirty_two_table.parse_itrace(addr)
            # then do the two bit pred...
            sys.stderr.write("Doing run with 16 two-bit counter entries\n")
            two_bit = TwoBitCounterDecisionMechanism(default_state=1)
            sixteen_table = TwoLevelAdaptiveBranchPredictorUnit(
                cfg, bhr_bit_count=4, decision_object=two_bit
            )
            sixteen_table.parse_itrace(addr)
            # Then do experimental trace
            sys.stderr.write("EXPERIMENTAL SUBMISSION: 8 perceptron entries\n")
            perceptron = PerceptronDecisionMechanism(
                default_state=None, bhr_bit_count=3
            )
            eight_table = TwoLevelAdaptiveBranchPredictorUnit(
                cfg, bhr_bit_count=3, decision_object=perceptron
            )
            eight_table.parse_itrace(addr)


if __name__ == "__main__":
    main()
