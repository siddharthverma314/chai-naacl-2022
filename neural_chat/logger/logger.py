from __future__ import annotations
from .utils import Container, consolidate_stats
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tabulate import tabulate
from flatten_dict import flatten

import torch
import datetime
import json
import termcolor
import pygments.lexers
import pygments.formatters


class Logger:
    """Handles hyperparameter and epoch logging. The logging is done via
    Tensorboard while additional material is stored within the log
    directory.

    """

    def __init__(self):
        self.modules = Container()
        self.logdir = None

        # for pretty printing
        self._lexer = pygments.lexers.JsonLexer()
        self._formatter = pygments.formatters.TerminalFormatter()

    def initialize(self, modules: dict, logdir=None, snapshot_gap=50):
        """
        Initialize the current logger.

        This method is kept separate from __init__ so that one can
        re-initialize the default logger.

        :param logdir: logging directory
        :param snapshot_gap: how often to snapshot
        """

        if logdir:
            self.logdir = Path(logdir)
        else:
            self.logdir = "/tmp/adversarial_log"
        self.snapshot_gap = snapshot_gap

        self.modules = Container(**modules)
        self.sw = SummaryWriter(logdir)
        self._epoch_num = 0

    def log(self, msg):
        "Log arbitrary text. Saved into output file if applicable."

        time = datetime.datetime.now().ctime()
        time_colored = termcolor.colored(time, "green")

        print(f"{time_colored}| {msg}")

        if self.logdir:
            output_file = self.logdir / "output"
            with open(output_file, "a") as f:
                f.write(f"{time}| {msg}\n")

    def _log_dict(self, dict, filename=None):
        "Log a dictionary and save to file if applicable"
        if filename and not filename.endswith(".json"):
            filename += ".json"

        fancy_json = json.dumps(dict, sort_keys=True, indent=2)
        colorful_json = pygments.highlight(fancy_json, self._lexer, self._formatter)

        time = datetime.datetime.now().ctime()
        time_colored = termcolor.colored(time, "green")

        print(f"{time_colored}| file {filename}")
        print(colorful_json)

        if filename and self.logdir:
            plain_json = json.dumps(dict)
            output_file = self.logdir / filename
            with open(output_file, "w") as f:
                f.write(plain_json)
                self.log("Wrote file")

    def epoch(self, epoch_num=None):
        "To be called at each epoch"
        if epoch_num:
            self._epoch_num = epoch_num

        self.log_epoch()
        if (self._epoch_num + 1) % self.snapshot_gap == 0:
            self.log_snapshot()

        self._epoch_num += 1

    def log_hyperparameters(self):
        "Log hyperparameters from self.modules"
        self.log("Logging hyperparameters")
        self._log_dict(self.modules.log_hyperparams(), "hyperparams")

    def log_epoch(self):
        "Log epoch from self.modules"
        self.log("Logging epoch {}".format(self._epoch_num))
        epoch = flatten(self.modules.log_epoch(), reducer="path")

        # first check for histograms
        for k, v in epoch.items():
            if v.squeeze().dim() > 0:
                self.sw.add_histogram(k, v, self._epoch_num)
                epoch[k] = consolidate_stats(v)

        # next, log scalars
        epoch = flatten(epoch, reducer="path")
        epoch = {k: v.item() for k, v in epoch.items()}
        for k, v in epoch.items():
            self.sw.add_scalar(k, v, self._epoch_num)

        self.log("params:\n" + tabulate([(k, v) for k, v in epoch.items()]))

    def log_snapshot(self):
        "Log snapshot from self.modules"
        self.log("Logging snapshot")
        if self.logdir:
            snapshot = self.modules.log_snapshot()
            output_file = self.logdir / f"snapshot_{self._epoch_num}.pkl"
            self.log("Saving snapshot to {}".format(output_file))
            torch.save(snapshot, output_file)


logger = Logger()
