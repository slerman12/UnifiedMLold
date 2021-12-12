# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import csv
import datetime
from pathlib import Path
from termcolor import colored

from torch.utils.tensorboard import SummaryWriter


def shorthand(log_name):
    return ''.join([s[0].upper() for s in log_name.split(' ')] if len(log_name) > 3 else log_name.upper())


def format(log, log_name):
    l = shorthand(log_name)

    if 'time' in log_name.lower():
        log = str(datetime.timedelta(seconds=int(log)))
        return f'{l}: {log}'
    elif float(log).is_integer():
        log = int(log)
        return f'{l}: {log}'
    else:
        return f'{l}: {log:.04f}'


class Logger:
    def __init__(self, root_path='.', experiment='Experiment', agent='Agent', task='Task', seed=-1):

        self.experiment = experiment
        self.agent = agent.replace('Agents.', '').replace('Agent', '')
        self.task = task
        self.seed = seed

        self.log_path = Path(f'{root_path}/Benchmarking/{experiment}/{self.agent}')
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logs = {}
        self.counts = {}

        self.tensorboard_writer = None

    def log(self, log=None, name="Logs", dump=False):
        if log is not None:

            if name not in self.logs:
                self.logs[name] = {}
                self.counts[name] = {}

            logs = self.logs[name]
            counts = self.counts[name]

            for k, l in log.items():  # TODO Aggregate per step
                if k in logs:
                    logs[k] += l
                    counts[k] += 1
                else:
                    logs[k] = l
                    counts[k] = 1

        if dump:
            self.dump_logs(name)

    def dump_logs(self, name=None):
        if name is None:
            for n in self.logs:
                for log_name in self.logs[n]:
                    self.logs[n][log_name] /= self.counts[n][log_name]
                self._dump_logs(self.logs[n], name=n)
                del self.logs[n]
                del self.counts[n]
        else:
            if name not in self.logs:
                return
            for log_name in self.logs[name]:
                self.logs[name][log_name] /= self.counts[name][log_name]
            self._dump_logs(self.logs[name], name=name)
            self.logs[name] = {}
            del self.logs[name]
            del self.counts[name]

    def _dump_logs(self, logs, name):
        self.dump_to_console(logs, name=name)
        self.dump_to_csv(logs, name=name)

    def dump_to_console(self, logs, name):
        name = colored(name, 'yellow' if name.lower() == 'train' else 'green')
        pieces = [f'| {name: <14}']
        for log_name, log in logs.items():
            pieces.append(format(log, log_name))
        print(' | '.join(pieces))

    def remove_old_entries(self, logs, file_name):
        rows = []
        with file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= logs['step']:
                    break
                rows.append(row)
        with file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=logs.keys(),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dump_to_csv(self, logs, name):
        logs = dict(logs)

        assert 'step' in logs
        for log_name in ['agent', 'task', 'seed', 'experiment']:
            logs[log_name] = getattr(self, log_name)

        file_name = self.log_path / f'{name}_{self.task}_{self.seed}_{self.experiment}.csv'
        write_header = True
        if file_name.exists():
            try:
                self.remove_old_entries(logs, file_name)
                write_header = False
            except Exception:
                print(f'Header mismatch: overwriting old entries of {file_name}')

        file = file_name.open('a')
        writer = csv.DictWriter(file,
                                fieldnames=logs.keys(),
                                restval=0.0)
        if write_header:
            writer.writeheader()

        writer.writerow(logs)
        file.flush()

    # TODO add log_weights_and_biases / log_wandb too
    def log_tensorboard(self, logs, name):
        if self.tensorboard_writer is None:
            self.tensorboard_writer = SummaryWriter(self.log_path / 'TensorBoard')

        for key in logs:
            if key != 'step' and key != 'episode':
                self.tensorboard_writer.add_scalar(f'{name}/{key}', logs[key], logs['step'])