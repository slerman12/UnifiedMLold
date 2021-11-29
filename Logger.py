# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import csv
import datetime
from collections import defaultdict

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter


def shorthand(log_name):
    return ''.join([s[0].upper() for s in log_name.split('_')] if len(log_name) > 3 else log_name.upper())


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
    def __init__(self, root_path):
        self.root_path = root_path

        self.logs = {}
        self.counts = {}

    def log(self, log=None, name="Logs", dump=False):
        if log is not None:

            if name not in self.logs:
                self.logs[name] = {}
                self.counts[name] = {}

            logs = self.logs[name]
            counts = self.counts[name]

            for k, l in log.items():
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
        # self.dump_to_csv(logs, name=name)
        self.dump_to_console(logs, name=name)

    def dump_to_csv(self, logs, name):
        if self.csv_writer is None:
            write_header = True
            if self.file_name.exists():
                self.remove_old_entries(logs, name)
                write_header = False

            file = self.file_name.open('a')
            writer = csv.DictWriter(file,
                                    fieldnames=sorted(logs.keys()),
                                    restval=0.0)
            if write_header:
                self.csv_writer.writeheader()

        self.csv_writer.writerow(logs)
        self.csv_file.flush()

    def dump_to_console(self, logs, name):
        name = colored(name, 'yellow' if name.lower() == 'train' else 'green')
        pieces = [f'| {name: <14}']
        for log_name, log in logs.items():
            pieces.append(format(log, log_name))
        print(' | '.join(pieces))

    def log_tensorboard(self, logs, name):
        pass


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise Exception(f'invalid format type: {ty}')

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        # data['frame'] = step * self.action_repeat
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class LoggerOld(object):
    def __init__(self, log_dir, use_tensorboard):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT)
        if use_tensorboard:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _log(self, key, value, step):
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log(self, logs, name='Logs', dump=False):
        assert 'step' in logs
        step = logs['step']
        for key, value in logs.items():
            self._log(f'{name}/{key}', value, step)
        if dump:
            self.dump_logs(step, name)

    def dump_logs(self, step, name='Logs'):
        self._train_mg.dump(step, name)
