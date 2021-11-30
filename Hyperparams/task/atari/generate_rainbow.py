RAINBOW_ENVS = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]

out = ""
for env in RAINBOW_ENVS:
    f = open(f"./{env.lower()}.yaml", "w")
    f.write(r"""defaults:
  - easy
  - _self_

suite: atari
task_name: {}""".format(env))
    f.close()
    out += ' "' + env.lower() + '"'
print(out)
