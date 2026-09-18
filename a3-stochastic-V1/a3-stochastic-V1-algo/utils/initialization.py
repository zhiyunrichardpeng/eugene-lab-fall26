import importlib
import os.path
import sys

import gym
from wrapping_env import wrapping_env


def create_env(**kwargs):
    env_name = kwargs["env_id"]
    env_name_data = env_name + "_data"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_gym_path = os.path.join(current_dir, "../env_gym")
    import sys
    sys.path.append(env_gym_path)
    try:
        file = __import__(env_name_data)
    except NotImplementedError:
        raise NotImplementedError("This environment does not exist")

    import sys
    from importlib import reload
    my_module = sys.modules[env_name_data]

    try:
        # First time the module imports, loaded is undefined and will trigger the exception.
        # loaded is then defined in the exception block
        # when reloaded, loaded is defined as true, so reload your submodules:
        # print("loaded l29", loaded_3)
        if loaded_3:
            reload(my_module)
            file = __import__(env_name_data)
    except:
        loaded_3 = True
        reload(my_module)
        file = __import__(env_name_data)        
        # print("loaded l38", loaded_3)

    env_name_camel = formatter(env_name)
    # print("loaded l41 in init.py")
    if hasattr(file, "env_creator"):
        # print("loaded l43 in init.py")
        # print(file)
        env_class = getattr(file, "env_creator")
        # print(env_class)
        env = env_class(**kwargs)
    elif hasattr(file, env_name_camel):
        # print("loaded l47 in init.py")
        env_class = getattr(file, env_name_camel)
        env = env_class(**kwargs)
    else:
        # print("Env name: ", env_name_camel)
        raise NotImplementedError("This environment is not properly defined")
    # print("loaded l51 init.py")
    # Wrapping the env
    max_episode_steps = kwargs.get("max_episode_steps", None)
    reward_scale = kwargs.get("reward_scale", None)
    reward_shift = kwargs.get("reward_shift", None)
    # print("loaded l56 init.py")
    env = wrapping_env(
        env=env,
        max_episode_steps=max_episode_steps,
        reward_shift=reward_shift,
        reward_scale=reward_scale,
    )

    print("Create environment successfully!")
    # print("environment name:", kwargs["env_id"])
    return env


def create_alg(**kwargs):
    alg_name = kwargs["algorithm"]
    alg_file_name = alg_name.lower()
    try:
        module = importlib.import_module(alg_file_name)
    except NotImplementedError:
        raise NotImplementedError("This algorithm does not exist")

    if hasattr(module, alg_name):
        alg_cls = getattr(module, alg_name)
        alg = alg_cls(**kwargs)
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create algorithm successfully!")
    return alg


def create_apprfunc(**kwargs):
    apprfunc_name = kwargs["apprfunc"]
    apprfunc_file_name = apprfunc_name.lower()
    try:
        file = importlib.import_module('networks.' + apprfunc_file_name)
    except NotImplementedError:
        raise NotImplementedError("This apprfunc does not exist")

    # name = kwargs['name'].upper()

    name = formatter(kwargs["name"])
    # print(name)
    # print(kwargs)

    if hasattr(file, name):  #
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError("This apprfunc is not properly defined")

    # print("--Initialize appr func: " + name + "...")
    return apprfunc


def create_buffer(**kwargs):
    buffer_file_name = kwargs["buffer_name"].lower()
    try:
        module = importlib.import_module("training." + buffer_file_name)
    except NotImplementedError:
        raise NotImplementedError("This buffer does not exist")

    buffer_name = formatter(buffer_file_name)

    if hasattr(module, buffer_name):  #
        buffer_cls = getattr(module, buffer_name)  #
        buffer = buffer_cls(**kwargs)
    else:
        raise NotImplementedError("This buffer is not properly defined")

    print("Create buffer successfully!")
    return buffer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
