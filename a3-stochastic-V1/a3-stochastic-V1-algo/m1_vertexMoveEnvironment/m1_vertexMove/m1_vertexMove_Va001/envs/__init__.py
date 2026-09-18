from m1_vertexMove_Va001.envs.m1_vertexMove_Va001_env import m1_vertexMove_Va001



import sys
from importlib import reload
my_module = sys.modules['m1_vertexMove_Va001.envs.m1_vertexMove_Va001_env']

try:
    # First time the module imports, loaded is undefined and will trigger the exception.
    # loaded is then defined in the exception block
    # when reloaded, loaded is defined as true, so reload your submodules:
    print("loaded l13", loaded_2)
    if loaded_2:
        reload(my_module)
        from m1_vertexMove_Va001.envs.m1_vertexMove_Va001_env import m1_vertexMove_Va001
except:
    loaded_2 = True
    print("loaded l19", loaded_2)
    reload(my_module)
    from m1_vertexMove_Va001.envs.m1_vertexMove_Va001_env import m1_vertexMove_Va001
