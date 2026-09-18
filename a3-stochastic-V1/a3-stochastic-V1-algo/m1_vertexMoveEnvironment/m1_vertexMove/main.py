import gym
import m1_vertexMove_Va001

from importlib import reload

vertexMove_update = sys.modules['m1_vertexMove_Va001']

try:
    # First time the module imports, loaded is undefined and will trigger the exception.
    # loaded is then defined in the exception block
    # when reloaded, loaded is defined as true, so reload your submodules:
    print("loaded l10 in vertex move main", loaded_6)
    if loaded_6:
        reload(vertexMove_update)
        import m1_vertexMove_Va001
except:
    loaded_6 = True
    print("loaded l16 in vertex move main", loaded_6)
    reload(vertexMove_update)
    import m1_vertexMove_Va001 


env = gym.make('m1_vertexMove_Va002')