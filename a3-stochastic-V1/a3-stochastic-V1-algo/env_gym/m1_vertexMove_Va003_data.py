import gym
import sys
from paths import a3_stochastic_relative_path
sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo" / "m1_vertexMoveEnvironment" / "m1_vertexMove"))
# import cartpoleV2
# import gym
import m1_vertexMove_Va001

from importlib import reload

vertexMove_update = sys.modules['m1_vertexMove_Va001']


try:
    # First time the module imports, loaded is undefined and will trigger the exception.
    # loaded is then defined in the exception block
    # when reloaded, loaded is defined as true, so reload your submodules:
    print("loaded l19 in _data", loaded_5)
    if loaded_5:
        reload(vertexMove_update)
        import m1_vertexMove_Va001
        reload(gym)
        import gym        
except:
    loaded_5 = True
    print("loaded l25 in _data", loaded_5)
    reload(vertexMove_update)
    import m1_vertexMove_Va001    
    reload(gym)
    import gym

def env_creator(**kwargs):
    print("l33 in _data")
    import gym
    reload(gym)
    import gym
    object = gym.make('m1_vertexMove_Va003', g=9.81, **kwargs)#vertexMove-V002, vertexMove-V1, vertexMove-V003
    print("l36 in _data")
    return object
    
if __name__ == "__main__":
    env = env_creator(**kwargs)
    env.reset()
    # env.render()
    # import time

    # time.sleep(100)
