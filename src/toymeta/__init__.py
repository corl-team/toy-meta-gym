from gymnasium.envs import register

__version__ = "0.0.1"

register(
    id="Dark-Room-9x9-v0",
    entry_point="toymeta.dark_room:DarkRoom",
    max_episode_steps=20,
    kwargs={
        "size": 9,
        "random_start": False,
        "terminate_on_goal": False,
    },
)

register(
    id="Dark-Room-3x3-v0",
    entry_point="toymeta.dark_room:DarkRoom",
    max_episode_steps=10,
    kwargs={
        "size": 3,
        "random_start": False,
        "terminate_on_goal": False,
    },
)


register(
    id="Dark-Room-Hard-17x17-v0",
    entry_point="toymeta.dark_room:DarkRoom",
    max_episode_steps=20,
    kwargs={
        "size": 17,
        "random_start": False,
        "terminate_on_goal": True,
    },
)


register(
    id="Dark-Key-To-Door-9x9-v0",
    entry_point="toymeta.dark_key_to_door:DarkKeyToDoor",
    max_episode_steps=50,
    kwargs={"size": 9, "random_start": True},
)

register(
    id="Dark-Key-To-Door-3x3-v0",
    entry_point="toymeta.dark_key_to_door:DarkKeyToDoor",
    max_episode_steps=20,
    kwargs={"size": 3, "random_start": True},
)

register(
    id="BernoulliBandit",
    entry_point="toymeta.bernoulli_bandit:MultiArmedBanditBernoulli",
)

register(
    id="HAD-Dark-Room",
    entry_point="toymeta.had_dark_room:HAD_DarkRoom",
    max_episode_steps=20,
    kwargs={"terminate_on_goal": True},
)

register(
    id="ContextualBandit", entry_point="toymeta.contextual_bandit:ContextualBandit"
)
