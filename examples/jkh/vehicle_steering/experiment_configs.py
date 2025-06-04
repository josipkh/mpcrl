"""This file contains the configurations for different experiments."""
configs = {
    "small_learn": {
        "vehicle_size": "small",
        "use_learned_parameters": False,
        "learning_rate": 1e-5,
        "max_episode_steps": 10_000,
    },
    "large_learn": {
        "vehicle_size": "large",
        "use_learned_parameters": False,
        "learning_rate": 1e-5,
        "max_episode_steps": 10_000,
    },
    "large_transfer": {
        "vehicle_size": "large",
        "use_learned_parameters": True,
        "learning_rate": 1e-5,
        "max_episode_steps": 10_000,
    },
    "test": {
        "vehicle_size": "large",
        "use_learned_parameters": False,
        "learning_rate": 1e-5,
        "max_episode_steps": 10_000,
    }
}
