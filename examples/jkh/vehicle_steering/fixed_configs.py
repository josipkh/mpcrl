"""This file contains the fixed configurations for repeating different experiments."""
configs = {
    "small_learn": {
        "dimensionless": True,
        "maneuver": "straight",
        "env": {
            "vehicle_size": "small",
            "road_bank_angle": 0.479433,
        },
        "use_learned_parameters": False,
        "learning_rate": 1e-5,
        "episodes": 1,
        "max_episode_steps": 10_000,
    },
    "large_learn": {
        "dimensionless": True,
        "maneuver": "straight",
        "env": {
            "vehicle_size": "large",
            "road_bank_angle": 5.0,
        },
        "use_learned_parameters": False,
        "learning_rate": 1e-5,
        "episodes": 1,
        "max_episode_steps": 10_000,
    },
    "large_transfer": {
        "dimensionless": True,
        "maneuver": "straight",
        "env": {
            "vehicle_size": "large",
            "road_bank_angle": 5.0,
        },
        "use_learned_parameters": True,
        "learning_rate": 1e-5,
        "episodes": 1,
        "max_episode_steps": 10_000,
    },
}
