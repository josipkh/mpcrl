"""This script generates the parameters for dynamically similar cartpole systems, given a reference one.

Dynamic matching is based on eq. (28) in this paper: https://ieeexplore.ieee.org/document/10178119
which contains five parameters: pole length, cart mass, pole mass, cart friction and gravity.

It is assumed that the friction and gravity cannot be changed.
"""

from math import sqrt

ref_cartpole_params = {
    "l": 0.5,       # [m]
    "m_c": 1.0,     # [kg]
    "m_p": 0.5,     # [kg]
    "mu_f": 1.0,    # [N/(m/s)]
    "g": 9.81,      # [m/s^2]
}

# we need to match two Pi-groups:
# Pi_1 = m_p / m_c
# Pi_2 = mu_f / m_c * sqrt(l / g)

# first, choose a new pole length
l = 10 * ref_cartpole_params["l"]

# then, choose m_c to match Pi_2
m_c = ref_cartpole_params["m_c"] * sqrt(l / ref_cartpole_params["l"])

# finally, choose m_p to match Pi_1
m_p = m_c * ref_cartpole_params["m_p"] / ref_cartpole_params["m_c"]

# verify that the Pi-groups match
print("New Pi-groups:")
print(f"Pi_1: {m_p / m_c:.2f} (should match {ref_cartpole_params['m_p'] / ref_cartpole_params['m_c']:.2f})")
print(f"Pi_2: {ref_cartpole_params['mu_f'] / m_c * sqrt(l / ref_cartpole_params['g']):.2f} "
      f"(should match {ref_cartpole_params['mu_f'] / ref_cartpole_params['m_c'] * sqrt(ref_cartpole_params['l'] / ref_cartpole_params['g']):.2f})")
print(20*"=")

# print the new parameters
print(f"New cartpole parameters:")
print(f"l = {l:.2f} m")
print(f"m_c = {m_c:.2f} kg")
print(f"m_p = {m_p:.2f} kg")
print(f"mu_f = {ref_cartpole_params['mu_f']:.2f} N/(m/s)")
print(f"g = {ref_cartpole_params['g']:.2f} m/s^2")