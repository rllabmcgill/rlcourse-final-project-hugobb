# l2communicate

Project by Tom Bosc and Hugo Berard

- lib/rpg_agent/: contains everything related to RPG as well as an agent that doesn't learn and take random actions uniformely. rpg.py and rpg_ep.py are preliminary versions (with constant baseline, with recurrent baseline but no communication) of the complete RPG algorithm with actions and communications output which is in rpg_ep_com.py.
- lib/env/: contains the environment

To train the different models:

` python train_multiagent_mdp_mlp.py [output]`

by default all the results are saved in `results/`, you can change this by specifying `output`.

To plot the results run:

`python plot_results.py [input] [output]`

by default looks into `results/` and save figures in `report`.
