# Succinct and Robust Multi-Agent Communication With Temporal Message Control

This repository is the official implementation of [Succinct and Robust Multi-Agent Communication With Temporal Message Control]. 

> 📋A video demo is available at: https://tmcpaper.github.io/tmc/

## Requirements
> 📋To run the code, please install the SMAC (StarCraft Multi-Agent Challenge) first, which is available at: https://github.com/oxwhirl/smac, please then follow the instructions to install the StarCraft II client. Download StarCraft II into the 3rdparty folder and copy the maps necessary to run over.

> 📋This code is developed based on the source code of QMIX, which is available at: https://github.com/oxwhirl/pymarl.

> 📋 To install the requirements, run:
```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python3 src/main.py --config=xxx --env-config=sc2 with env_args.map_name=xxx
```

> 📋--config can be one of the following six options: qmix_3s_vs_4z, qmix_3s_vs_5z, qmix_6h_vs_8z, qmix_corridor(corridor is 6z_vs_24zg scenario), qmix_2c_vs_64zg, qmix_3s5z. For example 'qmix_6h_vs_8z' means 6h_vs_8z map with QMIX as the mixing network.

> 📋--env_args.map_name can be one of the following six options: 3s_vs_4z, 3s_vs_5z, 6h_vs_8z, corridor (corridor is the 6z_vs_24zg scenario), 2c_vs_64zg, 3s5z. 


## Code Explanation
> 📋All the hyperparameters can be found and modified at:  `src/config/default.yaml`, `src/config/algs/*.yaml` and `src/config/envs/*.yaml`. The default parameters are for 3s_vs_4z scenario.

> 📋The test accuracy at each timestep will be saved in the 'accuracy' folder.

> 📋The test reward at each timestep will be saved in the 'reward' folder.

> 📋The agent network can be found at: `src/controllers/basic_controller_xxx.py`, where xxx is the name of the map. For example, the execution of the agent network for 3s_vs_4z can be found at: `src/controllers/basic_controller_3s_vs_4z.py`

> 📋The training of agent networks can be found at: `src/learners/q_learner_xxx.py`, where xxx is the name of the map. For example, the training of the agent network for 3s_vs_4z can be found at: `src/learners/q_learner_3s_vs_4z.py`

> 📋Model can be saved by setting `save_model = True` in `src/config/default.yaml`. You can also specify the frequency of the saving by changing the `save_model_interval` option in the config file. 

## Execution
> 📋To execute the model, specifying the path of the saved model by filling the `checkpoint_path` and setting `evaluate: True` in the `src/config/default.yaml`.


