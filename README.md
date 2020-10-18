## Pytorch implementation of "Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control"

This is the github repo for the work "Succinct and Robust Multi-Agent CommunicationWith Temporal Message Control" published in NeurIPS 2019 (https://arxiv.org/abs/1909.02682). A video demo is available at: https://bit.ly/2VFkvCZ.

### Prerequisites
1. To run the code, please install the SMAC (StarCraft Multi-Agent Challenge) first, which is available at: https://github.com/oxwhirl/smac, please then follow the instructions to install the StarCraft II client. Download StarCraft II into the 3rdparty folder and copy the maps necessary to run over.

2. This code is developed based on the source code of QMIX paper, which is available at: https://github.com/oxwhirl/pymarl.

3. The requirements.txt file can be used to install the necessary packages into a virtual environment.

### Run the code

1. To run the code, use the following command: 

```
python3 src/main.py --config=xxx_xxx --env-config=sc2 with env_args.map_name=xxx
```
--config can be one of the following four options: vdn_6h_vs_8z,vdn_corridor,qmix_6h_vs_8z,qmix_corridor (corridor is 6z_vs_24zerg scenario). For example 'vdn_6h_vs_8z' means 6h_vs_8z map with VDN as the mixing network.

--env_args.map_name can be one of the following two options:6h_vs_8z,corridor (corridor is the 6z_vs_24zerg scenario)

2. All the hyperparameters can be found at:  src/config/default.yaml, src/config/algs/*.yaml and src/config/envs/*.yaml

3. The test accuracy will be saved in the 'xxx_accuracy_list.txt', where xxx is the local_results_path parameter in default.yaml.

4. Communication overhead \beta will be saved in the 'xxx_comm_overhead.txt', where xxx is the local_results_path parameter in default.yaml.
