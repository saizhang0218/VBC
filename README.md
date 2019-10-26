1. This code is developed based on the source code of QMIX paper, which is available at: https://github.com/oxwhirl/pymarl

2. To run the code, please install the SMAC (StarCraft Multi-Agent Challenge) first, which is available at: https://github.com/oxwhirl/smac, please then follow the instructions to install the StarCraft II client. Download StarCraft II into the 3rdparty folder and copy the maps necessary to run over.

3. The requirements.txt file can be used to install the necessary packages into a virtual environment.

4. To run the code use the following command: python3 src/main.py --config=xxx_xxx --env-config=sc2 with env_args.map_name=xxx

--config can be one of the following four options: vdn_6h_vs_8z,vdn_corridor,qmix_6h_vs_8z,qmix_corridor (corridor is the 6z_vs_24zerg scenario).
For example 'vdn_6h_vs_8z' means 6h_vs_8z scenario with VDN

--env_args.map_name can be one of the following two options:6h_vs_8z,corridor (corridor is the 6z_vs_24zerg scenario)

5. All the hyperparameters can be found at:  src/config/default.yaml, src/config/algs/*.yaml and src/config/envs/*.yaml

6. The execution of the agent network can be found at: src/controllers/basic_controller_xxx_vbc.py, where xxx is the name of the map. For example, the execution of the agent network for 2s3z can be found at: src/controllers/basic_controller_2s3z_vbc.py

7. The learning of the agent network can be found at: src/learners/q_learner_xxx_vbc.py, where xxx is the name of the map. For example, the learning of the agent network for 2s3z can be found at: src/learners/q_learner_2s3z_vbc.py
     
8. Model can be saved by setting “save_model = True” in “src/config/default.yaml”. You can also specify the frequency of saving model by changing the “save_model_interval” option in the config file. 

9. To load model, specifying the path of the saved model by filling in the `checkpoint_path` parameter in the default.yaml.

10. The test accuracy will be saved in the 'xxx_accuracy_list.txt', where xxx is the local_results_path parameter in default.yaml.
