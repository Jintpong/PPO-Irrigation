# PPO-Irrigation

To run the experiment, the user must access the terminal and execute the training and testing scripts. 

# Training 
The training is handled by train.py, which uses the PPO algorithm to train an irrigation agent in the Wheat environment located in aquacrogymnasium/environment.py. It requires a weather file, which is the champion_climate.txt located in the weather_data folder which saves trained models. The user can specify arguments such as total timesteps, random seed, output directory. The user can use TensorBoard logging to view the training of the agent. During training, the console displays rollout logs, and TensorBoard (if enabled) can be used to visualise learning curves.

To run the training script:

On macOS/Linux: python3 train.py

On Windows: python train.py

# Evaluation/Testing
Once training is complete, the evaluation is performed using test.py, which loads the trained PPO models and their corresponding normalisers, evaluates them for a number of episodes, and compares their performance against several baselines including Random, Thresholds, Interval, and Rainfed strategies. The results are saved in the eval_output directory as CSV files containing detailed episode results, aggregated averages, and comparison summaries. The script also produces plots that show the trade-offs between irrigation water applied and crop yield, as well as irrigation efficiency. 

To run the testing script:

On macOS/Linux: python3 test.py

On Windows: python test.py
