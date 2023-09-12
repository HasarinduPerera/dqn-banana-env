[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


# DQN Banana Environment (Unity)

Welcome to the project repository! This README provides instructions on how to use this repository to train an agent to navigate and collect bananas in a large, square world.

![Trained Agent][image1]

## Project Details

In this project, the agent learns to navigate the environment and collect yellow bananas while avoiding blue bananas. The environment is a large, square world where the agent moves around. When the agent collects a yellow banana, it receives a reward of +1, and if it collects a blue banana, it receives a reward of -1. The objective is to maximize the number of yellow bananas collected.

The state space has 37 dimensions and includes the agent's velocity and perception of objects in its forward direction. Based on this information, the agent needs to select the best action. There are four discrete actions available:

- 0: Move forward
- 1: Move backward
- 2: Turn left
- 3: Turn right

The task is episodic, and to consider the environment solved, the agent must achieve an average score of +13 over 100 consecutive episodes.


## Getting Started

To get started with this repository, follow the instructions below to install the necessary dependencies and set up the project environment.

### Prerequisites

- Python 3.6.13
- PyTorch 0.4.0
- Unityagents 0.4.0

### Installation

1. Clone the repository to your local machine using the following command:

   ```
   git clone https://github.com/HasarinduPerera/dqn-banana-env
   ```

2. Change into the project directory:

   ```
   cd dqn-banana-env
   ```

3. Start the project without any additional work as the required environment, "Banana.app," is already uploaded in this project.

## Instructions

To train and test the agent, follow the instructions below.

1. Make sure you have completed the installation steps mentioned above.

2. Open the `Navigation.ipynb` notebook. It serves as the entry point for the project and contains two modes: one for training and one for testing.

3. If you already have a pre-trained model, make sure you have the `checkpoint.pth` file in your project directory. This file saves the weights of the trained model.

4. If you want to train the DQN-Agent, run the training mode in the `Navigation.ipynb` notebook. This will train the agent using reinforcement learning techniques.

5. If you only want to test the agent using a pre-trained model, load the `checkpoint.pth` file and start the test mode in the `Navigation.ipynb` notebook. This will evaluate the agent's performance in the environment.

Alternatively, you can use the `Navigation.py` file if you prefer not to use a Jupyter Notebook. It contains the same code as in the `Navigation.ipynb` notebook.

Congratulations! You have successfully trained and tested the agent in the project environment. Feel free to explore the code, experiment with different configurations, and adapt it to your specific requirements.

If you have any questions or encounter any issues while using this repository, please don't hesitate to open an issue.

Happy navigating and banana collecting!
