 Neural Mouse AI

Neural Mouse AI is a team project developed to explore reinforcement learning in a practical way. As a group of three, we wanted to better understand how a Deep Q-Network (DQN) learns through interaction with an environment and how it performs against a traditional pathfinding algorithm.

To do this, we built a simple maze game where a mouse learns to reach a piece of cheese while avoiding a cat. The mouse is controlled by a Deep Q-Network trained with reinforcement learning, while the cat uses the A* pathfinding algorithm to chase the player. This setup gave us a straightforward way to compare a learned policy with a rule-based approach.

  Features

* Deep Q-Network (DQN) built with PyTorch
* Experience Replay and Target Network
* Epsilon-greedy exploration
* A* pathfinding enemy
* Manual play mode
* AI training mode
* Trained model playback
* Save and load trained models

  Technologies

* Python
* PyTorch
* Pygame
* NumPy

 Running the Project

pip install pygame torch numpy
python main.py

From the main menu you can:

* Play the game manually
* Train the reinforcement learning agent
* Watch the trained AI play

  What We Learned

This project was created by a team of three as part of our effort to learn reinforcement learning through a hands-on implementation rather than relying only on theory. Our goal was to build the complete system ourselves and better understand the strengths and limitations of Deep Q-Learning when applied to a simple game environment.
