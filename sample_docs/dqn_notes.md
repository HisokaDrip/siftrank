# DQN notes
The agent stores transitions in a replay buffer and samples random minibatches, which breaks the correlation between consecutive frames. A target network is copied every few thousand steps so the bootstrap target doesn't chase itself. Double DQN picks the action with the online net and evaluates it with the target net, which cuts overestimation.
