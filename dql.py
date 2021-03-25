import gym
import numpy as np
from tensorflow import keras
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def DQL(alpha, gamma, slopesilon, testing=False):
    """---Deep Q-learning with Experience Replay---"""
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v0")
    eas = env.action_space.n
    eos = env.observation_space.shape[0]

    epsilon = 1.0
    batch_size = 64
    R_history = []

    # Initialize replay memory D to capacity N
    D = deque(maxlen=100000)

    # Initialize action-value function Q with random weights
    Q = q_net(alpha, eas, eos)

    # Loop for each episode
    for episode in range(1000):
        # Initialize the starting state S
        S = env.reset(); S = [S]; S = np.asarray(S)

        # Loop for each step of episode
        step = 0
        R_episodic = 0
        moves = 0
        while step == 0:
            env.render()

            # Select action A per epsilon greedy policy
            A = epsilon_greedy(Q, S, eas, epsilon)

            # Execute the selected action and observe R, S'
            S2, R, step, info = env.step(A); S2 = [S2]; S2 = np.asarray(S2)

            R_episodic += R

            # Store transition in D
            D.append((S, A, R, S2, step))

            # S <- S'
            S = S2

            if len(D) >= batch_size and np.mean(R_history[-10:]) < 200 and np.random.random() < 0.2:
                # Sample random mini batch of transitions
                D_arr = np.asarray(D)
                sample_xp = D_arr[np.random.choice(D_arr.shape[0], batch_size, replace=False), :]
                sample_A, sample_R, sample_S, sample_S2, sample_i, sample_info = vectorize(batch_size, sample_xp)

                # Set y_j
                Y0 = Q.predict_on_batch(sample_S)
                Y = sample_R + gamma * (np.amax(Q(sample_S2), axis=1)) * (1 - sample_info)
                Y0[[sample_i], [sample_A]] = Y

                # Perform gradient descent step --update the Q network
                Q.fit(sample_S, Y0, epochs=1, verbose=0)

            moves += 1
            if moves > 750:
                break

        if avg(R_history) > 200 and not testing:
            break

        print(episode, " reward:", R_episodic, " avg_reward:", avg(R_history))
        episode += 1

        epsilon = max(0.01, epsilon * slopesilon)

        R_history.append(R_episodic)

    return Q, R_history


def epsilon_greedy(Q, S, eas, epsilon):
    # With probability epsilon, select a random action --EXPLORATION
    if np.random.random() < epsilon:
        A = np.random.randint(eas)
    # otherwise select greedy action --EXPLOITATION
    else:
        A = np.argmax(Q(S)[0])
    return A


def q_net(alpha, eas, eos):
    Q = keras.Sequential(
        [
            keras.layers.Dense(600, activation="relu", input_shape=(eos,)),
            keras.layers.Dense(400, activation="relu"),
            keras.layers.Dense(eas, activation="linear"),
        ]
    )
    Q.compile(optimizer=keras.optimizers.Adam(lr=alpha), loss=keras.losses.MeanSquaredError(), run_eagerly=False)
    return Q


def vectorize(batch_size, sample_xp):
    sample_S = reformat(sample_xp, 0).squeeze()
    sample_A = reformat(sample_xp, 1)
    sample_R = reformat(sample_xp, 2)
    sample_S2 = reformat(sample_xp, 3).squeeze()
    sample_info = reformat(sample_xp, 4)
    sample_i = reformat(batch_size, special=True)
    return sample_A, sample_R, sample_S, sample_S2, sample_i, sample_info


def reformat(_list, _u=0, special=False):
    if special:
        return np.array([i for i in range(_list)])
    else:
        return np.array([i[_u] for i in _list])


def avg(_list):
    if len(_list) == 0:
        return 0
    elif len(_list) < 100:
        ave = 0
        for i in range(len(_list)):
            ave += _list[i]
        return ave / len(_list)
    else:
        return np.mean(_list[-100:])


def test(agent):
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v0")
    R_history = []
    for i in range(100):
        S = env.reset()
        S = [S];
        S = np.asarray(S)
        step = 0
        R_episodic = 0
        while step == 0:
            env.render()
            A = np.argmax(agent(S)[0])
            S2, R, step, info = env.step(A)
            R_episodic += R
            S2 = [S2];
            S2 = np.asarray(S2)
            S = S2
        R_history.append(R_episodic)
    return R_history


def test_alpha():
    alphas = [0.0001, 0.001, 0.01]
    R_hist = []
    for a in alphas:
        LL = Agent(alpha=a, gamma=0.99, slopesilon=0.995, testing=True)
        LL.dql()
        R_hist.append(LL.rewards)
    return R_hist


def test_gamma():
    gammas = [0.99, 1.0, 0.9]
    R_hist = []
    for g in gammas:
        LL = Agent(alpha=0.001, gamma=g, slopesilon=0.995, testing=True)
        LL.dql()
        R_hist.append(LL.rewards)
    return R_hist


def test_slopesilon():
    slopesilons = [0.99, 1.0, 0.9]
    R_hist = []
    for s in slopesilons:
        LL = Agent(alpha=0.001, gamma=0.99, slopesilon=s, testing=True)
        LL.dql()
        R_hist.append(LL.rewards)
    return R_hist


class Agent:
    def __init__(self, alpha, gamma, slopesilon, testing=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.slopesilon = slopesilon

        self.agent = None
        self.rewards = None
        self.testing = testing

    def dql(self):
        self.agent, self.rewards = DQL(self.alpha, self.gamma, self.slopesilon, self.testing)


def main():
    # Initialize and train the agent
    lunar_lander = Agent(alpha=0.001, gamma=0.99, slopesilon=0.995)
    lunar_lander.dql()

    # Plot rewards during training
    plt.plot([i for i in range(len(lunar_lander.rewards))],lunar_lander.rewards)
    plt.ylabel("Reward"); plt.xlabel("Episode"); plt.savefig("fig1.png"); plt.close()

    # Plot rewards after training
    plt.plot([i for i in range(100)],test(lunar_lander.agent))
    plt.ylabel("Reward"); plt.xlabel("Episode"); plt.savefig("fig2.png"); plt.close()

    # Effect of learning rate
    x = test_alpha()
    for t in range(3):
        plt.plot([i for i in range(len(x[0]))],[c for c in x[t]])
    plt.ylabel("Reward"); plt.xlabel("Episodes"); plt.gca().legend(("α =0.0001", "α =0.001", "α =0.01")); plt.ylim(-1000); plt.savefig("fig3.png")

    # Effect of discounting rate
    x = test_gamma()
    for t in range(3):
        plt.plot([i for i in range(len(x[0]))],[c for c in x[t]])
    plt.ylabel("Reward"); plt.xlabel("Episodes"); plt.gca().legend(("γ =0.99", "γ =1.0", "γ = 0.9")); plt.ylim(-1000); plt.savefig("fig4.png")

    # Effect of Ɛ-decay constant
    x = test_slopesilon()
    for t in range(4):
        plt.plot([i for i in range(len(x[0]))],[c for c in x[t]],alpha=0.82)
    plt.ylabel("Reward"); plt.xlabel("Episodes"); plt.gca().legend(("Ɛ-decay =1.0", "Ɛ-decay =0.9999", "Ɛ-decay =0.995", "Ɛ =0")); plt.ylim(-600); plt.savefig("fig5.png")


if __name__ == '__main__':
    main()