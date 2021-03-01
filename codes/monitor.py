import numpy as np
from collections import deque
import sys


def interact(env, agent, n_episodes, max_t, eps_start, eps_end, eps_decay):
    """Interaction between agent and environment.

    This function define the interaction between the agent and the openai gym
    environment, and printout the partial results

    Args:
        env: openai gym's environment
        agent: class agent to interact with the environment
        n_episodes: Integer. Maximum number of training episodes
        max_t: Integer. Maximum number of time-steps per episode
        eps_start: Float. Starting value of epsilon, for epsilon-greedy action selection
        eps_end: Float. Minimum value of epsilon
        eps_decay: Float. Multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    # Loop the define episodes
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0

        # Loop over the maximum number of time-steps per episode
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            # Break the loop if it is final state
            if done:
                break

        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', flush=True)
        sys.stdout.flush()

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')

            # Save model
            agent.model_local.save_weights('model/ddqn_weights')
            break

    return scores

