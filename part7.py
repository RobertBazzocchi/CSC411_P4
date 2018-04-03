from tictactoe import *

def train(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    num_invalid_moves = 0
    num_wins = 0
    num_losses = 0
    num_ties = 0
    episode_data = np.array([])
    first_move_data = np.zeros((1,9))

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False

        if i_episode % log_interval == 0:
            pr_data = first_move_distr(policy,env)
            episode_data = np.append(episode_data, [i_episode])
            first_move_data = np.vstack([first_move_data,np.array(pr_data)])
        
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)
            if len(saved_rewards) > 1000:
                break

        if status == "win":
            num_wins += 1
        if status == "lose":
            num_losses += 1
        if status == "tie":
            num_ties += 1
        
        for reward in saved_rewards:
            if reward == Invalid_reward:
                num_invalid_moves += 1

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            print("Wins: {}, Losses: {}, Ties: {}, Invalid Moves: {}\n".format(num_wins,num_losses,num_ties,num_invalid_moves))
            num_wins,num_losses,num_ties,num_invalid_moves = 0,0,0,0
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # STOP AFTER 50000 ITERATIONS
        if i_episode == 50000:
	        first_move_data = first_move_data[1:,:]
	        print(episode_data.shape)
	        print(first_move_data[:,0].shape)
	        for i in range(9):
	            plt.plot(episode_data,first_move_data[:,i])
	            plt.xlabel('Episodes')
	            plt.ylabel('Probability')
	            plt.title('Probability Distribution of {} being the first move'\
	            	.format(i))
	            plt.show()
	        break

if __name__ == '__main__':
    import sys
    env = Environment()
    policy = Policy()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env, log_interval=100)
        pr_data = first_move_distr(policy,env)
        # test(policy,env)

    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))