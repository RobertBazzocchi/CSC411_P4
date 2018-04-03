from tictactoe import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

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
    episode_track = []
    win_track = []
    lose_track = []
    tie_track = []

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False

        # if i_episode % log_interval == 0: print(select_action(policy,state)[0])
        
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)


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
            episode_track.append(i_episode)
            win_track.append(num_wins)
            tie_track.append(num_ties)
            lose_track.append(num_losses)
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
            return episode_track, win_track, lose_track, tie_track


if __name__ == '__main__': 
	env = Environment()
	policy = Policy()

	if len(sys.argv) == 1:
	    # `python tictactoe.py` to train the agent
	    episode_track, win_track, lose_track, tie_track = train(policy, env)

	else:
	    # `python tictactoe.py <ep>` to print the first move distribution
	    # using weightt checkpoint at episode int(<ep>)
	    ep = int(sys.argv[1])
	    load_weights(policy, ep)
	    print(first_move_distr(policy, env))

	plt.plot(episode_track, win_track, 'r', \
		episode_track, lose_track, 'b', \
		episode_track, tie_track, 'g')
	red_patch = mpatches.Patch(color='red', label='Wins')
	blue_patch = mpatches.Patch(color='blue', label='Losses')
	green_patch = mpatches.Patch(color='green', label='Ties')
	plt.legend(handles=[red_patch, blue_patch, green_patch])
	plt.xlabel('Episodes')
	plt.ylabel('Performance over 1000 games')
	plt.title('Performance of the Tictactoe Policy')
	plt.show()