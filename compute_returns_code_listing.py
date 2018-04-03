def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 
    """
    n = 0
    G = []
    while n < len(rewards):
        G.append(0)  
        for i in range(n,len(rewards)):
            G[n] += (gamma**(i-n))*rewards[i]
        n+=1

    return G