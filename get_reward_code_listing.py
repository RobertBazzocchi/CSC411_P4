def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  :    0,    
            Environment.STATUS_INVALID_MOVE: -150, 
            Environment.STATUS_WIN         :  100,
            Environment.STATUS_TIE         :    0,   
            Environment.STATUS_LOSE        : -100
    }[status]