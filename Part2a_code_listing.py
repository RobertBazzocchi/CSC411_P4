class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(27, 64)
        self.fc2 = torch.nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x,dim=1) 