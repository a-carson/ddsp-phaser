import torch


class MLP(torch.nn.Module):
    def __init__(self, in_features=1,
                 out_features=1,
                 width=8,
                 n_hidden_layers=1,
                 activation='tanh',
                 bias=True):
        super(MLP, self).__init__()

        self.model = torch.nn.Sequential()

        for n in range(n_hidden_layers):
            self.model.append(torch.nn.Linear(in_features=in_features, out_features=width, bias=bias))
            if activation == 'tanh':
                self.model.append(torch.nn.Tanh())
            else:
                self.model.append(torch.nn.ReLU())
            in_features = width

        self.model.append(torch.nn.Linear(in_features=width, out_features=out_features, bias=bias))

    # requires input shape (L, 1) where L is sequence length
    def forward(self, x):
        y = self.model(x)
        return y.view(x.shape)