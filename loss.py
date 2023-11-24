import torch

class MaskedMSELoss(torch.nn.Module):
    '''
    By user yangxi at:
    https://discuss.pytorch.org/t/how-to-write-a-loss-function-with-mask/53461/4
    '''
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input_, target, mask):
        diff2 = (torch.flatten(input_) - torch.flatten(target)) ** 2.0 * torch.flatten(mask.repeat(1,3,1,1))
        result = torch.sum(diff2) / torch.sum(mask)
        return result