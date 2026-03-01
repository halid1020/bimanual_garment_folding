class Dummy:
    
    def __init__(self, config=None):
        pass

    def __call__(self, sample, train=True, device='cpu'):    
        return sample