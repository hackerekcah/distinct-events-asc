
class ArgWrapper(dict):
    """
    wrapper argument, pass in a dict or key-value pairs
    subclass dict so could use dict functions and creations
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


if __name__ == '__main__':
    args = ArgWrapper(pooling='max', dropout=0.0)
    print(args)
