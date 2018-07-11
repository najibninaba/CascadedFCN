import os


class SimpleDataGenerator(object):
    '''
    Simple data generator that is able to read from a directory on disk

    Arguments:
        directory
    '''

    def __init__(self,
                 x_dir,
                 y_dir,
                 batch_size=32,
                 seed=None):

        self.x_dir = x_dir
        self.y_dir = y_dir
        self.batch_size = batch_size

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
