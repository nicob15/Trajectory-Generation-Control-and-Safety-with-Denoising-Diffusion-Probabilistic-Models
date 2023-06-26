import pickle
import os

class Logger():
    def __init__(self, folder, filename='dataset.pkl'):
        self.directory = folder
        self.filename = filename
        # make sure the folder exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.datalog = []

    def obslog(self, data):
        self.datalog.append(data)

    def save_obslog(self, filename='dataset.pkl', folder=''):
        if folder == '':
            folder = self.directory
        with open(folder + filename, 'wb') as f:
            pickle.dump(self.datalog, f, pickle.HIGHEST_PROTOCOL)

