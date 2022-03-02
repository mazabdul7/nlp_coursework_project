import os
import glob
from utils.configs import config

class DataLoader:
    def __init__(self) -> None:
        self.base_path = config['data_path']
        self.mturk_path = config['mturk_path']
        self.trip_path = config['trip_path']
        
    def retrieve_file_list(self, path):
        files = [f for f in glob.glob(os.path.join(path, '*/*.txt')) if '.bi' not in f and '.uni' not in f]
        return files
        
    def load_mturk(self):
        path = os.path.join(self.base_path, self.mturk_path)
        file_list = self.retrieve_file_list(path)
        
        reviews = []
        for file in file_list:
            with open(file, 'r') as f:
                reviews.append(f.read().splitlines())
                
        return reviews
    
    def load_trip(self):
        ''' This one is in unigram format already so needs special parsing. '''
        pass
    
    def load_amazon(self):
        pass