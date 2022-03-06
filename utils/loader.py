import os
import glob
import pandas as pd
from utils.configs import config

class DataLoader:
    """ Interface loader for loading datasets. """
    def __init__(self) -> None:
        self.base_path = config['base_path']
        self.pos_path = config['pos_path']
        self.neg_path = config['neg_path']
        self.types = {
            'dec': config['dec_path'],
            'truth': config['truth_path']
        }
        
    def retrieve_file_list(self, path):
        """ Retrieve all dataset files in the specified path, recursively.

        Args:
            path (str): Path to search.

        Returns:
            List[str]: List of all found data files.
        """
        files = [f for f in glob.glob(os.path.join(path, '*.txt'), recursive=True) if '.bi' not in f and '.uni' not in f]
        return files
        
    def load_gold_data(self, type, neg_polarity=False, pos_polarity=True):
        """ Loads data from the 'GOLD' standard datasets

        Args:
            type (str): Choose whether to load deceptive ('dec') or authentic ('truth') dataset.
            neg_polarity (bool, optional): Return negative polarity reviews.
            pos_polarity (bool, optional): Return positive polarity reviews.

        Returns:
            List[str]: List of dataset reviews as strings.
        """
        if type not in self.types.keys():
            assert Exception('Type is incorrect...')
    
        polarities = []
        file_list = []
        if neg_polarity: polarities.append(self.neg_path)
        if pos_polarity: polarities.append(self.pos_path)
        paths = [os.path.join(self.base_path, config['gold_path'], pol_path, self.types[type]) for pol_path in polarities]
        for path in paths:
            file_list.extend(self.retrieve_file_list(path))
        
        reviews = []
        for file in file_list:
            with open(file, 'r') as f:
                reviews.append(f.read().splitlines())
                
        return reviews

    def load_amazon(self, deceptive=False, all=False):
        """ Loads data from the Amazon dataset. Label 1 (1) is deceptive reviews, Label 2 (0) is authentic.

        Args:
            deceptive (bool, optional): Return deceptive instead of authentic.

        Returns:
            Dataframe: Returns a dataframe of the reviews.
        """
        data_path = self.retrieve_file_list(os.path.join(self.base_path, config['amazon_path']))[0]
        data = pd.read_table(data_path)
        data.loc[data['LABEL'] == '__label2__', 'LABEL'] = 1
        data.loc[data['LABEL'] == '__label1__', 'LABEL'] = 0
        
        if all: 
            return data
        else:
            if deceptive:
                return data.get(data['LABEL'] == 1)
            return data.get(data['LABEL'] == 0)