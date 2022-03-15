import os
from pathlib import Path
import pandas as pd
from utils.configs import config
from typing import List


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

	@staticmethod
	def list_all_txt_files(path) -> list:
		""" Retrieve all dataset files in the specified path, recursively.

		Args:
			path (str): Path to search.

		Returns:
			List[str]: List of all found data files.
		"""
		files = [p for p in Path(path).rglob('*.txt')]
		return files

	def load_gold_data(self, _type: str, neg_polarity=True, pos_polarity=True) -> list:
		# Possibly change it to return pandas to be same as amazon?
		""" Loads data from the 'GOLD' standard datasets

		Args:
			_type (str): Choose whether to load deceptive ('dec') or authentic ('truth') dataset.
			neg_polarity (bool, optional): Return negative polarity reviews.
			pos_polarity (bool, optional): Return positive polarity reviews.

		Returns:
			List[str]: List of dataset reviews as strings.
		"""
		if type not in self.types.keys():
			raise Exception('Type is incorrect...')

		polarities = []
		file_list = []
		if neg_polarity:
			polarities.append(self.neg_path)
		if pos_polarity:
			polarities.append(self.pos_path)
		paths = [os.path.join(self.base_path, config['gold_path'], pol_path, self.types[_type]) for pol_path in
		         polarities]
		for path in paths:
			file_list.extend(self.list_all_txt_files(path))

		reviews = []
		for file in file_list:
			with open(file, 'r') as f:
				reviews.extend(f.read().splitlines())

		return reviews

	def load_amazon(self, deceptive=False, all=False) -> pd.DataFrame:
		""" Loads data from the Amazon dataset. Label 1 (1) is deceptive reviews, Label 2 (0) is authentic.

		Args:
			deceptive (bool, optional): Return deceptive instead of authentic.

		Returns:
			Dataframe: Returns a dataframe of the reviews.
		"""
		data_path = self.list_all_txt_files(os.path.join(self.base_path, config['amazon_path']))[0]
		data = pd.read_table(data_path)
		data.loc[data['LABEL'] == '__label2__', 'LABEL'] = 0
		data.loc[data['LABEL'] == '__label1__', 'LABEL'] = 1

		if all:
			return data
		else:
			if deceptive:
				return data.get(data['LABEL'] == 1)
			return data.get(data['LABEL'] == 0)
