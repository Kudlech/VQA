"""
Here, we create a custom dataset
"""
import torch
import time
import pickle
import json
from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from compute_softscore import preprocess_answer


UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]
MAX_QUESTION_LEN = 20

class VQADataset(Dataset):
	"""
	VQA dataset
	"""
	def __init__(self, path_answers: PathT, path_image: PathT, path_questions: PathT, word_dict=None) -> None:
		# Set variables
		self.path_answers = path_answers
		self.path_image = path_image
		self.path_questions = path_questions

		# if word_dict is already built, it's val dataset
		self.is_val = False if word_dict is None else True

		# Load Q&A
		self.questionsAnswers = self._get_questions_answers()

		# Set picture size
		self.pic_size = 224

		if word_dict is None:
			self.word_dict = defaultdict(int)
			# Create vocabs of entries
			self.get_vocabs()
		else:
			self.word_dict = word_dict
		self.word_idx_mappings, self.idx_word_mappings = self.init_word_vocab(self.word_dict)
		self.vocab_size = len(self.idx_word_mappings)

		# Create list of entries
		self.entries = self._get_entries()

	def __getitem__(self, index: int) -> Tuple:
		"""
		:param index:
		:return: item's image, question, question len, labels
		"""
		path = self.path_image +str(self.entries[index]['image_id']).zfill(12)+'.jpg'
		image = self._get_images(path)
		return image, self.entries[index]['question'],self.entries[index]['question_len'], self.entries[index]['labels'].to_dense()

	def __len__(self) -> int:
		"""
		:return: the length of the dataset (number of sample).
		"""
		return len(self.entries)


	def _get_questions_answers(self) -> Any:
		"""
		Load all features into a structure
		:return:
		:rtype:
		"""
		with open(self.path_answers, "rb") as features_file:
			features = pickle.load(features_file)
		questions_answers = {item['question_id']: item for item in features}

		with open(self.path_questions, "rb") as f:
			features_questions = json.load(f)['questions']

		for item in features_questions:
			questions_answers[item['question_id']]['question'] = preprocess_answer(item['question'])

		return questions_answers

	def init_word_vocab(self, word_dict):
		"""
		Creating word vocabulary
		:return: list of mapping from idx to word
		"""
		idx_word_mappings = sorted([token for token in SPECIAL_TOKENS])
		word_idx_mappings = {token: idx_word_mappings.index(token) for token in idx_word_mappings}
		for i, word in enumerate(sorted(word_dict.keys())):
			word_idx_mappings[str(word)] = int(i + len(SPECIAL_TOKENS))
			idx_word_mappings.append(str(word))
		return word_idx_mappings, idx_word_mappings

	def _get_entries(self) -> List:
		"""
		This function create a list of all the entries. We will use it later in __getitem__
		:return: list of samples
		"""
		entries = []
		for item in self.questionsAnswers.values():
			entries.append(self._get_entry(item))

		return entries

	def get_vocabs(self):
		"""
			return frequencies dict
		"""
		for item in self.questionsAnswers.values():
			for word in item['question'].split():
				self.word_dict[word] += 1

	def _get_entry(self, item: Dict) -> Dict:
		"""
		:item: item from the data.
		"""

		labels_tensor = torch.tensor([item['labels']], requires_grad=False, dtype=torch.int64)
		scores_tensor = torch.tensor(item['scores'], requires_grad=False, dtype=torch.float32)
		labels = torch.sparse.FloatTensor(labels_tensor, scores_tensor, torch.Size([2410])) # 2410 - num of labels

		words_idx_list = []
		for idx, word in enumerate(item['question'].split()):  # going over the comment content
			if idx >= MAX_QUESTION_LEN:
				break
			# map word to index
			words_idx_list.append(self.word_idx_mappings.get(word, self.word_idx_mappings.get(UNKNOWN_TOKEN)))

		# pad the end of the tensor
		for i in range(len(item['question'].split()), MAX_QUESTION_LEN):
			words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
		question_tensor = torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False)
		question_len = len(item['question'].split())
		return {'image_id': item['image_id'], 'question': question_tensor,'question_len':question_len, 'labels': labels}


	def _get_images(self, path):
		try:
			image = Image.open(path)
			normalize = transforms.Normalize(mean=[0.4783, 0.4493, 0.4075],
											 std=[0.1214, 0.1191, 0.1429])
			if self.is_val:
				transform = transforms.Compose([
					transforms.Resize(255),
					transforms.CenterCrop(self.pic_size),
					transforms.ToTensor(),
				])
			else:
				transform = transforms.Compose([
					transforms.Resize(255),
					transforms.CenterCrop(self.pic_size),
					#transforms.RandomHorizontalFlip(p=0.1),
					#transforms.RandomVerticalFlip(p=0.1),
					transforms.ToTensor(),
				])
			tensor_image = transform(image)

			# if the image is greyscale change it to rgb representation
			if tensor_image.size(0) == 1:
				tensor_image = tensor_image.repeat(3, 1, 1)
			return tensor_image #normalize(tensor_image)

		# sometimes an image is locked by other students, so catch the exception, wait a second and repeat
		except:
			print("entered into the except :(")
			time.sleep(1)
			return self._get_images(path)
