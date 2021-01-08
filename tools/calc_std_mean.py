import torch
from tqdm import tqdm

def calc_std_mean(loader):
	sum_image = torch.zeros(3)
	std_image = torch.zeros(3)
	if torch.cuda.is_available():
		std_image = std_image.cuda()
		sum_image = sum_image.cuda()
	for i, (image, question, question_len, label) in tqdm(enumerate(loader),
														   total=len(loader)):
		if torch.cuda.is_available():
			image = image.cuda()
		sum_image += torch.sum(image, [0,2,3]) / (loader.dataset.pic_size**2)
	mean_image = sum_image / len(loader.dataset)
	for i, (image, question, question_len, label) in tqdm(enumerate(loader),
														  total=len(loader)):
		if torch.cuda.is_available():
			image = image.cuda()
		std_image += torch.sum(((torch.sum(image, [2,3])/(loader.dataset.pic_size**2)) - mean_image)**2, 0)
	std_image = torch.sqrt(std_image / len(loader.dataset))
	print(f"std: {std_image}, mean: {mean_image}")
	return std_image, mean_image