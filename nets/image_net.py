"""
Classifier for example.
"""


from abc import ABCMeta
from torch import Tensor, nn
from torch.nn.utils.weight_norm import weight_norm


class ImageNet(nn.Module, metaclass=ABCMeta):
	"""
	Classifier for example.
	Usage example:
	Classifier(1024, 512, 2, 0.2) - for a fully with weight normalization and dropout of 0.2. Out dimension of 0.2.
	"""
	def __init__(self, in_dim: int, out_dim: int, dropout: float):
		super(ImageNet, self).__init__()
		self.conv_layer = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=5, padding=2),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.MaxPool2d(4),

			nn.Conv2d(8, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			# nn.Dropout2d(0.3),
			nn.MaxPool2d(4),

			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			# nn.Dropout2d(0.3),
			nn.MaxPool2d(2),
		)

		# self.fc = nn.Linear(int((in_dim/32)**2 * 32), out_dim)

	def forward(self, image: Tensor) -> Tensor:
		"""
		Forward x through the classifier
		:param x:
		:return: logits
		"""
		image_tensor = self.conv_layer(image)
		# return self.fc(image_tensor.view(image.size(0), -1))
		return image_tensor