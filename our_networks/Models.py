import torch.nn as nn
import torch.nn.functional as F

# Based on https://github.com/trypag/pytorch-unet-segnet/blob/master/segnet.py,
# Implement real down/up sampling from SegNet as in the link

class SegNet(nn.Module):

	def __init__(self, num_classes, n_init_features, drop_rate, filters=(64, 128)):
		super(SegNet, self).__init__()

		self.encoders = nn.ModuleList()
		self.decoders = nn.ModuleList()

		encoder_filters = (n_init_features,) + filters

		decoder_filters = filters[::-1] + n_init_features 
		
		for i in range(0, 2):
			self.encoders.append(_Encoder(encoder_filters[i],
					encoder_filters[i+1],
					 drop_rate))

			self.decoders.append(_Decoder(decoder_filters[i],
					decoder_filters[i+1],
					 drop_rate)

		self.classifier = nn.Conv3D( n_init_feautres, num_classes, kernel_size=3, stride=1, padding=0, bias=True)

	def forward(self, features):
		
		indices = []
		unpool_sizes = []
		for i in range(0, 2):
			(features, ind), size = self.encoders[i](features)
			indices.append(ind)
			unpool_sizes.append(size)

		for i in range(0,2):
			features = self.decoders[i](features, indices[1-i], unpool_sizes[1-i])

		return self.classifier(features)


class _Encoder(nn.Module):
	
	def __init__(self, n_in_feat, n_out_feat, drop_rate=0.5)
		
		super(_Encoder, self).__init__()

		layers = [nn.Conv3D(n_in_feat, n_out_feat, kernel_size=3, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(n_out_feat),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(2)]

		self.features = nn.Sequential(*layers)

	def forward(self, x):
		output = self.features(x)
		return output, output.size()
		
		
		
class _Decoder(nn.Module):

	def __init__(self, n_in_feat, n_out_feat, drop_rate = 0.5):
		super(_Decoder, self).__init__()
		
		layers = [nn.Conv3d(n_in_feat, n_out_feat, kernel_size=3, stride=1, padding=0, bias=True),
			nn.BatchNorm3d(n_out_feat),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor = 2, mode='bilinear')]

		self.features = nn.Sequential(*layers)

	def forward(self, x, indices, size):
		return self.features(x)
		


		
