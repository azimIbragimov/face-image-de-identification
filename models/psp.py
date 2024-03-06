"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math
from lpips import LPIPS


import torch
import torchvision
from torch import nn
from arcface_torch.backbones import get_model
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from pytorch_msssim import SSIM

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, id_loss_threshold, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
				inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0
		
		

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents)

		if not resize:
			images = self.face_pool(images)
		

		# CHANGES: START
		arcface = get_model("r50", fp16=False)
		arcface.load_state_dict(torch.load("backbone.pth"))
		arcface.cuda()
		arcface.eval()

		resize_transform = torchvision.transforms.Resize((112, 112))

		MSELoss = torch.nn.MSELoss()
		LPIPLoss = LPIPS(net='alex').cuda()
		SSIMLoss = SSIM()

		max_iter_number = 100
		adjustable_iter_number = 6

		w_apostrophe = [codes]
		for n in range(max_iter_number):
                    x_hat, _ = self.decoder([w_apostrophe[n]], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents)
                    x_resized, x_hat_resized = resize_transform(x), resize_transform(x_hat) 

                    # Calculating Delta W_1
               	    face_embeddings_x, face_embeddings_x_hat = arcface(x_resized).squeeze(), arcface(x_hat_resized).squeeze()	
               	    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    id_loss = (1 - cos(face_embeddings_x, face_embeddings_x_hat).mean())

                    grad_id_loss_wrt_w_apostrophe = torch.autograd.grad(id_loss, w_apostrophe[n])
               	    delta_w_1 = 0.02 * torch.sign(*grad_id_loss_wrt_w_apostrophe)
                
                    w_apostrophe[n] += delta_w_1
                    
                    for i in range(adjustable_iter_number):
                        x_hat, _ = self.decoder([w_apostrophe[n]], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents)
                        x_resized, x_hat_resized = resize_transform(x), resize_transform(x_hat) 

                        # Calculating Delta W_2
                        mse = MSELoss(x_resized, x_hat_resized)
                        lpips = LPIPLoss(x_resized, x_hat_resized)
                        ssim = SSIMLoss(x_resized, x_hat_resized)
                        mse_grad = torch.autograd.grad(mse, codes, retain_graph=True)[0]
                        lpips_grad = torch.autograd.grad(lpips, codes, retain_graph=True)[0]
                        ssim_grad = torch.autograd.grad(ssim, codes)[0]
                        delta_w_2 = - 0.008 * (torch.sign(mse_grad) + torch.sign(lpips_grad) + torch.sign(ssim_grad))
                                 
                        w_apostrophe[n] += delta_w_2
                    w_apostrophe.append(w_apostrophe[n] )
                    x_hat, _ = self.decoder([w_apostrophe[n]], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents)
                    x_resized, x_hat_resized = resize_transform(x), resize_transform(x_hat) 
                    face_embeddings_x, face_embeddings_x_hat = arcface(x_resized).squeeze(), arcface(x_hat_resized).squeeze()	
                    id_loss = (1 - cos(face_embeddings_x, face_embeddings_x_hat).mean())
    
                    if id_loss > id_loss_threshold:
                        break

		if return_latents:
			return x_hat, result_latent
		else:
			return x_hat
		# CHANGES: END


		if return_latents:
			return images, result_latent
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
