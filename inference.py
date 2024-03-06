import os
from argparse import Namespace
from argparse import ArgumentParser
from torchvision import transforms
import torch
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from configs import data_configs
from datasets.inference_dataset import InferenceDatasetSingle
from utils.common import tensor2im, log_input_image
from models.psp import pSp

def run(img, id_loss_threshold):
    test_opts = {
        "checkpoint_path": os.path.join(sys.path[0], "psp_ffhq_encode.pt"),
        "id_loss_threshold": id_loss_threshold, 
        "latent_mask" : None, 
        "mix_alpha" : None, 
        "n_images" : None, 
        "n_outputs_to_generate" : 5,
        "resize_factors" :None,
        "resize_outputs":False,
        "test_batch_size":1,
        "test_workers":1
    }
    # update test options with options used during training
    ckpt = torch.load(test_opts['checkpoint_path'], map_location='cpu')
    opts = ckpt['opts']
    for o in opts:
        if opts[o] != None:
            if o not in test_opts:
                test_opts[o] = opts[o]
            elif test_opts[o] == None:
                test_opts[o] = opts[o]

    if 'learn_in_w' not in test_opts:
        test_opts['learn_in_w'] = False
    if 'output_size' not in test_opts:
        test_opts['output_size'] = 1024
    opts = Namespace(**test_opts)
    if opts.n_images is None:
        opts.n_images = 1

    net = pSp(opts)
    net.eval()
    net.cuda()

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDatasetSingle(img,
                              transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            )

    for img in dataloader:
    	img = img.cuda().float()
    	result_batch = run_on_batch(img, net, opts, opts.id_loss_threshold)
    	return result_batch



def run_on_batch(inputs, net, opts, id_loss_threshold):

    if opts.latent_mask is None:
        result_batch = net(inputs, id_loss_threshold, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # print(type(input_image))
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      id_loss_threshold,
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      id_loss_threshold,
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)


    return result_batch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to the image')
    parser.add_argument('--id_loss', type=float, help="Threshold value for id_loss")
    parser.add_argument('--out', type=str, help="Path of the output image") 
    parser = parser.parse_args()
    img = Image.open(parser.img)
    transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
    img = transform(img)
    img = run(img, parser.id_loss)[0].detach().cpu()
    img = tensor2im(img)
    Image.fromarray(np.array(img)).save(parser.out)
    
