import copy
import os
from time import perf_counter

from typing import List, Tuple
import imageio
import numpy as np
import PIL.Image
import click

import torch
import torch.nn.functional as F
import functools

import dnnlib
from dnnlib.util import format_time
import legacy

from torch_utils import gen_utils
from tqdm import tqdm

from metrics import metric_utils

def project(
        G,
        target: PIL.Image.Image,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        projection_seed: int,
        truncation_psi: float,
        num_steps: int = 1000,
        w_avg_samples: int = 10000,
        initial_learning_rate: float = 0.1,
        initial_noise_factor: float = 0.05,
        constant_learning_rate: bool = False,
        lr_rampdown_length: float = 0.25,
        lr_rampup_length: float = 0.05,
        noise_ramp_length: float = 0.75,
        regularize_noise_weight: float = 1e5,
        project_in_wplus: bool = False,
        loss_paper: str = 'sgan2',  # ['sgan2' || Experimental: 'im2sgan' | 'clip' | 'discriminator']
        normed: bool = False,
        sqrt_normed: bool = False,
        start_wavg: bool = True,
        device: torch.device,
        D = None) -> Tuple[torch.Tensor, dict]:  # output shape: [num_steps, C, 512], C depending on resolution of G
    """
    Copied from:
    https://github.com/PDillis/stylegan3-fun/blob/main/projector.py

    Projecting a 'target' image into the W latent space. The user has an option to project into W+, where all elements
    in the latent vector are different. Likewise, the projection process can start from the W midpoint or from a random
    point, though results have shown that starting from the midpoint (start_wavg) yields the best results.
    """
    assert target.size == (G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats.
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    if project_in_wplus:  # Thanks to @pbaylies for a clean way on how to do this
        print('Projecting in W+ latent space...')
        if start_wavg:
            print(f'Starting from W midpoint using {w_avg_samples} samples...')
            w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, L, C]
        else:
            print(f'Starting from a random vector (seed: {projection_seed})...')
            z = np.random.RandomState(projection_seed).randn(1, G.z_dim)
            w_avg = G.mapping(torch.from_numpy(z).to(device), None)  # [1, L, C]
            w_avg = G.mapping.w_avg + truncation_psi * (w_avg - G.mapping.w_avg)
    else:
        print('Projecting in W latent space...')
        w_samples = w_samples[:, :1, :]  # [N, 1, C]
        if start_wavg:
            print(f'Starting from W midpoint using {w_avg_samples} samples...')
            w_avg = torch.mean(w_samples, dim=0, keepdim=True)  # [1, 1, C]
        else:
            print(f'Starting from a random vector (seed: {projection_seed})...')
            z = np.random.RandomState(projection_seed).randn(1, G.z_dim)
            w_avg = G.mapping(torch.from_numpy(z).to(device), None)[:, :1, :]  # [1, 1, C]; fake w_avg
            w_avg = G.mapping.w_avg + truncation_psi * (w_avg - G.mapping.w_avg)
    w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    # Setup noise inputs (only for StyleGAN2 models)
    noise_buffs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image. Reshape to 256x256 if it's larger to use with VGG16 (unnecessary for CLIP due to preprocess step)
    if loss_paper in ['sgan2', 'im2sgan', 'discriminator']:
        target = np.array(target, dtype=np.uint8)
        target = torch.tensor(target.transpose([2, 0, 1]), device=device)
        target = target.unsqueeze(0).to(device).to(torch.float32)
        if target.shape[2] > 256:
            target = F.interpolate(target, size=(256, 256), mode='area')

    if loss_paper in ['sgan2', 'im2sgan']:
        # Load the VGG16 feature detector.
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
        vgg16 = metric_utils.get_feature_detector(url, device=device)

    # Define the target features and possible new losses
    if loss_paper == 'sgan2':
        target_features = vgg16(target, resize_images=False, return_lpips=True)

    elif loss_paper == 'im2sgan':
        # Use specific layers
        vgg16_features = VGG16FeaturesNVIDIA(vgg16)
        # Too cumbersome to add as command-line arg, so we leave it here; use whatever you need, as many times as needed
        layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
                  'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc1', 'fc2', 'fc3']
        target_features = vgg16_features.get_layers_features(target, layers, normed=normed, sqrt_normed=sqrt_normed)
        # Uncomment the next line if you also want to use LPIPS features
        # lpips_target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        mse = torch.nn.MSELoss(reduction='mean')
        ssim_out = SSIM()  # can be used as a loss; recommended usage: ssim_loss = 1 - ssim_out(img1, img2)

    elif loss_paper == 'discriminator':
        disc = DiscriminatorFeatures(D).requires_grad_(False).to(device)

        layers = ['b128_conv0', 'b128_conv1', 'b64_conv0', 'b64_conv1', 'b32_conv0', 'b32_conv1',
                  'b16_conv0', 'b16_conv1', 'b8_conv0', 'b8_conv1', 'b4_conv']

        target_features = disc.get_layers_features(target, layers, normed=normed, sqrt_normed=sqrt_normed)
        mse = torch.nn.MSELoss(reduction='mean')
        ssim_out = SSIM()

    elif loss_paper == 'clip':
        import clip
        model, preprocess = clip.load('ViT-B/32', device=device)  # TODO: let user decide which model to use (use list given by clip.available_models()

        target = preprocess(target).unsqueeze(0).to(device)
        # text = either we give a target image or a text as target
        target_features = model.encode_image(target)

        mse = torch.nn.MSELoss(reduction='mean')

    w_opt = w_avg.clone().detach().requires_grad_(True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_buffs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        if constant_learning_rate:
            # Turn off the rampup/rampdown of the learning rate
            lr_ramp = 1.0
        else:
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        if project_in_wplus:
            ws = w_opt + w_noise
        else:
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        if loss_paper == 'sgan2':
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_buffs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight
            # Print in the same line (avoid cluttering the commandline)
            n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
            message = f'step {step + 1:{n_digits}d}/{num_steps}: dist {dist:.7e} | loss {loss.item():.7e}'
            print(message, end='\r')

            last_status = {'dist': dist.item(), 'loss': loss.item()}

        elif loss_paper == 'im2sgan':
            # Uncomment to also use LPIPS features as loss (must be better fine-tuned):
            # lpips_synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

            synth_features = vgg16_features.get_layers_features(synth_images, layers, normed=normed, sqrt_normed=sqrt_normed)
            percept_error = sum(map(lambda x, y: mse(x, y), target_features, synth_features))

            # Also uncomment to add the LPIPS loss to the perception error (to-be better fine-tuned)
            # percept_error += 1e1 * (lpips_target_features - lpips_synth_features).square().sum()

            # Pixel-level MSE
            mse_error = mse(synth_images, target) / (G.img_channels * G.img_resolution * G.img_resolution)
            ssim_loss = ssim_out(target, synth_images)  # tracking SSIM (can also be added the total loss)
            loss = percept_error + mse_error  # + 1e-2 * (1 - ssim_loss)  # needs to be fine-tuned

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_buffs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss += reg_loss * regularize_noise_weight
            # We print in the same line (avoid cluttering the commandline)
            n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
            message = f'step {step + 1:{n_digits}d}/{num_steps}: percept loss {percept_error.item():.7e} | ' \
                      f'pixel mse {mse_error.item():.7e} | ssim {ssim_loss.item():.7e} | loss {loss.item():.7e}'
            print(message, end='\r')

            last_status = {'percept_error': percept_error.item(),
                           'pixel_mse': mse_error.item(),
                           'ssim': ssim_loss.item(),
                           'loss': loss.item()}

        elif loss_paper == 'discriminator':
            synth_features = disc.get_layers_features(synth_images, layers, normed=normed, sqrt_normed=sqrt_normed)
            percept_error = sum(map(lambda x, y: mse(x, y), target_features, synth_features))

            # Also uncomment to add the LPIPS loss to the perception error (to-be better fine-tuned)
            # percept_error += 1e1 * (lpips_target_features - lpips_synth_features).square().sum()

            # Pixel-level MSE
            mse_error = mse(synth_images, target) / (G.img_channels * G.img_resolution * G.img_resolution)
            ssim_loss = ssim_out(target, synth_images)  # tracking SSIM (can also be added the total loss)
            loss = percept_error + mse_error  # + 1e-2 * (1 - ssim_loss)  # needs to be fine-tuned

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_buffs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss += reg_loss * regularize_noise_weight
            # We print in the same line (avoid cluttering the commandline)
            n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
            message = f'step {step + 1:{n_digits}d}/{num_steps}: percept loss {percept_error.item():.7e} | ' \
                      f'pixel mse {mse_error.item():.7e} | ssim {ssim_loss.item():.7e} | loss {loss.item():.7e}'
            print(message, end='\r')

            last_status = {'percept_error': percept_error.item(),
                           'pixel_mse': mse_error.item(),
                           'ssim': ssim_loss.item(),
                           'loss': loss.item()}

        elif loss_paper == 'clip':

            import torchvision.transforms as T
            synth_img = F.interpolate(synth_images, size=(224, 224), mode='area')
            prep = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            synth_img = prep(synth_img)
            # synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]  # NCWH => WHC
            # synth_images = preprocess(PIL.Image.fromarray(synth_images, 'RGB')).unsqueeze(0).to(device)
            synth_features = model.encode_image(synth_img)
            dist = mse(target_features, synth_features)

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_buffs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight
            # Print in the same line (avoid cluttering the commandline)
            n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
            message = f'step {step + 1:{n_digits}d}/{num_steps}: dist {dist:.7e}'
            print(message, end='\r')

            last_status = {'dist': dist.item(), 'loss': loss.item()}

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_buffs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # Save run config
    run_config = {
        'optimization_options': {
            'num_steps': num_steps,
            'initial_learning_rate': initial_learning_rate,
            'constant_learning_rate': constant_learning_rate,
            'regularize_noise_weight': regularize_noise_weight,
        },
        'projection_options': {
            'w_avg_samples': w_avg_samples,
            'initial_noise_factor': initial_noise_factor,
            'lr_rampdown_length': lr_rampdown_length,
            'lr_rampup_length': lr_rampup_length,
            'noise_ramp_length': noise_ramp_length,
        },
        'latent_space_options': {
            'project_in_wplus': project_in_wplus,
            'start_wavg': start_wavg,
            'projection_seed': projection_seed,
            'truncation_psi': truncation_psi,
        },
        'loss_options': {
            'loss_paper': loss_paper,
            'vgg16_normed': normed,
            'vgg16_sqrt_normed': sqrt_normed,
        },
        'elapsed_time': '',
        'last_commandline_status': last_status
    }

    if project_in_wplus:
        return w_out, run_config  # [num_steps, L, C]
    return w_out.repeat([1, G.mapping.num_ws, 1]), run_config  # [num_steps, 1, C] => [num_steps, L, C]

def run_projection(
        network_pkl,
        target_fname,
        outloc,
        w_outloc,
        num_steps=1000,
        initial_learning_rate=0.1,
        constant_learning_rate=False,
        regularize_noise_weight=1e5,
        seed=20,
        stabilize_projection=True,
        project_in_wplus=False,
        start_wavg=True,
        projection_seed=20,
        truncation_psi=.9,
        loss_paper='sgan2', #['sgan2', 'im2sgan', 'discriminator', 'clip']
        normed=True,
        sqrt_normed=False,
        device='cuda' #Change to 'cpu' if not running on GPU
):
    """
    Derived from:
    https://github.com/PDillis/stylegan3-fun/blob/main/projector.py

    Project given image to the latent space of pretrained network pickle.
    """
    torch.manual_seed(seed)

    # If we're not starting from the W midpoint, assert the user fed a seed to start from
    if not start_wavg:
        if projection_seed is None:
            print('Provide a seed to start from if not starting from the midpoint. Use "--projection-seed" to do so')
            sys.exit()

    # Other losses are still broken, sorry!
    #if loss_paper != 'sgan2':
    #    print("All losses other than the o.g. StyleGAN2 is still in experimental phase. Sorry!")
    #    loss_paper = 'sgan2'

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    if loss_paper == 'discriminator':
        # We must also load the Discriminator
        with dnnlib.util.open_url(network_pkl) as fp:
            D = legacy.load_network_pkl(fp)['D'].requires_grad_(False).to(device)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Stabilize the latent space to make things easier (for StyleGAN3's config t and r models)
    if stabilize_projection:
        gen_utils.anchor_latent_space(G)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, run_config = project(
        G,
        target=target_pil,
        num_steps=num_steps,
        initial_learning_rate=initial_learning_rate,
        constant_learning_rate=constant_learning_rate,
        regularize_noise_weight=regularize_noise_weight,
        project_in_wplus=project_in_wplus,
        start_wavg=start_wavg,
        projection_seed=projection_seed,
        truncation_psi=truncation_psi,
        loss_paper=loss_paper,
        normed=normed,
        sqrt_normed=sqrt_normed,
        device=device,
        D=D if loss_paper == 'discriminator' else None
    )
    elapsed_time = format_time(perf_counter()-start_time)
    print(f'\nElapsed time: {elapsed_time}')

    # Save only the final projected frame and W vector.
    print('Saving projection results...')
    projected_w = projected_w_steps[-1]
    synth_image = gen_utils.w_to_img(G, dlatents=projected_w, noise_mode='const')[0]
    PIL.Image.fromarray(synth_image, 'RGB').save(outloc)

    projected_w = projected_w_steps[-1]
    np.save(w_outloc, projected_w.unsqueeze(0).cpu().numpy())

    return w_outloc

def _parse_cols(s: str, G, device: torch.device, truncation_psi: float) -> List[torch.Tensor]:
    """s can be a path to a npy/npz file or a seed number (int)"""
    s = s.split(',')
    w = torch.Tensor().to(device)
    for el in s:
        if os.path.isfile(el):
            w_el = gen_utils.get_w_from_file(el)  # np.ndarray
            w_el = torch.from_numpy(w_el).to(device)  # torch.tensor
            w = torch.cat((w_el, w))
        else:
            nums = gen_utils.num_range(el)
            for n in nums:
                w = torch.cat((gen_utils.get_w_from_seed(G, device, n, truncation_psi), w))
    return w


def generate_style_mix(
        network_pkl,
        outloc,
        name,
        w_input_loc,
        w_base_loc,
        col_ws,
        style_run=False,
        style_names=None,
        style_w_input_locs=None,
        style_col_ws=None,
        truncation_psi=0.6,
        noise_mode='const', #['const', 'random', 'none']
        anchor_latent_space=False,
        device='cuda',
):
    """
    Derived from:
    https://github.com/PDillis/stylegan3-fun/blob/main/style_mixing.py

    Generate style-mixing images using pretrained network pickle.
    """

    save_loc = outloc + "/" + name +  ".png"

    # TODO: add class_idx
    #print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Setup for using CPU
    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Sanity check: loaded model and selected styles must be compatible
    max_style = G.mapping.num_ws
    if max(col_ws) > max_style:
        click.secho(f'WARNING: Maximum col-style allowed: {max_style - 1} for loaded network "{network_pkl}" '
                    f'of resolution {G.img_resolution}x{G.img_resolution}', fg='red')
        click.secho('Removing col-styles exceeding this value...', fg='blue')
        col_ws[:] = [style for style in col_ws if style < max_style]

    base_w = _parse_cols(w_base_loc, G, device, truncation_psi)
    input_w = _parse_cols(w_input_loc, G, device, truncation_psi)
    all_w = torch.cat((base_w, input_w))

    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi

    out_w = all_w[1]
    out_w[col_ws] = all_w[0][col_ws]
    final_image = gen_utils.w_to_img(G, out_w, noise_mode).squeeze(0)
    PIL.Image.fromarray(final_image, 'RGB').save(save_loc)

    if style_run:

        for i in range(len(style_names)):

            col_style = style_col_ws[i]
            w_style_loc = style_w_input_locs[i]
            save_style_loc = outloc + "/" + name + "_" + str(style_names[i]) + ".png"

            style_w = _parse_cols(w_style_loc, G, device, truncation_psi)
            all_style_w = torch.cat((all_w, style_w))
            all_style_w = w_avg + (all_style_w - w_avg) * truncation_psi

            out_w_style = out_w.clone()
            out_w_style[col_style] = all_style_w[2][col_style]
            style_image = gen_utils.w_to_img(G, out_w_style, noise_mode).squeeze(0)
            PIL.Image.fromarray(style_image, 'RGB').save(save_style_loc)

    return save_loc

def generate_sprite_w_space(
        network_pkl,
        out_name,
        dest_w,
        dest_sprite,
        truncation_psi=.9,
        noise_mode='const',
        seed=None,
        device='cuda'):
    """
    Derived from:
    https://github.com/NVlabs/stylegan3/blob/main/gen_images.py

    Generate a random w-space for our character sprites
    """

    #device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    if seed==None:
        seed = np.random.randint(3000000)

    #Generate random z-space
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    #Generate sprite image and save for visual comparison
    #Our example does not use labels, but the logic is included for future work
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    #else:
    #    if class_idx is not None:
    #        print ('warn: --class=lbl ignored when running on an unconditional network')
    if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(dest_sprite+"/"+out_name+".png")

    #Use the above z-spce to find the w-space
    w = G.mapping(z, None)
    w_avg = G.mapping.w_avg
    w = w_avg + (w - w_avg) * truncation_psi
    #Save w-space to .npy file to be read in for style mixing
    np.save(dest_w+"/"+out_name+".npy", w.cpu().numpy())
