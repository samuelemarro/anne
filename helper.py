import pathlib
import torch
import torchvision
import utils

from PIL import Image

def get_default_args():
    return dict(
        BN_eps=1e-05,
        D_B1=0.0,
        D_B2=0.999,
        D_attn='64',
        D_ch=64,
        D_depth=1,
        D_fp16=False,
        D_init='ortho',
        D_lr=0.0002,
        D_mixed_precision=False,
        D_nl='relu',
        D_ortho=0.0,
        D_param='SN',
        D_wide=True,
        G_B1=0.0,
        G_B2=0.999,
        G_attn='64',
        G_batch_size=0,
        G_ch=64,
        G_depth=1,
        G_eval_mode=False,
        G_fp16=False,
        G_init='ortho',
        G_lr=5e-05,
        G_mixed_precision=False,
        G_nl='relu',
        G_ortho=0.0,
        G_param='SN',
        G_shared=False,
        SN_eps=1e-08,
        accumulate_stats=False,
        adam_eps=1e-08,
        augment=False,
        base_root='',
        batch_size=64,
        config_from_name=False,
        cross_replica=False,
        data_root='data',
        dataset='I128_hdf5',
        dim_z=128,
        ema=False,
        ema_decay=0.9999,
        ema_start=0,
        experiment_name='',
        hashname=False,
        hier=False,
        load_in_mem=False,
        load_weights='',
        log_D_spectra=False,
        log_G_spectra=False,
        logs_root='logs',
        logstyle='%3.3e',
        model='BigGAN',
        mybn=False,
        name_suffix='',
        no_fid=False,
        norm_style='bn',
        num_D_SV_itrs=1,
        num_D_SVs=1,
        num_D_accumulations=1,
        num_D_steps=2,
        num_G_SV_itrs=1,
        num_G_SVs=1,
        num_G_accumulations=1,
        num_best_copies=2,
        num_epochs=100,
        num_inception_images=50000,
        num_save_copies=2,
        num_standing_accumulations=16,
        num_workers=8,
        parallel=False,
        pbar='mine',
        pin_memory=True,
        resume=False,
        sample_inception_metrics=False,
        sample_interps=False,
        sample_npz=False,
        sample_num_npz=50000,
        sample_random=False,
        sample_sheet_folder_num=-1,
        sample_sheets=False,
        sample_trunc_curves='',
        samples_root='samples',
        save_every=2000,
        seed=0,
        shared_dim=0,
        shuffle=False,
        skip_init=False,
        split_D=False,
        sv_log_interval=10,
        test_every=5000,
        toggle_grads=True,
        use_ema=False,
        use_multiepoch_sampler=False,
        weights_root='weights',
        which_best='IS',
        which_train_fn='GAN',
        z_var=1.0)
    

def sample_images(weights_path, batch_size, out_folder):
    out_folder = pathlib.Path(out_folder)
    config = get_default_args()

    # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
    config['batch_size'] = batch_size
    config["resolution"] = utils.imsize_dict["I128_hdf5"]
    config["n_classes"] = utils.nclass_dict["I128_hdf5"]
    config["G_activation"] = utils.activation_dict["inplace_relu"]
    config["D_activation"] = utils.activation_dict["inplace_relu"]
    config["G_attn"] = "64"
    config["D_attn"] = "64"
    config["G_ch"] = 96
    config["D_ch"] = 96
    config["hier"] = True
    config["dim_z"] = 120
    config["shared_dim"] = 128
    config["G_shared"] = True
    config = utils.update_config_roots(config)
    config["skip_init"] = True
    config["no_optim"] = True
    config["device"] = "cuda"

    # Seed RNG.
    utils.seed_rng(config["seed"])

    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True

    # Import the model.
    model = __import__(config["model"])
    G = model.Generator(**config).to(config["device"])
    utils.count_parameters(G)

    # Load weights.
    G.load_state_dict(torch.load(weights_path))

    # Update batch size setting used for G.
    G_batch_size = max(config["G_batch_size"], config["batch_size"])
    (z_, y_) = utils.prepare_z_y(
        G_batch_size,
        G.dim_z,
        config["n_classes"],
        device=config["device"],
        fp16=config["G_fp16"],
        z_var=config["z_var"],
    )

    G.eval()

    if not out_folder.exists():
        out_folder.mkdir(parents=True)

    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        image_tensors = G(z_, G.shared(y_))
        image_tensors = torch.tensor(image_tensors.cpu().numpy())
        for i, img in enumerate(image_tensors):
            img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            img = Image.fromarray(img)
            img.save(out_folder / f'{i}.jpg', format=None)