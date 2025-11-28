import torch
from random import randint
from tqdm.rich import trange
from tqdm import tqdm as tqdm
from source.networks import Warper3DGS
import wandb
import sys

sys.path.append('./submodules/gaussian-splatting/')
import lpips
from source.losses import ssim, l1_loss, psnr
from rich.console import Console
from rich.theme import Theme
import multiprocessing as mp
import numpy as np
import gc
from PIL import Image as PILImage
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})
import cv2
import os


from source.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from source.utils_aux import log_samples

from source.timer import Timer
from torch.nn import Linear as Linear

class MLP(torch.nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.device = device
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def train(self, X, Y, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

class EDGSTrainer:
    def __init__(self,
                 GS: Warper3DGS,
                 training_config,
                 dataset_white_background=False,
                 device=torch.device('cuda'),
                 log_wandb=True,
                 ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.gaussians = GS.gaussians

        self.training_config = training_config
        self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = training_config.save_iterations
        self.evaluate_iterations = None
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        # Logs in the format {step:{"loss1":loss1_value, "loss2":loss2_value}}
        self.logs_losses = {}
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.device = device
        self.timer = Timer()
        self.log_wandb = log_wandb

    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            self.gs.gaussians.restore(
                torch.load(f"{load_cfg.gs}/chkpnt{load_cfg.gs_step}.pth")[0],
                self.training_config)
            self.GS_optimizer = self.GS.gaussians.optimizer
            self.CONSOLE.print(f"3DGS loaded from checkpoint for iteration {load_cfg.gs_step}",
                               style="info")
            self.training_step += load_cfg.gs_step
            self.gs_step += load_cfg.gs_step

    def train(self, train_cfg):
        # 3DGS training
        self.CONSOLE.print("Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info")    
        with trange(self.training_step, self.training_step + train_cfg.gs_epochs, desc="[green]Train gaussians") as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify)
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if self.gs_step < self.training_config.densify_until_iter and self.gs_step % 10 == 0:
                            opacities_new = torch.log(torch.exp(self.GS.gaussians._opacity.data) * 0.99)
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix({"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"}, refresh=True)
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (self.training_step <= 3000 and self.training_step % 500 == 0) or \
                            (self.training_step > 3000 and self.training_step % 1000 == 228) :
                            self.evaluate()

                    self.timer.start()


    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras(), 'cam_idx': self.training_config.TEST_CAM_IDX_TO_LOG},
                              {'name': 'train',
                               'cameras': [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in
                                           range(0, 150, 5)], 'cam_idx': 10})
        if self.log_wandb:
            wandb.log({f"Number of Gaussians": len(self.GS.gaussians._xyz)}, step=self.training_step)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(self.GS(viewpoint)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += self.lpips(image, gt_image).detach().double()
                    if idx in [config['cam_idx']]:
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_splat_test /= len(config['cameras'])
                if self.log_wandb:
                    wandb.log({f"{config['name']}/L1": l1_test.item(), f"{config['name']}/PSNR": psnr_test.item(), \
                            f"{config['name']}/SSIM": ssim_test.item(), f"{config['name']}/LPIPS_splat": lpips_splat_test.item()}, step = self.training_step)
                self.CONSOLE.print("\n[ITER {}], #{} gaussians, Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                    self.training_step, len(self.GS.gaussians._xyz), config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_splat_test.item()), style="info")
        if self.log_wandb:
            with torch.no_grad():
                log_samples(torch.stack((log_real_images[0],log_gen_images[0])) , [], self.training_step, caption="Real and Generated Samples")
                wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr = False, no_densify = False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
      
        render_pkg = self.GS(viewpoint_cam=viewpoint_cam)
        image = render_pkg["render"]
        # Loss
        gt_image = viewpoint_cam.original_image.to(self.device)
        L1_loss = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - self.training_config.lambda_dssim) * L1_loss + \
               self.training_config.lambda_dssim * ssim_loss
        self.timer.pause() 
        self.logs_losses[self.training_step] = {"loss": loss.item(),
                                                "L1_loss": L1_loss.item(),
                                                "ssim_loss": ssim_loss.item()}
        
        if self.log_wandb:
            for k, v in self.logs_losses[self.training_step].items():
                wandb.log({f"train/{k}": v}, step=self.training_step)
        self.ema_loss_for_log = 0.4 * self.logs_losses[self.training_step]["loss"] + 0.6 * self.ema_loss_for_log
        self.timer.start()
        self.GS_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if self.gs_step < self.training_config.densify_until_iter and not no_densify:
                self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]])
                self.GS.gaussians.add_densification_stats(render_pkg["viewspace_points"],
                                                                     render_pkg["visibility_filter"])

        # Optimizer step
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True)
        return render_pkg["radii"]

    def densify_and_prune(self, radii = None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if (self.gs_step > self.training_config.densify_from_iter) and \
                    (self.gs_step % self.training_config.densification_interval == 0):
                size_threshold = 20 if self.gs_step > self.training_config.opacity_reset_interval else None
                self.GS.gaussians.densify_and_prune(self.training_config.densify_grad_threshold,
                                                               0.005,
                                                               self.GS.scene.cameras_extent,
                                                               size_threshold, radii)
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                    self.dataset_white_background and self.gs_step == self.training_config.densify_from_iter):
                self.GS.gaussians.reset_opacity()             

          

    def save_model(self):
        print("\n[ITER {}] Saving Gaussians".format(self.gs_step))
        self.scene.save(self.gs_step)
        print("\n[ITER {}] Saving Checkpoint".format(self.gs_step))
        torch.save((self.GS.gaussians.capture(), self.gs_step),
                self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")


    def save_data_with_names(self, image_names, W2Cs, orig_maps, pred_maps, save_path):
    # Prepare dictionary for np.savez
        save_dict = {}
        for i, name in enumerate(image_names):
            # Clean image name to be a valid key (remove extension or replace dots)
            key_w2c = f"{name}_W2C"
            key_orig = f"{name}_orig"
            key_pred = f"{name}_pred"

            save_dict[key_w2c] = W2Cs[i]
            save_dict[key_orig] = orig_maps[i]
            save_dict[key_pred] = pred_maps[i]

        np.savez_compressed(save_path, **save_dict)

    def init_with_depth1(self, cfg, images_path, pcd, selected_indices, fx, fy, cx, cy, mde_model, N_epochs_MLP = 1000, npy_path= None, use_pcd = True, verbose=False):
        if not cfg.use:
            return None, None

        self.N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", self.N_splats_at_init)

        viewpoint_stack = self.scene.getTrainCameras().copy()
        selected_viewpoints = [viewpoint_stack[i] for i in selected_indices]
        resolution = 1.0
        device = self.device

        images = []
        est_depths = []
        depths = []
        masked_est_depths = []
        extrinsics = []
        W2Cs = []
        image_names = []

        K = np.array([[fx / resolution, 0, cx / resolution],
                    [0, fy / resolution, cy / resolution],
                    [0, 0, 1]])
        h, w = 0, 0

        # 1) Build training data for MLP
        for viewpoint in selected_viewpoints:
            image_names.append(viewpoint.image_name)
            pil_im = PILImage.open(f"{images_path}/{viewpoint.image_name}")
            cv2_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            pil_im.close()
            # del pil_im

            est_depth = mde_model.infer_image(cv2_im)  # HxW raw depth map in numpy
            h, w = est_depth.shape

            # Project point cloud to get ground truth depth
            W2C = viewpoint.world_view_transform.detach().cpu().numpy().T
            W2Cs.append(W2C)

            if use_pcd:
                # Project point cloud to get depth map
                homegeneous_coords = np.hstack((pcd, np.ones((pcd.shape[0], 1))))  # Nx4
                cam_coords = (W2C @ homegeneous_coords.T).T  # Nx4
                cam_coords = cam_coords[:, :3]
                pixel_coords = (K @ cam_coords.T).T  # Nx3
                u = pixel_coords[:, 0] / pixel_coords[:, 2]
                v = pixel_coords[:, 1] / pixel_coords[:, 2]
                depth = cam_coords[:, 2]  # Z in camera space

                
                valid_indices = np.where(
                    (depth > 0) &
                    (u >= 0) & (u < w) &
                    (v >= 0) & (v < h)
                )[0]
                valid_u = np.floor(u[valid_indices]).astype(int)
                valid_v = np.floor(v[valid_indices]).astype(int)
                valid_depth = depth[valid_indices].reshape(-1, 1)
            else: #get depth from npy
                npy_name = viewpoint.image_name.split('.')[0] + '.npy'
                uv_depth = np.load(os.path.join(npy_path, npy_name))
                valid_u = uv_depth[:, 0].astype(int)
                valid_v = uv_depth[:, 1].astype(int)
                valid_depth = uv_depth[:, 2].reshape(-1, 1)

            masked_est_depth = est_depth[valid_v, valid_u].reshape(-1, 1)

            # get extrinsic
            ext = viewpoint.world_view_transform.flatten().detach().cpu().numpy()
            ext_repeated = np.tile(ext, (len(valid_v), 1))

            est_depth_flat = est_depth.reshape(-1, 1)
            ext_repeated_est = np.tile(ext, (len(est_depth_flat), 1))
            est_depth_with_ext = np.hstack((est_depth_flat, ext_repeated_est))

            images.append(cv2_im)
            est_depths.append(est_depth_with_ext)
            depths.append(valid_depth)
            extrinsics.append(ext_repeated)
            masked_est_depths.append(masked_est_depth)

        all_masked_est_depths = np.vstack(masked_est_depths)
        all_gt_depths = np.vstack(depths)
        all_extrinsics = np.vstack(extrinsics)

        embeddings = np.hstack([all_masked_est_depths, all_extrinsics])



        # 2) Train MLP for depth correction
        input_dim = embeddings.shape[1]
        output_dim = all_gt_depths.shape[1]
        hidden_dim = 128

        mlp_model = MLP(device=device,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim)

        X = torch.tensor(embeddings, dtype=torch.float32).to(device)
        Y = torch.tensor(all_gt_depths, dtype=torch.float32).to(device)

        mlp_model.train(X, Y, epochs=N_epochs_MLP, lr=0.001)

        # # We no longer need training data on CPU

        orig_maps = []
        pred_maps = []

        with torch.no_grad():
            for test_X in est_depths:
                inv = test_X[:, 0]
                pseudo = 1.0 / (inv + 1e-6)
                orig_maps.append(pseudo.reshape(h, w))

                X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
                pred = mlp_model(X_tensor).cpu().numpy().reshape(h, w)
                pred_maps.append(pred)


        # Free MLP and training containers
        del mlp_model, est_depths, depths
        torch.cuda.empty_cache()
        # gc.collect()

        print('Done training MLP')
        
        # Save W2Cs and pred maps with image names
        save_path = os.path.join(self.scene.model_path, f"dense_init_depth_data.npz")
        self.save_data_with_names(image_names, W2Cs, orig_maps, pred_maps, save_path)
        print(f"Saved dense init depth data to {save_path}")


        # At this point, the worker process has exited and the OS has
        # reclaimed all RAM used by ScalableTSDFVolume inside dense_init_gaussians.
    
    def init_with_depth2(self, cfg, input_path):
        xyz_path = input_path + '/all_new_xyz.pt'
        features_dc_path = input_path + '/all_new_features_dc.pt'
        all_new_xyz = torch.load(xyz_path)
        all_new_features_dc = torch.load(features_dc_path)
        
        with torch.no_grad():
            N = all_new_xyz.shape[0]
            gaussians = self.gaussians
            all_new_features_rest = torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0)
            all_new_opacities = torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0)
            all_new_scaling = torch.stack([gaussians._scaling[-1].clone().detach()] * N, dim=0)
            all_new_rotation = torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0)
            new_tmp_radii = torch.zeros(all_new_xyz.shape[0])
            prune_mask = torch.ones(all_new_xyz.shape[0], dtype=torch.bool)
            gaussians.densification_postfix(
                all_new_xyz[prune_mask].to(self.device),
                all_new_features_dc[prune_mask].to(self.device),
                all_new_features_rest[prune_mask].to(self.device),
                all_new_opacities[prune_mask].to(self.device),
                all_new_scaling[prune_mask].to(self.device),
                all_new_rotation[prune_mask].to(self.device),
                new_tmp_radii[prune_mask].to(self.device)
            )


        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(
                    self.gaussians._xyz.shape[0],
                    device=self.device,
                )
                mask = torch.concat(
                    [
                        torch.ones(self.N_splats_at_init, dtype=torch.bool),
                        torch.zeros(N_splats_after_init - self.N_splats_at_init, dtype=torch.bool),
                    ],
                    axis=0,
                )
                self.GS.gaussians.prune_points(mask)

        # with torch.no_grad():
        #     gaussians = self.gaussians
        #     gaussians._scaling = gaussians.scaling_inverse_activation(
        #         gaussians.scaling_activation(gaussians._scaling) * 0.5
        #     )


    def init_with_corr(self, cfg, verbose=False, roma_model=None): 
        """
        Initializes image with matchings. Also removes SfM init points.
        Args:
            cfg: configuration part named init_wC. Check train.yaml
            verbose: whether you want to print intermediate results. Useful for debug.
            roma_model: optionally you can pass here preinit RoMA model to avoid reinit 
                it every time.  
        """
        if not cfg.use:
            return None
        N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", N_splats_at_init)
        if cfg.nns_per_ref == 1:
            init_fn = init_gaussians_with_corr_fast
        else:
            init_fn = init_gaussians_with_corr
        camera_set, selected_indices, visualization_dict = init_fn(
            self.GS.gaussians, 
            self.scene, 
            cfg, 
            self.device,                                                                                    
            verbose=verbose,
            roma_model=roma_model)

        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(self.gaussians._xyz.shape[0]).to(self.device)
                mask = torch.concat([torch.ones(N_splats_at_init, dtype=torch.bool),
                                    torch.zeros(N_splats_after_init-N_splats_at_init, dtype=torch.bool)],
                                axis=0)
                self.GS.gaussians.prune_points(mask)
        with torch.no_grad():
            gaussians =  self.gaussians
            gaussians._scaling =  gaussians.scaling_inverse_activation(gaussians.scaling_activation(gaussians._scaling)*0.5)
        return visualization_dict
    

    def prune(self, radii, min_opacity=0.005):
        self.GS.gaussians.tmp_radii = radii
        if self.gs_step < self.training_config.densify_until_iter:
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            self.GS.gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
        self.GS.gaussians.tmp_radii = None

