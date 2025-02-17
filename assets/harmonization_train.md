## Training a new harmonization model from scratch

Set up the harmonization module in the **config.yaml** to define the following parameters.

```yaml
Harmonization:
  mode: "train"  # Runs harmonization in training mode
  input_datasets:
    - name: NLST  # Source dataset ->  ✅ Valid - Must match a dataset in the `Datasets` section of this config
      in_uids: "path_to_nlst_subset.csv"
      tar_uids: "/path_to_nlst_target_subset.csv"  # Target dataset (paired data for training)
  models:
    - name: SNGAN  # Model name
      model_config:
        nc_in: 1  # Number of input channels
        nb: 8  # Number of residual blocks
        nf: 64  # Number of feature maps
        nc_out: 1  # Number of output channels
  param:
    use_shuffle: true  # Shuffle training data
    n_workers: 1  # Number of workers for data loading
    batch_size: 6  # Batch size
    gpu_id: 0  # GPU device ID
    tile_xy: 64  # Tile size along X & Y
    tile_z: 32  # Tile size along Z
    train_param:
      lr_G: 1e-5  # Learning rate for generator
      weight_decay_G: 0  # Weight decay for generator
      beta1_G: 0.5  # Beta1 for Adam optimizer (Generator)
      beta2_G: 0.999  # Beta2 for Adam optimizer (Generator)
      lr_D: 1e-5  # Learning rate for discriminator
      weight_decay_D: 0  # Weight decay for discriminator
      beta1_D: 0.5  # Beta1 for Adam optimizer (Discriminator)
      beta2_D: 0.999  # Beta2 for Adam optimizer (Discriminator)
      pixel_weight: 1  # Weight for pixel loss
      pixel_criterion: "l1"  # Loss function for pixel-wise loss
      gan_weight: 5e-3  # Weight for GAN loss
      lr_scheme: "MultiStepLR"  # Learning rate scheduling
      lr_steps: [20000, 40000, 60000]  # Steps for learning rate decay
      restarts: null  # Restart epochs (if any)
      restart_weights: null  # Restart weights (if any)
      lr_gamma: 0.5  # Decay factor for learning rate
      manual_seed: 42  # Random seed for reproducibility
      D_init_iters: 1  # Number of iterations for discriminator pretraining
      print_freq: 10  # Frequency of printing logs
      save_checkpoint_freq: 5e3 # Frequency of saving model checkpoints
      niter: 50e3  # Total number of training iterations
```
🚨 **Note:** **Slice thickness harmonization is NOT performed by default.**  
To enable it, you must **explicitly define the `scale` parameter** under `param`:
```yaml
param:
  scale: 2  # Converts 2mm slice thickness to 1mm
```
- Example: If the input scans are 2mm slices, setting `scale`: 2 will train the model to generate 1mm slices.
- If scale is not specified, the model will not alter slice thickness.
- If `scale` is specified, ensure that the target and source scans have the correctly scaled slice thicknesses.

### 🖥️ Run a Session
```bash
ctnorm --config config.yaml
```