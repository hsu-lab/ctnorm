# CT-Norm: A Toolkit To Characterize and Harmonize Variability in CT

<p align="center">
  <img src="./assets/workflow.png" alt="Example Image" width="70%"/>
</p>

The repository contains usage instructions for CTNorm: a toolkit that offer modules for data understanding, image harmonization, and model performance evaluation across different datasets.

## 🚀 Getting Started

These instructions will guide you through setting up and using CTNorm.

### Step 1: 📁 **Data Requirements**
Currently, **CTNorm** supports only **[DICOM](https://pydicom.github.io/pydicom/stable/tutorials/installation.html) (`.dcm`) format**.
- A **CSV file** is required that must contain a column named `uids`, where each row corresponds to the **path of a DICOM folder**.

| uids |
|-------------------------------|
| `/data/path/to/dicom/folder1` |
| `/data/path/to/dicom/folder2` |
| `/data/path/to/dicom/folder3` |

### Step 2: 🐳 Setup Environment
To ensure **easy setup and reproducibility**, we recommend running CTNorm using Docker.

1️⃣ Clone the CTNorm repository:
```bash
git clone https://github.com/hsu-lab/ctnorm.git
```
2️⃣ Pull the Docker image from [Docker Hub](https://hub.docker.com/) 🐳:
```bash
docker pull litou/mii-pytorch:20.11
```
3️⃣ Run Docker in interactive mode
```bash
docker run --name=<container_name> --shm-size=<memory_size> --gpus device=<gpu_id> -it --rm -p <port_number>:<port_number> -v /etc/localtime:/etc/localtime:ro -v "$(pwd)":/workspace -v <path_to_input_data>:/data litou/mii-pytorch:20.11
```
**Parameters:**
* *<container_name>*: Specify the name of the container.
* *<memory_size>*: Specify the shared memory size (e.g., `2g`, `4g`, `6g`).
* *<gpu_id>*: Specify the gpu device id. If no GPU is available, this parameter can be omitted.
* *<port_number>*: Specify a port number in Docker. This is required when running the Flask component of the toolkit.
* *<path_to_input_data>*: Path to the input data directory, which will be mounted as `/data` in the container.<br>
🚨 **Note:** **The paths specified in the CSV file must be accessible within the mounted Docker container**.

4️⃣ Install the CTNorm package locally:
```bash
cd /workspace/ctnorm # Move to the project directory
pip install -e .
```

### Step 3: 🛠️ Setup Configuratiion File

The CTNorm pipeline requires a YAML configuration file to define parameters needed to run each module. Below is a breakdown of each section in **config.yaml**.
```yaml
Global:
  session_base_path: "./SESSIONS" # Base path where all session folders will be created; each run creates a new session
```
```yaml
Datasets:
  NLST:
    in_uids: "/path_to_nlst_data_cases.csv" # Path to a CSV file containing UIDs (references to DICOM folders)
    in_dtype: ".dcm" # Specifies input data type. Must be in DICOM format at the moment
    description: "National Lung Screening Trial dataset" # Descriptive name for the dataset

  SPIE:
    in_uids: "/path_to_spie_data_cases.csv"
    in_dtype: ".dcm"
    description: "SPIE LungX dataset"

```
- Each dataset must have `in_uids`, `in_dtype`, and `description` fields.
- The dataset name (e.g., `NLST`, `SPIE`) is a user-defined key and should be unique.
```yaml
Modules:
  Characterization: true
  Harmonization: true
  Robustness: true
```
- Set `true` or `false` to enable/disable a module.
```yaml
Characterization:
  input_datasets:
    - name: NLST  # ✅ Valid - Must match a dataset in the `Datasets` section
    - name: SPIE  # ✅ Valid - Must match a dataset in the `Datasets` section
    - name: XYZ   # ❌ Invalid - Doesn't match any dataset in the `Datasets` section
  metrics:
    voxel:
      - all
    metadata:
      - all
  params:
    clip_range : [-1024, 3071]
    bins: 64
    kde_points: 1e3
    kde_sample: 1e5
```
- The dataset name specified in `input_datasets` must match one of key defined in the `Datasets` section.
- If multiple datasets are provided, each must be listed separately under input_datasets.
- The `metrics` field defines the types of properties that will be characterized in the dataset. It includes **voxel** statistics and **metadata** properties.
  
| **Available Options** | **Category** | **Description** |
|------------------------|-------------|------------------|
| **histogram**           | voxel       | Generates an intensity histogram to analyze intensity distribution. |
| **kde**                 | voxel       | Computes Kernel Density Estimation (KDE) for voxel intensities. |
| **snr**                 | voxel       | Measures the ratio of signal intensity to noise, indicating image quality. |
| **skewness**            | voxel       | Calculates the asymmetry of the voxel intensity distribution. |
| **kurtosis**            | voxel       | Measures the "tailedness" of the intensity distribution. |
| **all**                 | voxel       | Includes all the above voxel metrics. |
| **slice_thickness**     | metadata    | Analyzes the distribution of slice thickness across the dataset. |
| **convolution_kernel**  | metadata    | Summarizes the convolution kernels used during image reconstruction. |
| **manufacturer**        | metadata    | Lists the equipment manufacturers to assess scanner variability. |
| **all**                 | metadata    | Includes all the above metadata metrics. |

- **🚨 Note:** We plan to expand the available `metrics` to include **`radiomic`** feature extraction in an upcoming update.

- For **voxel-level** analysis, the following **`params`** can be set:
  - `clip_range`: Sets the intensity value range for voxel-level analysis.  
    - `[-1024, 3071]` → Analyzes voxel intensities only within this specified range.
  - `bins`: Defines the number of bins for histogram computation.  
    - `64` → The histogram will use 64 bins for distribution analysis.
  - `kde_points`: Specifies the number of points for Kernel Density Estimation (KDE) curve generation.  
    - `1e3` → Uses 1,000 points to compute the KDE curve.
  - `kde_sample`: Sets the sample size for KDE estimation.  
    - `1e5` → Uses 100,000 voxel samples for KDE calculations.

> The **Harmonization module** can run in two different modes.

1️⃣ Running Harmonization in Test Mode – Uses a pretrained model to harmonize input datasets.<br>
```yaml
Harmonization:
  mode: "test"  # Runs harmonization in inference mode
  input_datasets:
    - name: NLST  # ✅ Valid - Must match a dataset in the `Datasets` section
      in_uids: "/path_to_nlst_subset_data.csv"  # 🔹 (Optional) Overrides the `in_uids` from the `Datasets` section if specified
  models:
    - name: SNGAN  # The model being used
      pretrained_G: "./pretrained_weights/SNGAN/latestG-1-1.pth"  # Path to the pretrained generator model
  param:
    tile_xy: 512  # Tile size along X & Y (Keep 512 during inference)
    tile_z: 32  # Tile size along Z
    z_overlap: 4  # Overlap between slices
    gpu_id: 0  # GPU device ID
    out_dtype: ".dcm"  # Output data format (must be either .nii.gz or .dcm)
    save_lr: true
  metrics:
    - snr
    - sobel
    - radiomic 
```
- The **Harmonization module** supports multiple models for CT harmonization. Below are the available model options:<br>

| **Model Name** | **Description** |
|--------------|----------------|
| **[SNGAN](https://ieeexplore.ieee.org/abstract/document/9098724)** | Spectral Normalization GAN, used for image-to-image translation. |
| **[WGAN](https://arxiv.org/abs/1704.00028)** | Wasserstein GAN, improves stability of training for generative models. |
| **[Pix2Pix](https://phillipi.github.io/pix2pix/)** | Conditional GAN, useful for paired image transformation. |
| **[SRResNet](https://ieeexplore.ieee.org/document/8099502)** | Super-Resolution ResNet, designed for image enhancement. |
| **[RRDB](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf)** | Residual-in-Residual Dense Block, used in SRGAN-style super-resolution tasks. |
| **[BM3D](https://www.ipol.im/pub/art/2012/l-bm3d/)** | A non-deep learning method for denoising images. |

- We have provided the pretrained weights [here](https://drive.google.com/drive/folders/1QdSkDIIEG2IivyHLMTH_PEOuaOrnXMUv?usp=drive_link). Update the `pretrained_G` parameter depending on the model accordingly.

For **BM3D**, only one **optional parameter** can be specified; other parameters are not needed:
```yaml
models:
  - name: BM3D
param:
  noise_type: "psd"  # Optional, choose between "psd" or "std"
```
- To evaluate the effectivness of harmonization, the following `metrics` can be computed:

| **Metric**  | **Description**                                                                                   |
|-------------|---------------------------------------------------------------------------------------------------|
| **`snr`** *(Signal-to-Noise Ratio)*  | Measures the clarity of the image signal relative to background noise. A higher SNR indicates better image quality with less noise. |
| **`sobel`** *(Sobel Edge Detection)* | Applies a Sobel filter to evaluate edge sharpness, ensuring that anatomical boundaries are preserved. |
| **`radiomic`** *(Radiomic Feature Analysis)* | Extracts radiomic features to assess intensity, and texture characteristics for image-derived features. |

- **🚨 Note:** We plan to expand the available `metrics` to include **`tsne`** analysis in an upcoming update.

2️⃣ Want to train your own harmonization model from scratch ? Follow the steps outlined [here](assets/harmonization_train.md).

> We currently have **Sybil** model integrated as part of the robustness analysis module. It is a deep learning model developed to analyze chest CT scans and predict an individual's risk of developing lung cancer over multiple time horizons, including 1-year, 2-year, and 6-year periods. Read more about it [here.](https://github.com/reginabarzilaygroup/Sybil)
```yaml
Robustness:
  input_datasets:
    - name: NLST # ✅ Valid - Must match a dataset in the `Datasets` section
      in_uids: "/path_to_nlst_subset_data.csv" # 🔹 (Optional) Overrides the `in_uids` from the `Datasets` section if specified
      variability:
        - manufacturer
        - slice_thickness
        - convolution_kernel

  param:
    model_type: "sybil_ensemble"  # Options: sybil_1, sybil_2, sybil_3, sybil_4, sybil_5, sybil_ensemble
    evaluate: true  # Requires 'label' and 'time_to_event' columns in `in_uids` CSV. If set to false, it will save the predicted scores.
```
- `variability` defines the imaging variation to be assessed, as identified in **Characterization** module.
- If not specified, it will run **Sybil** on all cases specified in `in_uids`.<br>
🚨 **Note:** **Defined variability must exist in the generated `metadata_characterization.csv` file**.
- If the **Robustness** module is run at a different time (not together with the **Characterization** module), a `load_from` parameter must be specified to load previously generated characterization data csv as shown below:
```yaml
Robustness:
  input_datasets:
    - name: NLST
      in_uids: "/path_to_nlst_subset_data.csv"
      variability:
        - manufacturer
        - slice_thickness
        - convolution_kernel
      load_from: "20250203-033504-40"  # # 🔹 (Optional) Specify the session number from which to load `metadata_characterization.csv` 
      
  param:
    model_type: "sybil_ensemble"
    evaluate: true
```

### Step 4: 🖥️ Run a Session
```bash
ctnorm --config config.yaml
```

## 🌐 Launching a Web Server

CTNorm also offers a user-friendly interface to visualize the outputs of each session.<br>
🚨 **Note:** **The visualization component is continuously evolving, with new metric outputs and their visualizations being supported over time!**

- To launch the app:
```bash
ctnorm-webapp --port <port_number> --session-out <path_to_session_folder>
```
**Parameters:**
* *<port_number>*: Specify the **port exposed in the Docker container**. This must match the port defined when running Docker (**Step 2: 3️⃣ Running Docker in interactive mode**).
* *<path_to_session_folder>*: Provide the **path** to the folder where all session outputs are stored. This should be the same session directory used when running the CTNorm Toolkit.

Accessing the Web App on a Local Machine at:
```bash
http://localhost:<port_number>
```
Accessing the Web App on a Remote Machine at:
```bash
http://<remote_server_ip>:<port_number>
```

> The Flask application displays the status of all sessions. Sessions that are complete can be **started**.

🚨 **Popup Blocker Warning for Harmonization Viewer:**  
The image viewer from the **Harmonization** tab opens in a **popup window**. Some browsers may block popups by default. If you see a **popup blocked notification**, allow it to ensure the viewer opens correctly.

🚧 **Work in Progress** 🚧  
More features coming soon!
