from ctnorm import run_inference as ct_infer
import argparse
import sys


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model to run...Must be one of [`BM3D`, `HM`, `SRResNet`, `RRDB`, `SNGAN`, `WGAN`]")
    parser.add_argument("--gpu_id", help="GPU to run inference...Default is `cpu`", default='cpu')
    parser.add_argument("--in_path", help="Path to input cases")
    parser.add_argument("--in_type", help="Input file type...Must be one of [`nii`, 'nii.gz', `dcm`]")
    parser.add_argument("--out_path", help="Path to save outout cases")
    parser.add_argument("--out_type", help="File type of output cases", default='nii.gz')
    parser.add_argument("--gt_path", help="Path to full dose if exists")
    args = parser.parse_args()
    if not args.model or not args.in_path or not args.in_type or not args.out_path:
        print("All required parameters are not set!\n\
        --model must be specified --> [`BM3D`, `HM`, `SRResNet`, `RRDB`, `SNGAN`, `WGAN`]\n\
        --in_path must be specified\n\
        --in_type must be specified --> [`nii`, 'nii.gz', `dcm`]\n\
        --out_path must be specified...Exiting")
        sys.exit()
    else:
        ct_infer.execute(args.model, args.in_path, args.in_type, args.out_path, args.out_type, args.gpu_id, args.gt_path)
