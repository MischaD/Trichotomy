import os 
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run image generation model with configurable parameters.")
    parser.add_argument("--filelist", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_train.txt", help="Path to the filelist.")
    parser.add_argument("--target_dir", type=str, default="diadm_train_with_dse", help="Target directory for generated images.")
    parser.add_argument("--outname", type=str, default="filelist_out.txt", help="")
    return parser.parse_args()


def main(args): 
    filelist = args.filelist
    targetdir = args.target_dir
    outname = args.outname

    # Parse the file list
    with open(filelist, "r") as fp:
        with open(outname, "w") as fpout: 
            for line in fp:
                if line.strip():  # Avoid empty lines
                    fpout.write(line)
                    parts = line.split()
                    image_path = parts[0]
                    label = parts[1:]
                    new_img_name = image_path.split(".")[0] + "_2nd.png"
                    dup_path = os.path.join(targetdir, f"{new_img_name}")
                    if os.path.exists(dup_path):
                        fpout.write(new_img_name + " " + " ".join(label) + "\n")






if __name__ == "__main__":
    args = parse_args()
    main(args)