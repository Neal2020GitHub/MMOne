import os
import argparse


parser = argparse.ArgumentParser(description='Convert RGBT-Scenes to 3DGS dataset structure')
parser.add_argument("--path", required=True, type=str)
args = parser.parse_args()

for name in os.listdir(args.path):
    scene_path = os.path.join(args.path, name)
    if not os.path.isdir(scene_path):  # skip readme.txt
        continue

    rgb_path = os.path.join(scene_path, "rgb")
    thermal_path = os.path.join(scene_path, "thermal")

    print("Processing scene:", scene_path)

    os.system("mv " + os.path.join(rgb_path, "train", "*") + " " + rgb_path)
    os.system("mv " + os.path.join(rgb_path, "test", "*") + " " + rgb_path)
    os.system("mv " + os.path.join(thermal_path, "train", "*") + " " + thermal_path)
    os.system("mv " + os.path.join(thermal_path, "test", "*") + " " + thermal_path)

    os.system("rm -r " + os.path.join(rgb_path, "train"))
    os.system("rm -r " + os.path.join(rgb_path, "test"))
    os.system("rm -r " + os.path.join(thermal_path, "train"))
    os.system("rm -r " + os.path.join(thermal_path, "test"))
    
    os.system("mv " + rgb_path + " " + os.path.join(scene_path, "images"))
    
    os.system("cp -r " + os.path.join(scene_path, "colmap", "sparse") + " " + os.path.join(scene_path))
