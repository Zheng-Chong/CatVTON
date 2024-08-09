import os
import torch
from cleanfid import fid as FID
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm

from utils import scan_files_in_dir
from prettytable import PrettyTable

class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_folder, height=1024):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.height = height
        self.data = self.prepare_data()
        self.to_tensor = transforms.ToTensor()
    
    def extract_id_from_filename(self, filename):
        # find first number in filename
        start_i = None
        for i, c in enumerate(filename):
            if c.isdigit():
                start_i = i
                break
        if start_i is None:
            assert False, f"Cannot find number in filename {filename}"
        return filename[start_i:start_i+8]
    
    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={'.jpg', '.png'})
        gt_dict = {self.extract_id_from_filename(file.name): file for file in gt_files}
        pred_files = scan_files_in_dir(self.pred_folder, postfix={'.jpg', '.png'})
        
        tuples = []
        for pred_file in pred_files:
            pred_id = self.extract_id_from_filename(pred_file.name)
            if pred_id not in gt_dict:
                print(f"Cannot find gt file for {pred_file}")
            else:
                tuples.append((gt_dict[pred_id].path, pred_file.path))
        return tuples
        
    def resize(self, img):
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]
        gt, pred = self.resize(Image.open(gt_path)), self.resize(Image.open(pred_path))
        if gt.height != self.height:
            gt = self.resize(gt)
        if pred.height != self.height:
            pred = self.resize(pred)
        gt = self.to_tensor(gt)
        pred = self.to_tensor(pred)
        return gt, pred


def copy_resize_gt(gt_folder, height):
    new_folder = f"{gt_folder}_{height}"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue
        img = Image.open(os.path.join(gt_folder, file))
        w, h = img.size
        new_w = int(w * height / h)
        img = img.resize((new_w, height), Image.LANCZOS)
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        ssim_score += ssim(pred, gt) * batch_size
    return ssim_score / len(dataloader.dataset)


@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to("cuda")
    score = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        # LPIPS needs the images to be in the [-1, 1] range.
        gt = (gt * 2) - 1
        pred = (pred * 2) - 1
        score += lpips_score(gt, pred) * batch_size
    return score / len(dataloader.dataset)


def eval(args):
    # Check gt_folder has images with target height, resize if not
    pred_sample = os.listdir(args.pred_folder)[0]
    gt_sample = os.listdir(args.gt_folder)[0]
    img = Image.open(os.path.join(args.pred_folder, pred_sample))
    gt_img = Image.open(os.path.join(args.gt_folder, gt_sample))
    if img.height != gt_img.height:
        title = "--"*30 + "Resizing GT Images to height {img.height}" + "--"*30
        print(title)
        args.gt_folder = copy_resize_gt(args.gt_folder, img.height)
        print("-"*len(title))
    
    # Form dataset
    dataset = EvalDataset(args.gt_folder, args.pred_folder, img.height)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False
    )
    
    # Calculate Metrics
    header = []
    row = []
    header = ["FID", "KID"]
    fid_ = FID.compute_fid(args.gt_folder, args.pred_folder)
    kid_ = FID.compute_kid(args.gt_folder, args.pred_folder) * 1000
    row = [fid_, kid_]
    if args.paired:
        header += ["SSIM", "LPIPS"]
        ssim_ = ssim(dataloader).item()
        lpips_ = lpips(dataloader).item()
        row += [ssim_, lpips_]
    
    # Print Results
    print("GT Folder  : ", args.gt_folder)
    print("Pred Folder: ", args.pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)
    print(table)
    
         
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--pred_folder", type=str, required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    eval(args)