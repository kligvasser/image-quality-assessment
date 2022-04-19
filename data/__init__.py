from .datasets import  DatasetEval
from torch.utils.data import DataLoader

def get_loader(args):
    # dataset
    dataset = DatasetEval(root_input=args.input_dir, root_target=args.target_dir, max_size=args.max_size, num_crops=args.num_crops, crop_size=args.crop_size)

    # loader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    return loader