import argparse
import torch
import logging
import signal
import sys
import torch.backends.cudnn as cudnn
from evaluator import Evaluator
from datetime import datetime
from os import path
from utils import misc
from random import randint

def get_arguments():
    example_text = '''
    example:
      python3 main.py --device cpu --input-dir /home/klig/Dropbox/technion/thesis/restorations/BSD100/EDSR --target-dir /home/klig/Dropbox/technion/thesis/internal-external/results/sr/x4/bsd/img --metric-list dsd ssim niqe
      '''

    parser = argparse.ArgumentParser(description='Image Quality Assessment', epilog=example_text)
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--input-dir', required=True, help='input root dataset folder')
    parser.add_argument('--target-dir', required=True, help='target root dataset folder')
    parser.add_argument('--max-size', default=None, type=int, help='limited number of images (default: None)')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops (default: 1)')
    parser.add_argument('--crop-size', default=None, type=int, help='croping size (default: None)')
    parser.add_argument('--metric-list', nargs='+', default=[], help='metrics to measure: psnr,ssim,lpips,niqe,perceptual,style,wasserstein,dsd,noref-dsd,fid,single-fid (default: None)')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save == '':
        args.save = time_stamp
    args.save_path = path.join(args.results_dir, args.save)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    args.metric_list = [metric.lower() for metric in args.metric_list]
    return args

def main():
    args = get_arguments()

    torch.manual_seed(args.seed)

    # cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_id)
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # set logs
    misc.mkdir(args.save_path)
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # print logs
    logging.info(args)

    # evaluation
    runner = Evaluator(args)
    runner.eval()

if __name__ == '__main__':
    # enables a ctrl-c without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()