import logging
import pandas as pd
import lpips
from os import path
from models.loss import *
from data import get_loader
from utils.misc import psnr_yc, ssim
from utils.fid import calculate_fid_given_tensors, calculate_fid_given_paths
from utils.niqe import calculate_niqe_given_tensor


class Evaluator:
    def __init__(self, args):
        # parameters
        self.args = args
        self._init()

    def _init(self):
        # criterions
        self.lpips = (
            lpips.LPIPS(net='alex').to(self.args.device)
            if 'lpips' in self.args.metric_list
            else None
        )
        self.perceptual = (
            PerceptualLoss().to(self.args.device) if 'perceptual' in self.args.metric_list else None
        )
        self.style = StyleLoss().to(self.args.device) if 'style' in self.args.metric_list else None
        self.wasserstein = (
            WassersteinLoss().to(self.args.device)
            if 'wasserstein' in self.args.metric_list
            else None
        )
        self.dsd = DSDLoss().to(self.args.device) if 'dsd' in self.args.metric_list else None
        self.noref_dsd = (
            NoRefDSDLoss(batch_size=self.args.num_crops).to(self.args.device)
            if 'noref-dsd' in self.args.metric_list
            else None
        )

    def _eval_iteration(self, data, df):
        # set inputs
        inputs = data['input'].to(self.args.device)
        targets = data['target'].to(self.args.device)
        path_input = data['path_input'][0]
        path_target = data['path_target'][0]

        # squeeze
        if self.args.num_crops > 1:
            inputs = inputs.squeeze(dim=0)
            targets = targets.squeeze(dim=0)

        # evaluations
        with torch.no_grad():
            lpips_sc = (
                float(self.lpips(inputs, targets, normalize=True).mean().item())
                if 'lpips' in self.args.metric_list
                else 0
            )
            perceptual_sc = (
                float(self.perceptual(inputs, targets).item())
                if 'perceptual' in self.args.metric_list
                else 0
            )
            style_sc = (
                float(self.style(inputs, targets).item()) if 'style' in self.args.metric_list else 0
            )
            wasserstein_sc = (
                float(self.wasserstein(inputs, targets).item())
                if 'wasserstein' in self.args.metric_list
                else 0
            )
            dsd_sc = (
                float(self.dsd(inputs, targets).item()) if 'dsd' in self.args.metric_list else 0
            )
            noref_dsd_sc = (
                float(self.noref_dsd(inputs).item()) if 'noref-dsd' in self.args.metric_list else 0
            )
            psnr_sc = (
                float(psnr_yc(inputs, targets).mean().item())
                if 'psnr' in self.args.metric_list
                else 0
            )
            ssim_sc = (
                float(ssim(inputs, targets).mean().item()) if 'ssim' in self.args.metric_list else 0
            )
            niqe_sc = calculate_niqe_given_tensor(inputs) if 'niqe' in self.args.metric_list else 0

            if self.args.num_crops > 1 and 'single-fid' in self.args.metric_list:
                sfid_sc = calculate_fid_given_tensors(inputs, targets)
            else:
                sfid_sc = 0

        # logging
        line2print = '{}, {} : lpips {:.4f}, perceptual {:.4f}, style {:.4f}, wasserstein {:.4f}, dsd {:.5f}, noref-dsd {:.5f}, psnr {:.4f}, ssim {:.4f}, sfid {:.4f}, niqe {:.4f}'.format(
            path_input,
            path_target,
            lpips_sc,
            perceptual_sc,
            style_sc,
            wasserstein_sc,
            dsd_sc,
            noref_dsd_sc,
            psnr_sc,
            ssim_sc,
            sfid_sc,
            niqe_sc,
        )
        logging.info(line2print)

        # data-frame
        line_dict = {
            'input': path_input,
            'target': path_target,
            'lpips': lpips_sc,
            'perceptual': perceptual_sc,
            'style': style_sc,
            'wasserstein': wasserstein_sc,
            'dsd': dsd_sc,
            'noref-dsd': noref_dsd_sc,
            'psnr': psnr_sc,
            'ssim': ssim_sc,
            'sfid': sfid_sc,
            'niqe': niqe_sc,
        }
        df = pd.concat([df, pd.DataFrame([line_dict])], ignore_index=True)

        return df

    def _compute_fid(self, df):
        # run fid
        fid_sc = calculate_fid_given_paths(
            paths=[self.args.input_dir, self.args.target_dir],
            batch_size=256,
            cuda=self.args.device != 'cpu',
            dims=2048,
        )

        # logging
        line2print = '{}, {} : fid {:.4f}'.format(self.args.input_dir, self.args.target_dir, fid_sc)
        logging.info(line2print)

        # data-frame
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{'input': self.args.input_dir, 'target': self.args.target_dir, 'fid': fid_sc}]
                ),
            ],
            ignore_index=True,
        )

        return df

    def eval(self):
        # get loader
        loader = get_loader(self.args)

        # data frames
        df = pd.DataFrame(
            columns=[
                'input',
                'target',
                'lpips',
                'perceptual',
                'style',
                'wasserstein',
                'dsd',
                'noref-dsd',
                'psnr',
                'ssim',
                'sfid',
                'niqe',
                'fid',
            ]
        )

        # evalulation
        if not self.args.metric_list == [] and self.args.metric_list != ['fid']:
            for _, data in enumerate(loader):
                df = self._eval_iteration(data, df)

        # compute fid
        if 'fid' in self.args.metric_list:
            df = self._compute_fid(df)

        # logging
        logging.info('Evaluation: \n{}'.format(df.describe()))
        df.to_csv(path.join(self.args.save_path, 'stats.csv'))
