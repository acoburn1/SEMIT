"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import os
import sys
import argparse
import shutil

from tensorboardX import SummaryWriter

from utils import (get_config, get_train_loaders, make_result_folders, get_evaluation_loaders)
from utils import write_loss, write_html, write_1images, Timer

from trainer import Trainer
import pdb

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True


### new ###
def run_evaluation_epoch(trainer, content_loader, class_loader,
                         epoch, output_directory,
                         max_batches, multigpus):
    """
    Run seen->unseen evaluation for one epoch and save images.

    content_loader: from eval_config['data_folder_train'] (seen classes, unseen images)
    class_loader:   from eval_config['data_folder_test']  (unseen-target classes)
    """
    if content_loader is None or class_loader is None:
        return

    eval_root = os.path.join(output_directory, "eval")
    os.makedirs(eval_root, exist_ok=True)
    image_directory = os.path.join(eval_root, f"epoch_{epoch:04d}")
    os.makedirs(image_directory, exist_ok=True)

    print(f"[Eval] Running evaluation for epoch {epoch}...")
    with torch.no_grad():
        for t, (co_data, cl_data) in enumerate(zip(content_loader, class_loader)):
            if t >= max_batches:
                break
            image_outputs = trainer.test(co_data, cl_data, multigpus)
            write_1images(image_outputs, image_directory,
                          f"eval_{epoch:04d}_{t:02d}")

    print(f"[Eval] Saved evaluation images for epoch {epoch} in {image_directory}")

def save_epoch_sample_trios(trainer,
                            content_loader,
                            class_loader,
                            epoch,
                            output_directory,
                            max_trios,
                            multigpus):
    """
    Save a few sample (source, target, generated) triplets from the test loaders
    for a given epoch.
    """
    epoch_dir = os.path.join(output_directory, 'epoch_samples',
                             f'epoch_{epoch:04d}')
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        for t, (co_data, cl_data) in enumerate(zip(content_loader, class_loader)):
            if t >= max_trios:
                break
            image_outputs = trainer.test(co_data, cl_data, multigpus)

            # SEMIT_model.test returns:
            #  0: xa (source)
            #  2: xt_current_set[5] (generated)
            #  3: xb (target)
            xa = image_outputs[0]
            xb = image_outputs[3]
            xt = image_outputs[2]

            triplet = [xa, xb, xt]
            save_name = f'epoch{epoch:04d}_sample{t:02d}'
            write_1images(triplet, epoch_dir, save_name)

    print(f"[Epoch {epoch}] Saved sample trios in {epoch_dir}")
### end new ###

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='.',
                    help="outputs path")
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument('--batch_size',
                    type=int,
                    default=0)
parser.add_argument('--test_batch_size',
                    type=int,
                    default=4)
parser.add_argument("--resume",
                    action="store_true")
parser.add_argument('--eval_config', type=str, default=None,
                    help='Optional YAML config for seen→unseen evaluation '
                         '(used with get_evaluation_loaders).')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
eval_config = get_config(opts.eval_config) if opts.eval_config is not None else None
# Override the batch size if specified.
if opts.batch_size != 0:
    config['batch_size'] = opts.batch_size

checkpoint_epoch_freq = config.get('checkpoint_epoch_freq', 20)
epoch_metrics_file = None 

trainer = Trainer(config)
trainer.cuda()
if opts.multigpus:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(
        trainer.model, device_ids=range(ngpus))
else:
    config['gpus'] = 1

loaders = get_train_loaders(config)
train_content_loader = loaders[0]
train_class_loader = loaders[1]
test_content_loader = loaders[2]
test_class_loader = loaders[3]

if eval_config is not None:
    eval_content_loader, eval_class_loader = get_evaluation_loaders(eval_config)
else:
    eval_content_loader, eval_class_loader = None, None

num_batches_per_epoch = min(len(train_content_loader), len(train_class_loader))

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = SummaryWriter(
    os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))
epoch_metrics_file = os.path.join(output_directory, 'epoch_metrics.csv')

iterations = trainer.resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus) if opts.resume else 0

epoch = iterations // num_batches_per_epoch
print(f"Starting training from iteration {iterations}, epoch {epoch}")

epoch_sums = {}
epoch_count = 0

while True:
    for it, (co_data, cl_data) in enumerate(
            zip(train_content_loader, train_class_loader)):
        with Timer("Elapsed time in update: %f"):
            octave_alpha = torch.rand(1).cuda()
            #In Disc I use constant octave: 0.25, which is made inside
            d_acc = trainer.dis_update(co_data, cl_data, octave_alpha, config)
            g_acc = trainer.gen_update(co_data, cl_data, octave_alpha, config,
                                       opts.multigpus)
            torch.cuda.synchronize()
            print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # accumulate metrics for epoch averages
        metric_members = [attr for attr in dir(trainer)
                          if ((not callable(getattr(trainer, attr))
                               and not attr.startswith("__"))
                              and ('loss' in attr
                                   or 'grad' in attr
                                   or 'nwd' in attr
                                   or 'accuracy' in attr))]
        for m in metric_members:
            v = getattr(trainer, m)
            if torch.is_tensor(v):
                v = v.item()
            v = float(v)
            epoch_sums[m] = epoch_sums.get(m, 0.0) + v
        epoch_count += 1

        if ((iterations + 1) % config['image_save_iter'] == 0 or (
                iterations + 1) % config['image_display_iter'] == 0):
       # if True:
            if (iterations + 1) % config['image_save_iter'] == 0:
                key_str = '%08d' % (iterations + 1)
                write_html(output_directory + "/index.html", iterations + 1,
                           config['image_save_iter'], 'images')
            else:
                key_str = 'current'
            with torch.no_grad():
                for t, (val_co_data, val_cl_data) in enumerate(
                        zip(train_content_loader, train_class_loader)):
                    if t >= opts.test_batch_size:
                        break
                    val_image_outputs = trainer.test(val_co_data, val_cl_data,
                                                     opts.multigpus)
                    write_1images(val_image_outputs, image_directory,
                                  'train_%s_%02d' % (key_str, t))
                for t, (test_co_data, test_cl_data) in enumerate(
                            zip(test_content_loader, test_class_loader)):
                    if t >= opts.test_batch_size:
                        break
                    test_image_outputs = trainer.test(test_co_data,
                                                      test_cl_data,
                                                      opts.multigpus)
                    write_1images(test_image_outputs, image_directory,
                                  'test_%s_%02d' % (key_str, t))

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1

        # ==== End-of-epoch hook ====
        if iterations % num_batches_per_epoch == 0:
            epoch += 1
            print(f"==== Finished epoch {epoch} (iterations={iterations}) ====")

            # epoch-averaged metrics
            if epoch_count > 0 and epoch_sums:
                metric_names = sorted(epoch_sums.keys())
                avg_metrics = {k: epoch_sums[k] / epoch_count
                               for k in metric_names}

                # log to TensorBoard as epoch/<metric>
                for name, value in avg_metrics.items():
                    train_writer.add_scalar(f'epoch/{name}', value, epoch)

                # append to CSV
                header_needed = (not os.path.exists(epoch_metrics_file)
                                 or os.path.getsize(epoch_metrics_file) == 0)
                with open(epoch_metrics_file, 'a') as f:
                    if header_needed:
                        f.write('epoch,iterations,' +
                                ','.join(metric_names) + '\n')
                    row = [str(epoch), str(iterations)] + \
                          [f"{avg_metrics[m]:.6f}" for m in metric_names]
                    f.write(','.join(row) + '\n')
                print(f"[Epoch {epoch}] Wrote epoch metrics to {epoch_metrics_file}")

            # save two sample trios (source, target, generated) from test loaders
            save_epoch_sample_trios(trainer,
                                    test_content_loader,
                                    test_class_loader,
                                    epoch,
                                    output_directory,
                                    max_trios=2,
                                    multigpus=opts.multigpus)

            # Run seen->unseen evaluation for this epoch, if eval_config given
            if eval_content_loader is not None and eval_class_loader is not None:
                run_evaluation_epoch(trainer,
                                     eval_content_loader,
                                     eval_class_loader,
                                     epoch,
                                     output_directory,
                                     opts.test_batch_size,
                                     opts.multigpus)

            # checkpoint every N epochs (in addition to snapshot_save_iter if desired)
            if checkpoint_epoch_freq > 0 and epoch % checkpoint_epoch_freq == 0:
                trainer.save(checkpoint_directory, iterations, opts.multigpus)
                print(f"Saved model at end of epoch {epoch} (iteration {iterations})")

            # reset epoch accumulators
            epoch_sums = {}
            epoch_count = 0

        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)

