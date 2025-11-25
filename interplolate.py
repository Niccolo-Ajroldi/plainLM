import torch
from spectra import build_dataloader, build_model 
from pathlib import Path
from absl import app, flags
import pdb
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
from collections import defaultdict
from torch.nn import CrossEntropyLoss
import json
from functools import partial

from engine.engine import _move_to_device
import utils


# Job specs.
flags.DEFINE_integer('row_idx', -1, 'Row index.')
flags.DEFINE_string('save_path', None, 'Path to save results.')
flags.DEFINE_integer('grid_len', 51, 'Row index.')

# Dataloader specs.
flags.DEFINE_integer('samples', 100, 'Number of samples in dataloader.')
flags.DEFINE_integer('bsz', 48, 'Batch size.') # max for Pythia-160M

# Parse flags.
FLAGS = flags.FLAGS

AVG_CKPT = \
    "/fast/atatjer/scalinglawsquantization/checkpoints/p100BT_lawa_on_resume/job_idx_3/ckpt_step_190000.pth"
DECAY_CKPTS = [
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_174000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_176000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_178000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_180000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_182000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_184000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_186000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_188000.pth",
    "/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_190000.pth",
]


def strip_off_state_dict(ckpt):
    # Manipulate the saved state_dict, to allow loading LAWA.
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def _eval(model, dataloader, seq_len, device):
    model.eval()
    criterion = CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        inputs, targets, _ = _move_to_device(batch, seq_len, device)
        output = model(inputs)
        logits = getattr(output, 'logits', output)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        if torch.isnan(loss) or loss is None:
            raise ValueError("Validation loss is nan")

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss


    
def main(_):
    
    device = f'cuda:0'
    torch.cuda.device(device)

    # Mode: parallel (row_idx>=0) or sequential (row_idx==-1).
    if FLAGS.row_idx == -1:
        row_indices = range(len(DECAY_CKPTS))
    else:
        row_indices = [FLAGS.row_idx]

    # Load checkpoints, extract state_dict.
    p_avg = Path(AVG_CKPT)
    print(f"p_avg = {p_avg}")
    sd_start = torch.load(p_avg)
    sd_start = strip_off_state_dict(sd_start)

    # Load config.
    cfg_path = p_avg.parent / "config.yaml"
    print(f"Loading config from {cfg_path}")
    cfg, _ = utils.load_config(cfg_path)    

    # Dataloader.
    validloader = build_dataloader(validset_path=cfg.validset_path, samples=FLAGS.samples, bsz=FLAGS.bsz)
    print(f"Number of validation batches: {len(validloader)}")
    print(f"Batch size: {validloader.batch_size}")
    
    # Model.
    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
    model = AutoModelForCausalLM.from_config(model_cfg)
    model.init_weights()
    model.to(device)

    # For ease of life.
    eval = partial(_eval, dataloader=validloader, seq_len=cfg.seq_len, device=device)

    for row_idx in row_indices:

        p_decay = Path(DECAY_CKPTS[row_idx])
        print(f"\tp_decay = {p_decay}")
        sd_end = torch.load(p_decay)
        sd_end = strip_off_state_dict(sd_end)
        
        metrics = defaultdict(list)
        for alpha in np.linspace(0, 1, FLAGS.grid_len):

            print(f"\talpha={alpha}")

            # Interpolate.
            new_sd = {k: (1 - alpha) * sd_start[k] + alpha * sd_end[k] for k in sd_start}
            model.load_state_dict(new_sd)
        
            # Eval
            val_loss = eval(model)
            
            # Book-keeping
            metrics['alpha'].append(alpha)
            metrics['val_loss'].append(val_loss)

        # Eval start and end as a check
        model.load_state_dict(sd_start)
        val_loss_start = eval(model)    
        model.load_state_dict(sd_end)
        val_loss_end = eval(model)
        metrics['val_loss_end'] = val_loss_end
        metrics['val_loss_start'] = val_loss_start

        # Save results
        if FLAGS.save_path:
            metrics_path = Path(FLAGS.save_path) / f"row_{row_idx}" / f'scalars.json'
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\tMetrics saved to {metrics_path}")

    print('Finished!')

if __name__ == "__main__":
  app.run(main)
