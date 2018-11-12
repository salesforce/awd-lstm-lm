import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def init_device(args):
    """Sets the `args.device` attribute based on `args.no_cuda` and host
    availability. It also sets the CUDA seed if needed.
    """
    if torch.cuda.is_available():
        if args.no_cuda:
            print('WARNING: You have a CUDA device,'
                  'so you should probably not run with --no_cuda')
            setattr(args, 'device', torch.device('cpu'))
        else:
            if hasattr(args, 'seed'):
                torch.cuda.manual_seed(args.seed)
            setattr(args, 'device', torch.device('cuda'))
    else:
        if not args.no_cuda:
            print('WARNING: No CUDA device found, using CPU. '
                  'It would be best to explicitly run with --no_cuda')
        setattr(args, 'device', torch.device('cpu'))
