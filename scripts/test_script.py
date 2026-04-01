"""Run from repository root: python scripts/test_script.py"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code_search.handle_db import addDB  # noqa: E402


def function_1():
    addDB("""import wandb
import gpustat
import torch
import math
import time
from torch import topk as TopK
from torch.distributed import is_initialized, get_rank
from helpers import get_cuda_capability, import_correct_cuda_cadam, get_gpu_mem_usage, get_first_device, ModelBlockSplitter

cuda_cadam = import_correct_cuda_cadam()

class CompressedAdamCUDALayerwise(torch.optim.Optimizer):
    def __init__(self, params, m, lr, quant_block_size, carveout=100, threads=512, streams=1, k_init=0.01, beta1=0.9, beta2=0.999, weight_decay=0, eps=1e-8, verbose=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps)
        super(CompressedAdamCUDALayerwise, self).__init__(params, defaults)
        print(f'Using CompressedAdamCUDALayerwise optimizer')

        self.m = m
        self.lr = lr
        self.quant_block_size = int(quant_block_size)
        self.k_init = k_init
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.verbose = verbose

        self.model_size = sum([p.numel() for group in self.param_groups for p in group['params']])

        self.steps = 0  # how many optimization steps were performed so far
        self.log_interval = 100
        self.device = get_first_device()

        self._is_state_initialized = False

        self.shared_memory_carveout = int(carveout) # int that represents a percentage. Accepted values: 25, 50, 100
        self.streams_count = int(streams)

        self.blocks = cuda_cadam.get_sm_count() * int(100 / self.shared_memory_carveout)
        self.threads = int(threads)

        if self.streams_count == 1:
            self.step = self.step_single_stream
            self.dict_size_count = {}  # key = layer size, value = how many layers of that size the model has
            for param in self.param_groups:
                for p in param['params']:
                    size = p.numel()
                    self.dict_size_count[size] = 1 + self.dict_size_count.get(size, 0)
            print(f'\n@@@@@ {self.blocks=}, {self.dict_size_count=}')
            # pause_process(10)
        else:
            raise RuntimeError('Multi-stream is not supported yet!')
            self.step = self.step_multi_stream
            self.multi_stream_approach = 'seq'
            # self.multi_stream_approach = 'bulk'
            self.dict_size_count = None
            self.streams_load = None
            self.streams = None
            # self.layers_count = sum([1 for group in self.param_groups for p in group['params']])
            self._multi_stream_setup()

        self._init_state()

    def _init_state(self):
        max_floats = cuda_cadam.get_max_floats_for_shared_memory_per_thread_block()
        d_block_size = max_floats // int(100 / self.shared_memory_carveout) // 2
        print(f'\n[CompressedAdamCUDALayerwise][init_state] {max_floats=} {d_block_size=} ({d_block_size * 4 * 2 / 1024}KB)\n')
        count = 0
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay) # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                if not p.requires_grad:
                    continue
                count += 1
                layer_size = p.numel()
                st = self.state[p]

                # B * t / d * nt
                st['blocks'] = max(1, int(math.floor(self.blocks * layer_size * self.dict_size_count[layer_size] / self.model_size)))
                print(f'Layer {count} of size {p.shape} has {st["blocks"]} blocks')
                # print(f'Use {st["blocks"]} for a layer of size {layer_size}')

                st['lr'] = lr
                st['weight_decay'] = wd
                st['d'] = layer_size

                ##### variables for Top-K: d_index_topk is the index where the last, smaller topk block starts
                st['d_block_size'] = layer_size if layer_size < d_block_size else d_block_size
                st['topk_full_blocks_count'], st['d_index_topk'] = ModelBlockSplitter.block_split(st['d'], st['d_block_size'])
                st['k_block_size_many'] = int(math.ceil(st['d_block_size'] * self.k_init))
                st['k_block_size_few'] = int(math.ceil((st['d'] - st['d_index_topk']) * self.k_init))  # 0 for d % d_block_size = 0
                st['k_index'] = st['topk_full_blocks_count'] * st['k_block_size_many']
                st['k'] = st['k_block_size_many'] * st['topk_full_blocks_count'] + st['k_block_size_few']

                ##### variables for the ring buffer
                st['index'] = 0  # the position to place a new gradient at
                st['I'] = torch.zeros(self.m, st['k'], dtype=torch.int16, device=self.device)  # 2mk bytes
                st['V'] = torch.zeros(self.m, st['k'], dtype=torch.bfloat16, device=self.device)  # 2mk bytes
                # print(f'{layer_size=}, {st["I"].dtype=}')

                ### variables for error feedback: d_index_quant is the index where the last, smaller quantization block starts
                # st['quant_block_size'] = layer_size if layer_size < self.quant_block_size else self.quant_block_size
                st['quant_full_blocks_count'], st['d_index_quant'] = ModelBlockSplitter.block_split(st['d'], self.quant_block_size)
                st['error'] = torch.zeros(int(math.ceil(st['d'] / 2)), dtype=torch.uint8, device=self.device)  # ceil(d/2) bytes
                st['min_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)  # ceil(d/q_bsz)*2 bytes
                st['max_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)  # ceil(d/q_bsz)*2 bytes

    def _multi_stream_setup(self):
        self.streams_load = {i: [] for i in range(self.streams_count)}  # key = stream index, value = list of parameters for that stream
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.streams_count)]

        self.dict_size_count = {}  # key = layer size, value = how many layers of that size the model has
        for param in self.param_groups:
            for p in param['params']:
                size = p.numel()
                self.dict_size_count[size] = 1 + self.dict_size_count.get(size, 0)

        print(f'[CompressedAdamCUDALayerwise][multi_stream_setup]\n\n\n')
        # for size, count in self.dict_size_count.items():
        #     print(f'{size=}: {count=}')

        # sort (size, count) by total number of parameters size*count to add them sequentially
        list_pairs_size_count = [(size, count) for size, count in self.dict_size_count.items()]
        print(f'unsorted {list_pairs_size_count=}')
        list_pairs_size_count = sorted(list_pairs_size_count, key=lambda x: x[0] * x[1], reverse=True)
        print(f'  sorted {list_pairs_size_count=}')

        # get parameters in the order given by the list_pairs_size_count
        params = []
        for size, count in list_pairs_size_count:
            for param in self.param_groups:
                for p in param['params']:
                    if p.numel() == size:
                        params.append(p)

        # add parameters to each stream list sequentially
        stream_index = 0
        for p in params:
            self.streams_load[stream_index].append(p)
            stream_index = (stream_index + 1) % self.streams_count

        # pause_process(180, 'pausing process to inspect parameters')
        # exit(666)

    @torch.no_grad()
    def step_single_stream(self, closure=None):
        self.steps += 1

        self._update_lr_wd()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        time_start = time.time()

        norm_g, norm_u, norm_e, sparsity_u = 0, 0, 0, 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                ng, nu, ne, sp_u = self.update_step(p)
                norm_g += ng
                norm_u += nu
                norm_e += ne
                sparsity_u += sp_u

        # torch.cuda.synchronize()
        time_end = time.time()
        elapsed_step = time_end - time_start
        self._log(norm_g, norm_u, norm_e, sparsity_u, elapsed_step)
        if self.verbose: print(f'Optimization step {self.steps} took {elapsed_step}')

        return loss

    @torch.no_grad()
    def step_multi_stream(self, closure=None):
        self.steps += 1

        self._update_lr_wd()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        main_stream = torch.cuda.default_stream(self.device)
        for stream in self.streams:
            stream.wait_stream(main_stream)

        norm_g, norm_u, norm_e = 0, 0, 0

        if self.multi_stream_approach == 'seq':
            load_indices = [0 for _ in range(self.streams_count)] # the indices of the current parameter to be processed in each stream
            load_lenghts = [len(self.streams_load[i]) for i in range(self.streams_count)] # contains the number of layers from each stream

            time_start = time.time()
            while any([i < n for i, n in zip(load_indices, load_lenghts)]):
                for stream_id, stream in enumerate(self.streams):
                    load_idx = load_indices[stream_id]
                    load_size = load_lenghts[stream_id]
                    if load_idx < load_size:
                        load_indices[stream_id] += 1
                        with torch.cuda.stream(stream):
                            ng, nu, ne = self.update_step(self.streams_load[stream_id][load_idx], stream)
                            norm_g += ng
                            norm_u += nu
                            norm_e += ne
        elif self.multi_stream_approach == 'bulk':
            time_start = time.time()
            for stream_id, stream in enumerate(self.streams):
                with torch.cuda.stream(stream):
                    for p in self.streams_load[stream_id]:
                        ng, nu, ne = self.update_step(p, stream)
                        norm_g += ng
                        norm_u += nu
                        norm_e += ne
        else:
            raise RuntimeError(f'Unsupported multi_stream_approach {self.multi_stream_approach}.')

        torch.cuda.synchronize()
        time_end = time.time()

        elapsed_step = time_end - time_start
        self._log(norm_g, norm_u, norm_e, elapsed_step)

        print(f'Optimization step {self.steps} took {elapsed_step} using {self.multi_stream_approach} multi-stream approach')
        return loss

    @torch.no_grad()
    def update_step(self, p):
        norm_g, norm_u, norm_e, sp_u = 0, 0, 0, 0

        st = self.state[p]
        grad = p.grad.view(-1)

        if self.steps % self.log_interval == 0:
            norm_g = grad.norm(p=2) ** 2

        blocks = st['blocks']
        lr = st['lr']
        wd = st['weight_decay']
        d = st['d']
        d_block_size = st['d_block_size']
        topk_full_blocks_count, d_index_topk = st['topk_full_blocks_count'], st['d_index_topk']
        k_block_size_many = st['k_block_size_many']
        k_block_size_few = st['k_block_size_few']
        k_index = st['k_index']
        k = st['k']

        # HuggingFace has a setting that converts st['I'] to bfloat16, even though it is declared as int16
        # This happens somewhere between constructor call and step call. Converting it to int16 was the simplest solution
        if st['I'].dtype != torch.int16:
            st['I'] = st['I'].to(torch.int16)

        index = st['index']
        I = st['I']
        V = st['V']
        if self.verbose: print(f'[update_step] {I.dtype=}, {st["I"].dtype=}')

        quant_full_blocks_count, d_index_quant = st['quant_full_blocks_count'], st['d_index_quant']
        error = st['error']
        min_vals = st['min_vals']
        max_vals = st['max_vals']

        ##### STEP 4
        if self.verbose: print(f'STEP 4: grad += Qinv(error)')
        cuda_cadam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad)

        ##### STEP 5 + 9 (only for I)
        if self.verbose: print(f'STEP 5 + 9: I_t = TopK(|grad|).indices ({I[index, :].dtype=})')
        I[index, :k_index] = TopK(input=grad[0:d_index_topk].abs().view(topk_full_blocks_count, d_block_size),
                                  k=k_block_size_many,
                                  sorted=False).indices.to(dtype=torch.int16).view(-1)
        if self.verbose:print(f'{I[index, :k_index].dtype=}')
        if k_block_size_few > 0:  # there is a small block left
            I[index, k_index:] = TopK(input=grad[d_index_topk:].abs(),
                                      k=k_block_size_few,  # example: slice has size 1, but ks[-1] is 4
                                      sorted=False).indices.to(dtype=torch.int16).view(-1)
            if self.verbose:print(f'{I[index, k_index:].dtype=}')

        # STEP 9 (only for V)
        if self.verbose:
            print(f'STEP 9: V_t = grad[I] ({V[index, :].dtype=})')
            print(f'{d=}, {self.m=}, {k=}, {d_block_size=}, {k_block_size_many=}')
            print(f'{I.shape=}, {V.shape=}, {I.dtype=}, {V.dtype=}')
            print(f'{I[index, :]=}, {I[index, :].dtype=}')
        cuda_cadam.copy_values_at_relative_indices(d,
                                                   k,
                                                   d_block_size,
                                                   k_block_size_many,
                                                   I[index, :],
                                                   grad,
                                                   V[index, :], )  # this does V[index,:] = a[I[index]]
        st['index'] = (index + 1) % self.m

        ##### STEP 6
        if self.verbose: print(f'STEP 6: grad[I] = 0')
        cuda_cadam.zerorize_block_components(grad, I[index, :], d, k, d_block_size, k_block_size_many)  # this does a[I[index]] = 0

        ##### STEP 7
        if self.verbose: print(f'STEP 7: update min & max')
        if quant_full_blocks_count == 1:
            min_vals[:quant_full_blocks_count] = grad[:d_index_quant].min()
            max_vals[:quant_full_blocks_count] = grad[:d_index_quant].max()
        else:
            min_vals[:quant_full_blocks_count] = grad[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).min(dim=1).values
            max_vals[:quant_full_blocks_count] = grad[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).max(dim=1).values
        if d_index_quant < d:
            min_vals[quant_full_blocks_count] = grad[d_index_quant:].min()
            max_vals[quant_full_blocks_count] = grad[d_index_quant:].max()

        ##### STEP 8
        if self.verbose: print(f'STEP 8: error = Q(grad)')
        cuda_cadam.asymm_block_quant(d, self.quant_block_size, error, min_vals, max_vals, grad) # error = Q(a, min, max)

        ##### STEPS 10-11
        grad.zero_()
        if self.verbose: print(f'STEP 10-11: compute Adam update')
        cuda_cadam.compute_cadam_update(blocks,  # blocks
                                        self.threads,  # threads
                                        self.shared_memory_carveout,  # carveout
                                        self.steps,  # optimization step
                                        self.beta1,  # beta1
                                        self.beta2,  # beta2
                                        self.eps,  # eps
                                        d_block_size,  # d_block_size
                                        k_block_size_many,  # k_block_size
                                        d,  # d
                                        self.m,  # m
                                        k,  # k
                                        I,  # indices
                                        V,  # values
                                        grad)  # update will be stored here

        ##### STEP 12
        if self.verbose: print(f'STEP 12: update model')
        p.mul_(1 - lr * wd).add_(p.grad, alpha=-lr)

        # compute error norm
        if self.steps % self.log_interval == 0:
            norm_u = grad.norm(p=2) ** 2
            sp_u = (grad == 0).sum() # check sparsity before zerorizing

            grad.zero_()
            cuda_cadam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad)

            norm_e = grad.norm(p=2) ** 2

        ### record tensors to stream
        # if stream is not None: p.record_stream(stream)
        # if stream is not None: I.record_stream(stream)
        # if stream is not None: V.record_stream(stream)
        # if stream is not None: error.record_stream(stream)
        # if stream is not None: min_vals.record_stream(stream)
        # if stream is not None: max_vals.record_stream(stream)

        return norm_g, norm_u, norm_e, sp_u

    def _log(self, norm_g, norm_u, norm_e, sparsity_u, elapsed_step):
        if self.steps % self.log_interval == 0:
            wandb_data = dict(
                optimizer_steps=self.steps,
                gpu_mem_usage=get_gpu_mem_usage(),
                norm_g=math.sqrt(norm_g),
                norm_u=math.sqrt(norm_u),
                norm_error=math.sqrt(norm_e),
                sparsity_u=sparsity_u / self.model_size * 100.,
                elapsed_step=elapsed_step
            )

            if self.steps == 1:
                wandb_data['carveout'] = self.shared_memory_carveout  # log this only once

            if not is_initialized() or get_rank() == 0:
                wandb.log(wandb_data, commit=False)
                print(wandb_data)

    def _update_lr_wd(self):
        # copy the learning rate group to parameter state because the lr scheduler updates the one in the group
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay)  # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                self.state[p]['lr'] = lr
                self.state[p]['wd'] = wd
""", "adam_cudalayerwise_optimizer", "1234567890")