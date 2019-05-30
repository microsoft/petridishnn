import numpy as np

__all__ = ['sieve_loss_weights',  'eann_sieve', 
    'optimal_at',
    'exponential_weights', 'at_func', 
    'constant_weights', 'stack_loss_weights',
    'half_constant_half_optimal', 'linear',
    'recursive_heavy_end', 
    'quater_constant_half_optimal',
    'loss_weights']

def _normalize_weights(weights, normalize):
    sum_weights = np.sum(weights)
    if normalize == 'log':
        weights /= sum_weights / np.log2(len(weights)+1)
    elif normalize == 'all':
        weights /= sum_weights
    elif normalize == 'last':
        weights /= weights[-1]
    return weights


def sieve_loss_weights(N, last_weight_to_early_sum=1.0):
    if N == 1:
        return np.ones(1)
    num_steps = 0
    step = 1
    delt_weight = 1.0
    weights = np.zeros(N)
    while step < N:
        weights[0:N:step] += delt_weight
        step *= 2
        num_steps += 1
    weights[0] = np.sum(weights[1:]) * last_weight_to_early_sum
    weights = np.flipud(weights)
    weights /= np.sum(weights) / np.log2(N+1)
    return weights

def sieve_loss_weights_v2(N, last_weight_to_early_sum=1.0):
    if N == 1:
        return np.ones(1)
    weights = np.ones(N, dtype=np.float32)
    step = N/2.0
    while np.round(step) > 1:
        for i in np.arange(step, N, step):
            weights[int(i)] += 1
        step /= 2.0
    weights[-1] = np.sum(weights[:-1]) * last_weight_to_early_sum
    weights /= np.sum(weights) / np.log2(N+1)
    return weights

def eann_sieve(N):
    weights = sieve_loss_weights(N)
    weights[:N//2] = 0.0
    return weights

def constant_weights(N, normalize=False):
    weights = np.ones(N,dtype=np.float32) 
    if normalize:
        weights /= N / np.log2(N)
    return weights

def linear(N, a=0.25, b=1.0, normalize=True):
    delta = (b-a) / (N-1.0)
    weights = np.arange(N, dtype=np.float32) * delta + a
    if normalize:
        weights /= np.sum(weights) / np.log2(N)
    return weights

def optimal_at(N, optimal_l):
    """ Note that optimal_l is zero-based """
    weights = np.zeros(N)
    weights[optimal_l] = 1.0
    return weights

def half_constant_half_optimal(N, optimal_l=-1):
    weights = np.ones(N, dtype=np.float32)
    if N > 1:
        weights[optimal_l] = N-1
    weights /= np.float(N-1)
    return weights

def quater_constant_half_optimal(N):
    """
    Not sure what this was doing... emphasize the end and the start
    """
    weights = np.ones(N, dtype=np.float32)
    if N <= 2: 
        return weights
    weights[-1] = 2*N-4
    weights[0] = N-2
    weights /= np.float(4 * N - 8)
    return weights

def recursive_heavy_end(N, last_weight_to_early_sum=1.0):
    """
    N, N/2, N/4,... are set to have 1/3 of the total weights up to
    its depth. 
    N, N/2, N/4,... have weights decay exponentially

    The other weights have constant weight
    """
    weights = np.ones(N, dtype=np.float32)
    i = N-1
    w = 1.0 * N
    while True:
        weights[i] += w
        if i == 0:
            break
        i = i // 2
        w = w / 2.0 
    weights[ N*3 // 4  - 1 ] += N / 8.0
    weights[-1] = np.sum(weights[:-1]) * last_weight_to_early_sum
    weights /= np.sum(weights) / np.log2(N)
    return weights

def exponential_weights(N, base=2.0):
    weights = np.zeros(N, dtype=np.float32)
    weights[0] = 1.0
    for i in range(1,N):
        weights[i] = weights[i-1] * base
    if base >= 1.0:
        max_val = weights[-1]
    else:
        max_val = weights[0]
    weights /= max_val / int(np.log2(N))
    return weights

def at_func(N, func=lambda x:x, method=sieve_loss_weights):
    pos = []
    i = 0
    do_append = True
    while do_append:
        fi = int(func(i))
        if fi >= N:
            do_append = False
            break
        pos.append(fi)
        i += 1
    if len(pos) == 0 or pos[-1] != N-1:
        pos.append(N-1)
    #elif pos[-1] != N-1:
    #    pos = (N-1-pos[-1]) + np.asarray(pos) 
    weights = np.zeros(N, dtype=np.float32)
    weights[pos] = method(len(pos))
    return weights

def stack_loss_weights(N, stack, method=sieve_loss_weights):
    weights = np.zeros(N, dtype=np.float32)
    weights[(N-1)%stack:N:stack] = method(1+(N-1)//stack)
    return weights


def loss_weights(N, args, cfg=None):
    if hasattr(args, "weights_at_block_ends") and args.weights_at_block_ends:
        N = len(cfg)
    lwtes = 1
    if hasattr(args, 'last_weight_to_early_sum'):
        lwtes = args.last_weight_to_early_sum

    FUNC_TYPE = args.func_type
    if FUNC_TYPE == 0: # exponential spacing
        weights = at_func(N, func=lambda x:2**x, method=constant_weights)
    elif FUNC_TYPE == 1: # square spacing
        weights = at_func(N, func=lambda x:x**2, method=constant_weights)
    elif FUNC_TYPE == 2: #optimal at ?
        weights = optimal_at(N, args.opt_at)
    elif FUNC_TYPE == 3: #exponential weights
        weights = exponential_weights(N, base=args.exponential_base)
    elif FUNC_TYPE == 4: #constant weights
        weights = stack_loss_weights(N, args.stack, constant_weights)
    elif FUNC_TYPE == 5: # sieve with stack
        weights = stack_loss_weights(N, args.stack, 
            lambda N : sieve_loss_weights(N, last_weight_to_early_sum=lwtes))
    elif FUNC_TYPE == 6: # linear
        weights = stack_loss_weights(N, args.stack, linear)
    elif FUNC_TYPE == 7: # half constant, half optimal at -1
        weights = stack_loss_weights(N, args.stack, half_constant_half_optimal)
    elif FUNC_TYPE == 8: # quater constant, half optimal
        weights = stack_loss_weights(N, args.stack, quater_constant_half_optimal)
    elif FUNC_TYPE == 9: # recursive heavy end
        weights = stack_loss_weights(N, args.stack, 
            lambda N : recursive_heavy_end(N, last_weight_to_early_sum=lwtes)) 
    elif FUNC_TYPE == 10: # sieve v2 
        weights = stack_loss_weights(N, args.stack, 
            lambda N: sieve_loss_weights_v2(N, last_weight_to_early_sum=lwtes))
    elif FUNC_TYPE == 11: # constant = 1
        weights = stack_loss_weights(N, args.stack, lambda N:constant_weights(N, normalize=False))
    elif FUNC_TYPE == 12: # linear = 0.25...1
        weights = stack_loss_weights(N, args.stack, lambda N:linear(N,normalize=False))
    elif FUNC_TYPE == 13:
        weights = np.ones(N, dtype=np.float32)
        weights[N//2] = 3.0
        weights[N-1] = 3.0
        weights /= weights[-1]
    elif FUNC_TYPE == 14:
        weights = np.ones(N, dtype=np.float32)
        weights[N//4] = 3.0
        weights[N//2] = 5.0
        weights[3*N //4] = 3.0
        weights[N-1] = 5.0
        weights /= weights[-1]
    elif FUNC_TYPE == 15:
        weights = np.ones(N, dtype=np.float32)
        weights[N // 8] = 2.0
        weights[N // 4] = 3.0
        weights[3 * N // 8] = 2.0
        weights[N // 2] = 4.0
        weights[5 * N // 8] = 2.0
        weights[3 * N // 4] = 3.0
        weights[7 * N // 8] = 2.0
        weights[N-1] = 4.0
        weights /= weights[-1]
    else:
        raise NameError('func type must be either 0: exponential or 1: square' \
            + ' or 2: optimal at --opt_at, or 3: exponential weight with base --base')
    
    if hasattr(args, 'normalize_weights') and FUNC_TYPE in [5,9,10]:
        weights = _normalize_weights(weights, args.normalize_weights)
        
    if hasattr(args, "weights_at_block_ends") and args.weights_at_block_ends:
        weights_tmp = np.zeros(np.sum(cfg))
        weights_tmp[np.cumsum(cfg)-1] = weights
        weights = weights_tmp

    if hasattr(args, "min_predict_unit") and args.min_predict_unit > 0:
        weights[:args.min_predict_unit] = 0.0
        
    return weights
