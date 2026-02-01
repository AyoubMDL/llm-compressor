from math import ceil

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    dequantize,
    quantize,
)
from compressed_tensors.utils import get_execution_device


def _shrink_op(x, beta, lp_norm):
    return torch.sign(x) * torch.nn.functional.relu(
        torch.abs(x) - 1.0 / beta * torch.pow(torch.abs(x), lp_norm - 1)
    )


@torch.inference_mode()
def optimize_weights_proximal(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    *,
    lp_norm: float = 0.7,
    beta: float = 1e1,
    kappa: float = 1.01,
    iters: int = 20,
    early_stop: bool = True,
):
    """Quantize the scale/zero of quantized tensor using the HQQ.

    Args:
        module (torch.nn.Module): The quantized module containing weight, scale,
            and zero-point.
        quant_args (QuantizationArgs): The quantization arguments.
        lp_norm (float, optional): The Lp norm to use for the proximal operator.
            Defaults to 0.7.
        beta (float, optional): The initial beta value for the optimization.
            Defaults to 1e1.
        kappa (float, optional): The scaling factor for beta. Defaults to 1.01.
        iters (int, optional): The number of iterations for optimization.
            Defaults to 20.
        early_stop (bool, optional): Whether to stop early if no improvement is seen.
            Defaults to True.

    Returns:
        tuple: A tuple containing the optimized scale and zero-point tensors.
    """
    device = get_execution_device(module)
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    # Update zp_dtype in quant_args for optimization
    quant_args.zp_dtype = dtype

    weights = module.weight.clone()
    W_f = weights.to(dtype)
    scale = module.weight_scale.to(dtype)
    zero_point = module.weight_zero_point.to(dtype)

    best_error = 1e4
    best_zero_point = zero_point.clone()

    for _ in range(iters):
        W_q = quantize(W_f, scale, zero_point, quant_args)
        W_r = dequantize(W_q, scale, zero_point, quant_args)
        W_e = _shrink_op(W_f - W_r, beta, lp_norm)

        # HQQ uses inverse scale for quantization
        if quant_args.strategy == QuantizationStrategy.TENSOR:
            zero_point = torch.mean(W_q - (W_f - W_e) * (1.0 / scale)).view(1)

        elif quant_args.strategy == QuantizationStrategy.CHANNEL:
            # For CHANNEL/TENSOR: compute mean along input dimension
            zero_point = torch.mean(
                W_q - (W_f - W_e) * (1.0 / scale), axis=1, keepdim=True
            )
        elif quant_args.strategy in (
            QuantizationStrategy.GROUP,
            QuantizationStrategy.TENSOR_GROUP,
        ):
            reshaped_dims = (
                ceil(W_q.shape[-1] / quant_args.group_size),
                quant_args.group_size,
            )
            W_q = W_q.unflatten(-1, reshaped_dims)
            W_f = W_f.unflatten(-1, reshaped_dims)
            W_e = W_e.unflatten(-1, reshaped_dims)

            # For GROUP: reshape to (num_rows, num_groups, group_size)
            # then compute mean within each group (last dim)
            W_diff = W_q - (W_f - W_e) * (1.0 / scale).unsqueeze(-1)
            W_diff = W_diff.flatten(start_dim=-2)
            W_q = W_q.flatten(start_dim=-2)
            W_f = W_f.flatten(start_dim=-2)
            W_e = W_e.flatten(start_dim=-2)

            num_rows, num_cols = W_diff.shape
            num_groups = num_cols // quant_args.group_size
            W_diff_grouped = W_diff.reshape(num_rows, num_groups, quant_args.group_size)
            zero_point = torch.mean(W_diff_grouped, axis=2)

        else:
            raise ValueError(
                f"Quantization strategy is not supported for HQQ: "
                f"{quant_args.strategy}"
            )

        beta *= kappa

        # Compute current error
        current_error = float(torch.abs(W_f - W_r).mean())
        if current_error < best_error:
            best_error = current_error
            best_zero_point = zero_point.clone()

            if early_stop:
                break

    del W_f, W_q, W_r, W_e

    # Cast zero point to scale dtype
    return best_zero_point.to(module.weight_scale.dtype)
