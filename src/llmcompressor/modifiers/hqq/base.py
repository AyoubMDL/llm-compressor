import torch
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    getattr_chain,
    match_named_modules,
    register_offload_parameter,
)
from torch.nn import Parameter

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.hqq.optimizer import optimize_weights_proximal
from llmcompressor.modifiers.quantization import QuantizationModifier

__all__ = ["HQQModifier"]


class HQQModifier(QuantizationModifier):
    # Hqq modifier arguments
    lp_norm: float = 0.7
    beta: float = 1e1
    kappa: float = 1.01
    iters: int = 20
    early_stop: bool = True

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self._optimize(state.model)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self._optimize(state.model)

            if not self.ended_:
                self.on_end(state, None)

    def _optimize(self, model):
        named_modules = list(
            match_named_modules(model, self.resolved_targets, self.ignore)
        )

        for _, module in named_modules:
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            with torch.no_grad(), align_module_device(module):
                zero_point = optimize_weights_proximal(
                    module,
                    quant_args,
                    lp_norm=self.lp_norm,
                    beta=self.beta,
                    kappa=self.kappa,
                    iters=self.iters,
                )

            # Delete old zero point
            delete_offload_parameter(module, "weight_zero_point")

            # Register new zero point with float dtype
            register_offload_parameter(
                module, "weight_zero_point", Parameter(zero_point, requires_grad=False)
            )

            # Update quant_args zp_dtype
            quant_args.zp_dtype = zero_point.dtype
            quant_args.symmetric = False
