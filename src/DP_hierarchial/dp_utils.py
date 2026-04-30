from __future__ import annotations

import logging
import random
from typing import List

import numpy as np
import types
import torch.nn.functional as F
import torch
import torch.distributed as dist
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.validators import ModuleValidator

logger = logging.getLogger(__name__)

def patch_densenet_inplace_forward(model: torch.nn.Module) -> torch.nn.Module:
    """
    Torchvision DenseNet uses a hardcoded functional call:

        F.relu(features, inplace=True)

    inside DenseNet.forward(). This is not an nn.ReLU module.
    Opacus backward hooks are incompatible with it, so we patch it to
    inplace=False after ModuleValidator.fix().
    """

    def safe_densenet_forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    patched = 0

    for module in model.modules():
        if (
            module.__class__.__name__ == "DenseNet"
            and hasattr(module, "features")
            and hasattr(module, "classifier")
        ):
            module.forward = types.MethodType(safe_densenet_forward, module)
            patched += 1

    if patched > 0:
        logger.warning(
            "Patched %d DenseNet forward method(s) to use F.relu(..., inplace=False).",
            patched,
        )

    return model

def disable_inplace_ops(model: torch.nn.Module) -> torch.nn.Module:
    """
    Opacus GradSampleModule is not safe with inplace ops.
    DenseNet/torchvision can contain inplace ReLU operations that trigger:

    RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace.

    This function both:
    1. Replaces activation modules where possible.
    2. Sets any remaining module.inplace = False.
    """

    def _replace(module: torch.nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU(inplace=False))

            elif isinstance(child, nn.ReLU6):
                setattr(module, name, nn.ReLU6(inplace=False))

            elif isinstance(child, nn.LeakyReLU):
                setattr(
                    module,
                    name,
                    nn.LeakyReLU(
                        negative_slope=child.negative_slope,
                        inplace=False,
                    ),
                )

            elif isinstance(child, nn.ELU):
                setattr(
                    module,
                    name,
                    nn.ELU(
                        alpha=child.alpha,
                        inplace=False,
                    ),
                )

            elif isinstance(child, nn.CELU):
                setattr(
                    module,
                    name,
                    nn.CELU(
                        alpha=child.alpha,
                        inplace=False,
                    ),
                )

            elif isinstance(child, nn.SELU):
                setattr(module, name, nn.SELU(inplace=False))

            elif isinstance(child, nn.SiLU):
                setattr(module, name, nn.SiLU(inplace=False))

            elif isinstance(child, nn.Hardswish):
                setattr(module, name, nn.Hardswish(inplace=False))

            else:
                _replace(child)

    _replace(model)

    # Extra safety: any module with an inplace attribute gets disabled.
    for module in model.modules():
        if hasattr(module, "inplace"):
            try:
                module.inplace = False
            except Exception:
                pass

    return model


def assert_no_inplace_ops(model: torch.nn.Module) -> None:
    bad = []

    for name, module in model.named_modules():
        if hasattr(module, "inplace") and bool(module.inplace):
            bad.append(f"{name}: {module.__class__.__name__}(inplace=True)")

    if bad:
        raise RuntimeError(
            "Inplace modules remain after disable_inplace_ops():\n"
            + "\n".join(bad)
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def disable_inplace_ops(model: torch.nn.Module) -> torch.nn.Module:
    """
    Opacus GradSampleModule is not safe with inplace ops such as ReLU(inplace=True).
    Convert all inplace ReLU modules to non-inplace.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    return model


def get_learning_rate(cfg, default: float = 1e-4) -> float:
    for name in ["lr", "learning_rate", "learning_rate_backbone"]:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return default


def get_weight_decay(cfg, default: float = 0.0) -> float:
    for name in ["weight_decay", "wd"]:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, (int, float)) and value >= 0:
                return float(value)
    return default


def make_opacus_compatible(model: torch.nn.Module) -> torch.nn.Module:
    """
    Make the model compatible with Opacus.

    Important:
    - Do NOT patch DenseNet.forward before ModuleValidator.fix().
      ModuleValidator.fix() clones the model internally, and monkey-patched
      methods can break cloning.
    - First let Opacus replace BatchNorm.
    - Then patch DenseNet's hardcoded F.relu(..., inplace=True).
    """

    # Safe before cloning: module-level inplace flags only
    model = disable_inplace_ops(model)

    errors = ModuleValidator.validate(model, strict=False)

    if errors:
        logger.warning(
            "Model is not fully Opacus-compatible. Applying ModuleValidator.fix()."
        )

        for err in errors:
            logger.warning("Opacus validator warning: %s", err)

        # Important: this internally clones/pickles the model.
        # So DenseNet monkey-patching must NOT happen before this line.
        model = ModuleValidator.fix(model)

    # After fix, clean again
    model = disable_inplace_ops(model)

    # Now patch torchvision DenseNet's functional inplace ReLU
    model = patch_densenet_inplace_forward(model)

    remaining = ModuleValidator.validate(model, strict=False)

    if remaining:
        logger.warning("Some Opacus compatibility warnings remain after fix:")
        for err in remaining:
            logger.warning("Remaining Opacus warning: %s", err)

    return model


def unwrap_private_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap Opacus / DPDDP wrappers to access the real underlying model.

    Possible wrapping order:
    DPDDP -> GradSampleModule -> original model
    """
    current = model

    # DPDDP usually stores the wrapped model in .module
    if hasattr(current, "module"):
        current = current.module

    # Opacus GradSampleModule usually stores original model in ._module
    if hasattr(current, "_module"):
        current = current._module

    return current


def get_trainable_named_parameters(model: torch.nn.Module):
    """
    Return only trainable parameters.

    We intentionally do NOT use state_dict() for Flower communication because
    state_dict() also includes non-trainable buffers such as:
    - BatchNorm running_mean
    - BatchNorm running_var
    - num_batches_tracked
    - pos_weight / class weights

    Those buffers can differ between the master and hospitals, especially after
    Opacus ModuleValidator.fix().
    """
    base_model = unwrap_private_model(model)

    return [
        (name, param)
        for name, param in base_model.named_parameters()
        if param.requires_grad
    ]


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Flower parameter serialization.

    Send only trainable parameters in named_parameters() order.
    Do not send full state_dict().
    """
    trainable_params = get_trainable_named_parameters(model)

    return [
        param.detach().cpu().numpy()
        for _, param in trainable_params
    ]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Flower parameter loading.

    Load only trainable parameters. Keep local buffers unchanged.
    This prevents mismatches caused by BatchNorm buffers or hospital-specific
    pos_weight values.
    """
    base_model = unwrap_private_model(model)
    trainable_params = get_trainable_named_parameters(base_model)

    if len(parameters) != len(trainable_params):
        raise ValueError(
            f"Flower parameter length mismatch: received {len(parameters)} tensors, "
            f"but model expects {len(trainable_params)} trainable parameters. "
            f"This usually means master/coordinator/worker are not using the same "
            f"get_parameters/set_parameters functions or not using the same model."
        )

    for idx, ((name, param), new_param) in enumerate(zip(trainable_params, parameters)):
        new_tensor = torch.tensor(
            new_param,
            dtype=param.data.dtype,
            device=param.data.device,
        )

        if tuple(new_tensor.shape) != tuple(param.data.shape):
            raise ValueError(
                f"Shape mismatch at trainable parameter index {idx}: {name}. "
                f"Received {tuple(new_tensor.shape)}, expected {tuple(param.data.shape)}."
            )

        param.data.copy_(new_tensor)


def broadcast_model_state(model: torch.nn.Module, src: int = 0) -> None:
    """
    Broadcast rank-0 model state to every DDP worker.

    This can still use state_dict() because it happens inside the local DDP group,
    where rank 0 and worker ranks use the same DP-compatible model construction.
    Flower communication should use named_parameters(), but local rank broadcast
    can safely broadcast full model state.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    base_model = unwrap_private_model(model)
    state_dict = base_model.state_dict()

    for _, tensor in state_dict.items():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def broadcast_trainable_parameters(model: torch.nn.Module, src: int = 0) -> None:
    """
    Optional safer broadcast: broadcast only trainable parameters.

    Use this instead of broadcast_model_state() if state_dict broadcast causes
    mismatch between coordinator and workers.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    base_model = unwrap_private_model(model)

    for _, param in get_trainable_named_parameters(base_model):
        dist.broadcast(param.data, src=src)


def setup_process_group(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    if dist.is_available() and dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    import os

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    logger.info(
        "Initializing torch.distributed process group | backend=%s rank=%d world_size=%d master=%s:%d",
        backend,
        rank,
        world_size,
        master_addr,
        master_port,
    )

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def attach_dpddp_privacy_engine(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    noise_multiplier: float,
    max_grad_norm: float,
    secure_mode: bool = False,
):
    """
    Attach Opacus PrivacyEngine for distributed DP training.

    Important:
    - Use DPDDP, not regular torch.nn.parallel.DistributedDataParallel.
    - Pass a normal non-distributed train_loader to make_private.
      Opacus handles private distributed sampling internally.
    """
    model = DPDDP(model)

    privacy_engine = PrivacyEngine(secure_mode=secure_mode)

    private_model, private_optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    return private_model, private_optimizer, private_train_loader, privacy_engine