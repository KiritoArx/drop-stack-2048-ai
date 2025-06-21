import jax


def print_device_info() -> None:
    """Print a simple report on the available JAX devices."""
    backend = jax.default_backend()
    devices = jax.local_devices()
    if backend in ("gpu", "tpu"):
        mark = "\u2705"  # check mark
        msg = f"{mark} GPU available - using {backend.upper()}"
    else:
        mark = "\u274C"  # cross mark
        msg = f"{mark} GPU not available - using CPU"
    print(msg)
    device_list = ", ".join([d.platform for d in devices])
    print(f"Detected device(s): {device_list}")
