"""Confirm the GPU is real, visible to TensorFlow, and fast. Run inside a job.

Worth its own script because there are three separate ways this goes wrong and
they look identical from the pipeline: no GPU allocated, a GPU allocated that
TF cannot see (CUDA/driver mismatch), and a GPU that TF sees but silently falls
back off. Each would just look like "the run is slow".
"""

import time

import numpy as np
import tensorflow as tf


def main() -> int:
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible to TensorFlow: {len(gpus)}")
    for g in gpus:
        try:
            d = tf.config.experimental.get_device_details(g)
            print(f"   {g.name}  {d.get('device_name', '?')}  "
                  f"compute {d.get('compute_capability', '?')}")
        except Exception:
            print(f"   {g.name}")

    if not gpus:
        print("NO GPU -- the pipeline would run on CPU without complaining")
        return 1

    n = 4096
    with tf.device("/GPU:0"):
        a = tf.random.normal([n, n])
        b = tf.random.normal([n, n])
        tf.matmul(a, b).numpy()          # warm up: first call builds the kernel
        t = time.time()
        for _ in range(5):
            c = tf.matmul(a, b)
        c.numpy()
        dt = (time.time() - t) / 5
    print(f"   {n}^3 matmul: {dt * 1000:.1f} ms -> "
          f"{2 * n ** 3 / dt / 1e12:.1f} TFLOP/s (fp32)")

    # The real question is not matmul but whether StarDist runs on it.
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained("2D_versatile_he")
    rgb = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    norm = rgb.astype(np.float32) / 255.0
    model.predict_instances(norm)        # warm up
    t = time.time()
    labels, _ = model.predict_instances(norm)
    print(f"   StarDist 512x512 tile: {(time.time() - t) * 1000:.0f} ms, "
          f"{labels.max()} objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
