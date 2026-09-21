"""Give the CPU libraries the number of cores this container actually has.

PyTorch and onnxruntime both size their thread pools from the *host's* core count, and neither
of them looks at the cgroup CPU quota a container is given. On the machine this was found on
that means twenty threads inside a four-CPU container: the threads do not get more CPU, they
just take turns, and the work gets slower rather than faster. Measured inside a 4-CPU
container, at the deployed input size:

    torch threads 8  ->  100 ms per detection    (what it picks on its own)
    torch threads 4  ->   68 ms                  (the quota)
    torch threads 2  ->   90 ms

    classifier, onnxruntime default  ->  1.594 ms per window
    classifier, intra_op 4           ->  0.188 ms per window   (8.5x)

This matters most where it hurts most. The production server is a four-core VM with no GPU,
where the pose pass is the whole budget and frame rate is the largest single factor in whether
a fall is caught at all -- so threads spent fighting each other come straight off recall.

`cpu_quota()` reads the quota the way the kernel reports it, cgroup v2 first and then v1, and
falls back to the host count when neither is present (a bare-metal install, where the host
count is the right answer anyway). `CPU_THREADS` overrides it for anyone who wants to leave
cores for something else.
"""
import os

_CGROUP_V2 = '/sys/fs/cgroup/cpu.max'
_CGROUP_V1_QUOTA = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'
_CGROUP_V1_PERIOD = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def cpu_quota():
    """How many CPUs this process may actually use, as an integer of at least 1."""
    override = os.environ.get('CPU_THREADS')
    if override and override.isdigit() and int(override) > 0:
        return int(override)

    v2 = _read(_CGROUP_V2)
    if v2 and not v2.startswith('max'):
        try:
            quota, period = (float(x) for x in v2.split())
            return max(1, int(quota / period))
        except ValueError:
            pass

    quota, period = _read(_CGROUP_V1_QUOTA), _read(_CGROUP_V1_PERIOD)
    try:
        if quota and period and float(quota) > 0:
            return max(1, int(float(quota) / float(period)))
    except ValueError:
        pass

    # No quota set: the host's count is the honest answer. os.sched_getaffinity is better than
    # os.cpu_count where it exists, because a VM can have cores allocated but offline -- which
    # is exactly the case on the production server, 14 allocated and 4 online.
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def tune_threads(reason=''):
    """Pin torch (and anything reading OMP) to the quota. Call at the start of a task, after
    any fork, and after the model is loaded -- ultralytics sets the thread count itself when it
    builds a model, so doing this earlier is silently undone.

    Returns the number set, so callers can log what actually happened rather than what was
    intended.
    """
    n = cpu_quota()
    os.environ.setdefault('OMP_NUM_THREADS', str(n))
    os.environ.setdefault('MKL_NUM_THREADS', str(n))
    try:
        import torch
        if torch.get_num_threads() != n:
            torch.set_num_threads(n)
        torch.set_num_interop_threads(1) if _interop_unset(torch) else None
    except Exception:                       # torch missing or already started: not fatal
        pass
    return n


def _interop_unset(torch):
    """set_num_interop_threads raises once the pool exists, so only try it while it is safe."""
    try:
        return torch.get_num_interop_threads() != 1
    except Exception:
        return False


def ort_session_options():
    """SessionOptions sized to the quota, for every onnxruntime session this app creates.

    Left to itself onnxruntime opens one thread per host core for a model that is 560 KB and
    runs in a fifth of a millisecond; inside a small container that is pure contention.
    """
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = cpu_quota()
    opts.inter_op_num_threads = 1
    return opts
