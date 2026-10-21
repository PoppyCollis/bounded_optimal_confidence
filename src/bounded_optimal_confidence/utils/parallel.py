"""Worker pools that never leave orphans behind.

`with ProcessPoolExecutor(...)` is not enough on its own: __exit__ calls
shutdown(wait=True), which does not kill the workers, it waits for them. Ctrl-C
therefore blocks for the length of the longest in-flight job, and if the parent is
SIGKILLed, the terminal closes, or the OOM killer fires, no parent code runs at all
and the children reparent to init and keep burning cores until someone kills them
by hand.

So the guarantee has to live in the child (see `init_worker`), with parent-side
teardown layered on top (see `worker_pool`) so that an interrupt returns promptly.
"""
import ctypes
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

_BLAS_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)

_PR_SET_PDEATHSIG = 1  # <sys/prctl.h>
_PARENT_POLL_SECONDS = 5.0
_TERMINATE_GRACE_SECONDS = 2.0


def _set_pdeathsig():
    """Ask the kernel to SIGKILL this process when its parent dies.

    Linux-only; a missing or incompatible libc is not fatal, since the getppid
    watchdog below covers the same ground more slowly."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except (OSError, AttributeError):
        pass


def _watch_parent(original_ppid):
    """Backstop for _set_pdeathsig, and the only mechanism off Linux.

    PR_SET_PDEATHSIG fires when the spawning *thread* exits rather than when the
    parent process does, so it is deliberately not relied on alone."""
    while True:
        time.sleep(_PARENT_POLL_SECONDS)
        if os.getppid() != original_ppid:
            os._exit(1)


def init_worker():
    """ProcessPoolExecutor initializer: pin BLAS to 1 thread per worker, then arrange
    for the worker to die with its parent.

    The thread pinning stops N_WORKERS processes each spawning a thread-per-core BLAS
    pool underneath (oversubscription). Combined with the 'spawn' start method, these
    env vars are set before numpy/BLAS initializes in the child."""
    for var in _BLAS_VARS:
        os.environ[var] = "1"

    _set_pdeathsig()

    # The parent may already have died in the window before the prctl call landed, in
    # which case the signal it arms will never be delivered.
    ppid = os.getppid()
    if ppid == 1:
        os._exit(1)

    threading.Thread(target=_watch_parent, args=(ppid,), daemon=True).start()


def _hard_shutdown(executor, grace=_TERMINATE_GRACE_SECONDS):
    """Cancel what has not started and kill what has, rather than waiting it out."""
    # Snapshot first: shutdown() clears the executor's process table.
    processes = list((getattr(executor, "_processes", None) or {}).values())
    if not processes:
        processes = multiprocessing.active_children()

    executor.shutdown(wait=False, cancel_futures=True)

    for proc in processes:
        proc.terminate()

    deadline = time.monotonic() + grace
    for proc in processes:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))
    for proc in processes:
        if proc.is_alive():
            proc.kill()


@contextmanager
def worker_pool(max_workers, initializer=init_worker):
    """A ProcessPoolExecutor whose workers are cleaned up on every exit path.

    Normal exit joins the workers as before, so results are unchanged. Any exception
    -- KeyboardInterrupt included -- tears the pool down immediately instead of
    blocking in shutdown(wait=True) until the longest running job happens to finish."""
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
    )
    try:
        yield executor
    except BaseException:
        _hard_shutdown(executor)
        raise
    else:
        executor.shutdown(wait=True)


def run_parallel(fn, tasks, n_workers, label, every=50):
    """[fn(task) for task in tasks], computed in `n_workers` processes, in task order. Prints progress."""
    out = []
    with worker_pool(n_workers) as ex:
        for n, result in enumerate(ex.map(fn, tasks, chunksize=1), 1):
            out.append(result)
            if n % every == 0 or n == len(tasks):
                print(f"  {label}: {n}/{len(tasks)}", flush=True)
    return out
