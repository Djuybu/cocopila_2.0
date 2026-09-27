"""Sample process RSS and record torch CUDA allocation when available."""
import sys
import threading
import time


class ResourceProfile:
    def __enter__(self):
        self.started, self.ram_peak, self._stop = time.perf_counter(), None, threading.Event()
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
        torch = sys.modules.get("torch")
        self.cuda = torch is not None and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        def sample():
            while not self._stop.is_set():
                if self.process:
                    value = self.process.memory_info().rss
                    self.ram_peak = max(self.ram_peak or 0, value)
                self._stop.wait(.05)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self.thread.join(timeout=2)
        torch = sys.modules.get("torch")
        # The model may have imported torch after profiling started.
        cuda = torch is not None and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        if self.process:
            self.ram_peak = max(self.ram_peak or 0, self.process.memory_info().rss)
        self.result = {"wall_seconds": time.perf_counter() - self.started,
                       "peak_process_rss_bytes": self.ram_peak,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
                       "memory_scope": "sampled_process_rss_and_torch_cuda_allocator"}
