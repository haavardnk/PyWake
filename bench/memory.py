import ctypes
import ctypes.util
import sys

import psutil

if sys.platform == "darwin":
    LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
    RUSAGE_INFO_V2 = 2
    PHYS_FOOTPRINT = slice(72, 80)

    def footprint(pid):
        buffer = ctypes.create_string_buffer(512)
        if LIBC.proc_pid_rusage(pid, RUSAGE_INFO_V2, buffer) != 0:
            return 0
        return int.from_bytes(buffer.raw[PHYS_FOOTPRINT], sys.byteorder)

    def free_percent():
        level, size = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if LIBC.sysctlbyname(b"kern.memorystatus_level", ctypes.byref(level), ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "kern.memorystatus_level")
        return level.value

else:

    def footprint(pid):
        try:
            return psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:
            return 0

    def free_percent():
        memory = psutil.virtual_memory()
        return 100 * memory.available / memory.total
