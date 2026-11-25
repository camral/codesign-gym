"""Limit backend startup chatter without filtering warnings or stderr."""
from functools import wraps
from threading import RLock


# Warp's logging configuration is process-wide. Serialize our construction
# scopes so overlapping factories cannot restore one another's settings.
_WARP_LOG_LOCK = RLock()


def warp_construction(build=None, *, forward_verbose=False):
    """Decorate a Warp builder with quiet startup and an opt-in verbose kwarg.

    Configuration is restored even if construction raises. Only backend INFO
    logs are reduced; warnings, errors, and environment diagnostics stay visible.
    """
    if build is None:
        return lambda fn: warp_construction(fn, forward_verbose=forward_verbose)

    @wraps(build)
    def wrapped(*args, verbose=None, **kwargs):
        import warp as wp

        config = kwargs.get("config")
        enabled = bool(kwargs.get("debug", False) or getattr(config, "debug", False))
        enabled |= bool(getattr(config, "verbose", False) if verbose is None else verbose)
        if forward_verbose and verbose is not None:
            kwargs["verbose"] = verbose
        with _WARP_LOG_LOCK:
            # Warp 1.17 exposes levels; 1.16 (our minimum) uses config.quiet.
            attr = "log_level" if hasattr(wp.config, "log_level") else "quiet"
            previous = getattr(wp.config, attr)
            if attr == "log_level":
                value = wp.LOG_INFO if enabled else max(previous, wp.LOG_WARNING)
            else:
                value = not enabled
            setattr(wp.config, attr, value)
            try:
                return build(*args, **kwargs)
            finally:
                setattr(wp.config, attr, previous)

    return wrapped
