"""Reconciler — Outpost's control plane.

State machine + MongoDB/file store + NodeRegistry (exclusive locks) + the
ProviderAdapter interface that drives a job through:

    submit -> provision -> bootstrap -> run -> collect -> teardown
           -> validate -> promote

Execution is dry-run by default (no VMs touched); pass --execute to wire the
libvirt adapter + ansible in for real.
"""

__all__ = ["states", "models", "store"]
