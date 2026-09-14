

class TestSshKeepalives:
    """A dead connection must fail in bounded time, not hang the whole loop."""

    def test_every_ssh_and_scp_argv_bounds_a_dead_connection(self):
        from reconciler.adapter import scp_to, ssh_base

        argv = ssh_base("10.0.0.1", "ubuntu", "/k")
        for opt in ("ServerAliveInterval=", "ServerAliveCountMax=", "ConnectTimeout="):
            assert any(a.startswith(opt) for a in argv), f"ssh_base missing {opt}"

        seen = {}
        import reconciler.adapter as ad
        orig = ad.run_logged
        ad.run_logged = lambda argv, job_id, **kw: seen.setdefault("argv", argv)
        try:
            scp_to("10.0.0.1", "/src", "/dst", "job-a", "ubuntu", "/k")
        finally:
            ad.run_logged = orig
        for opt in ("ServerAliveInterval=", "ServerAliveCountMax=", "ConnectTimeout="):
            assert any(a.startswith(opt) for a in seen["argv"]), f"scp_to missing {opt}"

    def test_the_bound_is_tunable_without_a_code_change(self, monkeypatch):
        # An operator on a slow or lossy link must be able to widen the window
        # rather than patch the adapter.
        import importlib

        import reconciler.adapter as ad
        monkeypatch.setenv("CLUSTER_SSH_ALIVE_INTERVAL", "5")
        monkeypatch.setenv("CLUSTER_SSH_ALIVE_COUNT", "3")
        importlib.reload(ad)
        try:
            assert "ServerAliveInterval=5" in ad.ssh_base("1.2.3.4")
            assert "ServerAliveCountMax=3" in ad.ssh_base("1.2.3.4")
        finally:
            monkeypatch.undo()
            importlib.reload(ad)
