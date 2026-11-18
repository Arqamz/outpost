"""AMD Instinct (ROCm) GPU nodes — the `gpu_vendor: amd` branches.

The rule these serve: an AMD node gets `apptainer --rocm` and its GPUs pinned
through ROCR_VISIBLE_DEVICES (ROCr `GPU-<hex>` UUIDs), and an NVIDIA node gets
exactly what it got before vendors existed. A backend that only knows the
NVIDIA path refuses an AMD node instead of starting it with no GPU.

The amd-smi fixtures under fixtures/amd-smi/ and topology_mi300x_vf_1gpu.json
were captured on a real DigitalOcean MI300X VF droplet (amd-smi 27.0.0, ROCm
10.0.0 host, amdgpu 7.1.3), whose rocminfo reports the device as
`GPU-8ac1449593d1d319`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler.adapter import (LibvirtAdapter, StaticSshAdapter, container_argv,
                                job_gpu_vendor, require_nvidia)
from reconciler.launcher import LocalProcessLauncherAdapter, OpenMPILauncherAdapter
from reconciler.models import JobSpec, NodeRecord, gpu_visible_env, normalize_gpu_vendor
from reconciler.placement import resolve
from reconciler.receipt import build_receipt
from reconciler.topology import amd_gpus, normalize, rocr_uuid

FIXTURES = Path(REPO_DIR) / "tests" / "fixtures"
AMD = FIXTURES / "amd-smi"
EXAMPLES = Path(REPO_DIR) / "contract" / "launch-intent" / "v1" / "examples"
MI300X_UUID = "GPU-8ac1449593d1d319"


def amd_topo(node: str = "fp-mi300x-1"):
    return normalize(json.loads((FIXTURES / "topology_mi300x_vf_1gpu.json").read_text()), node)


def intent() -> dict:
    return json.loads((EXAMPLES / "one-rank-per-gpu.json").read_text())


def amd_plan(node: str = "fp-mi300x-1"):
    return resolve(intent(), [amd_topo(node)])


def gpu_spec(**kw) -> JobSpec:
    return JobSpec(**{"name": "rocm", "image": "/sif/rocm.sif", "command": ["rocminfo"],
                      "gpu": True, "output_dir": "/out", **kw})


def amd_node(name="fp-mi300x-1", **kw) -> NodeRecord:
    base = dict(node_id=name, name=name, index=-1, ip="203.0.113.7", gpu=True,
                provider="static-ssh", gpu_vendor="amd")
    return NodeRecord(**{**base, **kw})


# ── vendor basics ──────────────────────────────────────────────────────────

class TestVendor:
    def test_visible_env_per_vendor(self):
        assert gpu_visible_env("nvidia") == "CUDA_VISIBLE_DEVICES"
        assert gpu_visible_env("amd") == "ROCR_VISIBLE_DEVICES"
        assert gpu_visible_env("") == "CUDA_VISIBLE_DEVICES"
        assert gpu_visible_env(None) == "CUDA_VISIBLE_DEVICES"

    def test_an_unknown_vendor_is_refused_not_run_as_nvidia(self):
        with pytest.raises(ValueError, match="unknown gpu_vendor"):
            normalize_gpu_vendor("intel")

    def test_a_record_written_before_vendors_existed_is_nvidia(self):
        old = {"node_id": "pc", "name": "pc", "index": -1, "ip": "1.2.3.4", "gpu": True}
        assert NodeRecord.from_dict(old).gpu_vendor == "nvidia"

    def test_the_vendor_survives_a_store_round_trip(self):
        assert NodeRecord.from_dict(amd_node().to_dict()).gpu_vendor == "amd"


# ── container argv ─────────────────────────────────────────────────────────

class TestContainerArgv:
    def test_amd_gets_rocm_and_not_nv(self):
        argv = container_argv(gpu_spec(), "/w", "/out", gpu_vendor="amd")
        assert "--rocm" in argv and "--nv" not in argv
        assert argv[:3] == ["apptainer", "exec", "--rocm"]

    def test_nvidia_argv_is_unchanged(self):
        # The exact argv an NVIDIA GPU job always got.
        assert container_argv(gpu_spec(), "/w", "/out") == [
            "apptainer", "exec", "--nv", "--bind", "/w:/out", "/sif/rocm.sif", "rocminfo"]
        assert container_argv(gpu_spec(), "/w", "/out", gpu_vendor="nvidia") == \
            container_argv(gpu_spec(), "/w", "/out")

    def test_a_cpu_job_gets_no_gpu_flag_on_either_vendor(self):
        argv = container_argv(gpu_spec(gpu=False), "/w", "/out", gpu_vendor="amd")
        assert "--rocm" not in argv and "--nv" not in argv

    def test_docker_has_no_amd_path_and_says_so(self):
        with pytest.raises(RuntimeError, match="runtime: apptainer"):
            container_argv(gpu_spec(runtime="docker"), "/w", "/out", gpu_vendor="amd")


# ── amd-smi parsing (real fixtures) ────────────────────────────────────────

class TestAmdSmiParsing:
    def test_rocr_uuid_from_asic_serial(self):
        assert rocr_uuid("0x8AC1449593D1D319") == MI300X_UUID
        assert rocr_uuid("8ac1449593d1d319") == MI300X_UUID
        assert rocr_uuid("") == "" and rocr_uuid(None) == "" and rocr_uuid("N/A") == ""

    def test_the_three_captured_views_parse_into_one_gpu(self):
        # The full `amd-smi static --json` (wrapped in gpu_data) as captured.
        gpus = amd_gpus({"list": (AMD / "list.json").read_text(),
                         "static": (AMD / "static.json").read_text(),
                         "topology": (AMD / "topology.json").read_text(),
                         "pci": "0000:83:00.0 0 0-19\n"})
        assert gpus == [{
            "index": 0, "uuid": MI300X_UUID, "pci_bus_id": "0000:83:00.0",
            "memory_mib": 196288, "name": "AMD Instinct MI300X VF", "numa": 0,
            "cpu_affinity": tuple(range(20)), "links": {"GPU0": "X"},
        }]

    def test_an_older_bare_list_static_parses_too(self):
        static = json.loads((AMD / "static.json").read_text())["gpu_data"]
        gpus = amd_gpus({"list": (AMD / "list.json").read_text(), "static": json.dumps(static)})
        assert gpus[0]["uuid"] == MI300X_UUID and gpus[0]["numa"] is None

    def test_garbage_from_a_broken_driver_yields_nothing_not_a_guess(self):
        assert amd_gpus({"list": "amdsmi_cli_exceptions...: error", "static": None}) == []
        assert amd_gpus(None) == []

    def test_the_real_probe_normalizes_to_the_mi300x(self):
        topo = amd_topo()
        assert len(topo.gpus) == 1
        g = topo.gpus[0]
        assert (g.uuid, g.pci_bus_id, g.memory_mib, g.name) == (
            MI300X_UUID, "0000:83:00.0", 196288, "AMD Instinct MI300X VF")
        assert g.numa == 0 and g.numa_source == "sysfs"
        assert g.cpu_affinity == tuple(range(20))
        # A DO droplet is a KVM guest; the probe must say so.
        assert topo.scope == "guest" and topo.confidence == "low"

    def test_an_nvidia_probe_is_not_touched_by_the_amd_branch(self):
        raw = json.loads((FIXTURES / "topology_dev_1gpu.json").read_text())
        assert "gpu_vendor" not in raw
        assert normalize(raw, "n1") == normalize(dict(raw), "n1")


# ── placement + launcher ───────────────────────────────────────────────────

class TestLauncher:
    def test_the_plan_pins_the_rocr_uuid(self):
        plan = amd_plan()
        assert plan.ranks[0].gpu_uuids == (MI300X_UUID,)

    def test_openmpi_sets_rocr_visible_devices_and_rocm(self):
        plan = amd_plan()
        r = OpenMPILauncherAdapter().compile(
            plan, gpu_spec(), node_ips={"fp-mi300x-1": "10.0.0.5"}, workdir="/tmp/w",
            image="/tmp/w/i.sif", gpu_vendors={"fp-mi300x-1": "amd"})
        script = r.files["launcher-rank-0.sh"]
        assert f"export ROCR_VISIBLE_DEVICES={MI300X_UUID}\n" in script
        assert f"ROCR_VISIBLE_DEVICES={MI300X_UUID}" in script.split("\nexec ")[1]
        assert "CUDA_VISIBLE_DEVICES" not in script
        assert "--rocm" in script and "--nv" not in script

    def test_openmpi_without_vendors_is_the_nvidia_rendering(self):
        plan = amd_plan()
        kw = dict(node_ips={"fp-mi300x-1": "10.0.0.5"}, workdir="/tmp/w", image="/tmp/w/i.sif")
        plain = OpenMPILauncherAdapter().compile(plan, gpu_spec(), **kw)
        explicit = OpenMPILauncherAdapter().compile(
            plan, gpu_spec(), gpu_vendors={"fp-mi300x-1": "nvidia"}, **kw)
        assert plain.files == explicit.files and plain.argv == explicit.argv
        assert "export CUDA_VISIBLE_DEVICES=" in plain.files["launcher-rank-0.sh"]
        assert "--nv" in plain.files["launcher-rank-0.sh"]

    def test_single_rank_launcher_sets_rocr_visible_devices(self):
        r = LocalProcessLauncherAdapter().compile(
            amd_plan(), gpu_spec(), node_ips={"fp-mi300x-1": "10.0.0.5"}, workdir="/tmp/w",
            image="/tmp/w/i.sif", gpu_vendors={"fp-mi300x-1": "amd"})
        script = "\n".join(r.files.values()) + " ".join(r.argv)
        assert f"ROCR_VISIBLE_DEVICES={MI300X_UUID}" in script
        assert "CUDA_VISIBLE_DEVICES" not in script
        assert "--rocm" in script and "--nv" not in script


# ── receipt ────────────────────────────────────────────────────────────────

class TestReceipt:
    def observation(self, **over):
        # The real rank-0 observation placement-probe.sh wrote on the droplet
        # under ROCR_VISIBLE_DEVICES=GPU-8ac1449593d1d319. Its allowed CPUs are
        # the whole VM (it was not run under the taskset), so only the GPU half
        # is meaningful here; CPUs are aligned to the plan.
        doc = json.loads((AMD / "placement_rank0_mi300x.json").read_text())
        doc["allowed_cpus"] = "0-7"
        return {**doc, **over}

    def test_an_amd_rank_verifies_against_rocr_visible_devices(self):
        plan = amd_plan()
        receipt = build_receipt(plan, [self.observation()],
                                gpu_vendors={"fp-mi300x-1": "amd"})
        assert receipt.status == "verified", receipt.mismatches
        assert receipt.ranks[0].observed_gpu_uuids == (MI300X_UUID,)
        assert receipt.sources["gpu_visibility_env"] == {"fp-mi300x-1": "ROCR_VISIBLE_DEVICES"}

    def test_the_wrong_rocr_value_is_a_mismatch_naming_the_variable(self):
        plan = amd_plan()
        receipt = build_receipt(plan, [self.observation(rocr_visible_devices="GPU-0")],
                                gpu_vendors={"fp-mi300x-1": "amd"})
        assert receipt.status == "mismatched"
        assert any("ROCR_VISIBLE_DEVICES='GPU-0'" in m for m in receipt.mismatches)

    def test_an_amd_rank_checked_as_nvidia_does_not_falsely_verify(self):
        # Without the vendor the receipt reads CUDA_VISIBLE_DEVICES, which the
        # AMD rank never had: a mismatch, never a silent pass.
        receipt = build_receipt(amd_plan(), [self.observation()])
        assert receipt.status == "mismatched"
        assert "gpu_visibility_env" not in receipt.sources


# ── adapters: the AMD run path and the refusals ────────────────────────────

class TestAdapters:
    def test_require_nvidia_refuses_an_amd_gpu_node(self):
        with pytest.raises(RuntimeError, match="k8s backend cannot run AMD"):
            require_nvidia([amd_node()], "k8s")
        require_nvidia([amd_node(gpu=False)], "k8s")       # a CPU node's vendor is moot
        require_nvidia([NodeRecord(node_id="h", name="h", index=0, ip="", gpu=True)], "k8s")

    def test_mixed_vendor_gpu_nodes_are_refused(self):
        nv = NodeRecord(node_id="nv", name="nv", index=-1, ip="1.1.1.1", gpu=True,
                        provider="static-ssh")
        with pytest.raises(RuntimeError, match="different vendors"):
            job_gpu_vendor([amd_node(), nv], gpu_spec())
        assert job_gpu_vendor([amd_node(), nv], gpu_spec(gpu=False)) == "nvidia"

    def test_static_ssh_runs_an_amd_job_with_rocm(self, monkeypatch):
        sent: list[list[str]] = []

        def fake_run_logged(argv, job_id, **kw):
            sent.append(argv)
            return 0
        monkeypatch.setattr("reconciler.adapter.run_logged", fake_run_logged)
        result = StaticSshAdapter(log=lambda m: None).run(
            [amd_node()], "job-amd", gpu_spec())
        assert result.exit_code == 0
        remote = sent[-1][-1]
        assert "apptainer exec --rocm" in remote and "--nv" not in remote

    def test_a_libvirt_vm_claiming_amd_is_refused(self):
        vm = NodeRecord(node_id="vm", name="vm", index=1, ip="192.168.71.11", gpu=True,
                        gpu_vendor="amd")
        with pytest.raises(RuntimeError, match="libvirt backend cannot run AMD"):
            LibvirtAdapter(log=lambda m: None).run([vm], "job-x", gpu_spec())

    def test_the_local_host_adapter_refuses_amd(self):
        from reconciler.adapter import LocalHostAdapter
        host = NodeRecord(node_id="h", name="h", index=0, ip="", gpu=True, local=True,
                          gpu_vendor="amd")
        with pytest.raises(RuntimeError, match="local-host backend cannot run AMD"):
            LocalHostAdapter.run(LocalHostAdapter.__new__(LocalHostAdapter),
                                 [host], "job-x", gpu_spec())

    def test_bootstrap_checks_for_kfd_on_an_amd_node(self, monkeypatch):
        sent: list[str] = []

        def fake_run_logged(argv, job_id, **kw):
            sent.append(argv[-1])
            return 1 if "kfd" in argv[-1] else 0
        monkeypatch.setattr("reconciler.adapter.run_logged", fake_run_logged)
        with pytest.raises(RuntimeError, match="no /dev/kfd"):
            StaticSshAdapter(log=lambda m: None).bootstrap([amd_node()], "job-x")
        sent.clear()
        nv = NodeRecord(node_id="nv", name="nv", index=-1, ip="1.1.1.1", gpu=True,
                        provider="static-ssh")
        StaticSshAdapter(log=lambda m: None).bootstrap([nv], "job-x")
        assert sent == ["command -v apptainer"]


# ── reconciler: probe vs registration ──────────────────────────────────────

class TestProbeCrossCheck:
    def _reconciler(self, raw):
        from reconciler.reconciler import Reconciler

        class Probe:
            def probe_topology(self, node, job_id):
                return raw
        r = Reconciler.__new__(Reconciler)
        r.adapters = {"static-ssh": Probe()}
        return r

    def test_an_amd_box_registered_as_nvidia_is_refused(self):
        raw = json.loads((FIXTURES / "topology_mi300x_vf_1gpu.json").read_text())
        nv = NodeRecord(node_id="fp", name="fp", index=-1, ip="x", gpu=True,
                        provider="static-ssh")
        with pytest.raises(RuntimeError, match="gpu_vendor: amd"):
            self._reconciler(raw)._probe_topology(nv, "job-x")

    def test_a_matching_registration_normalizes(self):
        raw = json.loads((FIXTURES / "topology_mi300x_vf_1gpu.json").read_text())
        topo = self._reconciler(raw)._probe_topology(amd_node("fp"), "job-x")
        assert topo.gpus[0].uuid == MI300X_UUID


# ── cli add-node ───────────────────────────────────────────────────────────

class TestAddNode:
    def _add(self, tmp_path, monkeypatch, manifest: str, detected: str = ""):
        import argparse
        from reconciler import cli
        from reconciler.store import FileStore
        store = FileStore(str(tmp_path / "state.json"))
        monkeypatch.setattr(cli, "_store", lambda: store)
        monkeypatch.setattr(cli, "_probe_host_memory", lambda *a: 2 ** 30)
        monkeypatch.setattr(cli, "_probe_gpu_vendor", lambda *a: detected)
        spec = tmp_path / "node.yaml"
        spec.write_text(manifest)
        cli.cmd_add_node(argparse.Namespace(spec=str(spec)))
        return store.get_node("fp-mi300x-1")

    def test_manifest_vendor_is_recorded(self, tmp_path, monkeypatch):
        n = self._add(tmp_path, monkeypatch,
                      "name: fp-mi300x-1\nip: 203.0.113.7\ngpu: true\ngpu_vendor: amd\n")
        assert n.gpu_vendor == "amd"

    def test_vendor_is_detected_when_the_manifest_omits_it(self, tmp_path, monkeypatch):
        n = self._add(tmp_path, monkeypatch, "name: fp-mi300x-1\nip: 203.0.113.7\ngpu: true\n",
                      detected="amd")
        assert n.gpu_vendor == "amd"

    def test_nothing_detected_defaults_to_nvidia(self, tmp_path, monkeypatch):
        n = self._add(tmp_path, monkeypatch, "name: fp-mi300x-1\nip: 203.0.113.7\ngpu: true\n")
        assert n.gpu_vendor == "nvidia"

    def test_a_bogus_vendor_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="unknown gpu_vendor"):
            self._add(tmp_path, monkeypatch,
                      "name: fp-mi300x-1\nip: 203.0.113.7\ngpu: true\ngpu_vendor: intel\n")
