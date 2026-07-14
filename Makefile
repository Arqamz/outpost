# Convenience wrapper over infra/libvirt + infra/ansible + the reconciler.
# Run inside the nix dev shell (`nix develop` or `nix-shell`).
LV := infra/libvirt
ANS := infra/ansible

.PHONY: help net-up net-down template cluster-up cluster-down status \
        inventory bootstrap bootstrap-check fail clean \
        seed-nodes submit reconcile jobs mpi-demo mpi-sif submit-mpi dashboard \
        demo-scheduling

help:
	@echo "Cluster targets:"
	@echo "  make net-up        define+start the cluster0 libvirt network"
	@echo "  make template      build the base qcow2 (downloads Ubuntu image)"
	@echo "  make cluster-up    define+start all VMs, wait for SSH, write inventory"
	@echo "  make status        show per-node state / IP / lease / SSH"
	@echo "  make inventory     (re)generate the ansible inventory"
	@echo "  make bootstrap     run the bootstrap role against live nodes"
	@echo "  make bootstrap-check   ansible dry-run (--check)"
	@echo "  make fail N=3      inject a failure on node 3 (default: kill)"
	@echo "  make cluster-down  destroy all VMs (keeps network + template)"
	@echo "  make net-down      undefine the network"
	@echo "  make clean         cluster-down + net-down (template kept)"
	@echo "  --- control plane (dry-run by default) ---"
	@echo "  make seed-nodes    register nodes in the reconciler (VMs + host GPU node)"
	@echo "  make submit        submit job.example.yaml"
	@echo "  make submit-mpi    submit job.mpi.example.yaml (containerized multi-node MPI)"
	@echo "  make reconcile     drive the state machine one tick (--once)"
	@echo "  make jobs          list jobs + states"
	@echo "  make mpi-sif       build demo/mpi_demo.sif (containerized MPI demo)"
	@echo "  make demo-scheduling  submit 4 jobs (contention + wait + GPU) and tick live"
	@echo "  --- log lookup ---"
	@echo "  bin/cluster logs <job-id>     full per-job replay transcript"
	@echo "  bin/cluster reconciler-log    cross-job chronological narration"
	@echo "  make dashboard     live view: node stats + job queue + per-job logs, one page"

net-up:        ; $(LV)/net-up.sh
net-down:      ; $(LV)/net-down.sh
template:      ; $(LV)/build-template.sh
cluster-up:    ; $(LV)/cluster-up.sh
cluster-down:  ; $(LV)/cluster-down.sh
status:        ; $(LV)/cluster-status.sh
inventory:     ; $(LV)/gen-inventory.sh
fail:          ; $(LV)/inject-failure.sh $(N)

bootstrap: inventory
	cd $(ANS) && ansible-playbook site.yml

bootstrap-check: inventory
	cd $(ANS) && ansible-playbook site.yml --check --diff

clean: cluster-down net-down

# --- reconciler (dry-run: NullAdapter, nothing launched) ---
seed-nodes: ; bin/cluster seed-nodes
submit:     ; bin/cluster submit --spec job.example.yaml
submit-mpi: ; bin/cluster submit --spec job.mpi.example.yaml
reconcile:  ; bin/cluster reconcile --once
jobs:       ; bin/cluster list

# --- MPI demo (bare OpenMPI across the VMs; needs cluster-up + bootstrap) ---
mpi-demo:   ; demo/run-mpi-demo.sh

# --- containerized MPI demo (apptainer image for the reconciler's mpi launcher) ---
mpi-sif:    ; apptainer build demo/mpi_demo.sif demo/mpi_demo.def

# --- scheduling demo: 4 jobs (2x pool-filling MPI, 1 waiting MPI, 1 GPU), ticked live ---
demo-scheduling: ; demo/run-scheduling-demo.sh

# --- live dashboard (per-VM CPU/RAM + host GPU + job queue + per-job logs) ---
dashboard:  ; python3 viz/dashboard.py
