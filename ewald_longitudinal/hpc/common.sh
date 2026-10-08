# Sourced by the Slurm scripts and submit.sh: load hpc/cluster.env and refuse placeholders.
toy_load_env() {
    local env_file="${1:?env file}"
    [ -f "$env_file" ] || { echo "ERROR: $env_file not found (copy hpc/cluster.env.example)" >&2; return 1; }
    # shellcheck disable=SC1090
    source "$env_file"
    local v missing=()
    for v in SLURM_ACCOUNT SLURM_PARTITION REPO_DIR RUN_ROOT LOG_ROOT PY_SETUP PYTHON; do
        if [ -z "${!v:-}" ] || [[ "${!v}" == *FILL_ME* ]]; then missing+=("$v"); fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: fill these in $env_file: ${missing[*]}" >&2
        return 1
    fi
    [ -d "$REPO_DIR/ewald_longitudinal" ] || { echo "ERROR: $REPO_DIR/ewald_longitudinal not found" >&2; return 1; }
}

toy_threads() {
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
}
