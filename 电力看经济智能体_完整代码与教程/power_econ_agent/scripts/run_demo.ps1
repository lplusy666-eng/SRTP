param(
  [string]$Config = "configs/demo.yaml"
)
$ErrorActionPreference = "Stop"
power-econ demo -c $Config
python scripts/verify_all.py --config $Config
