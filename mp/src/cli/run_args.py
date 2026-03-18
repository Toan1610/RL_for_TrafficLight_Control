import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Allow running as module: python -m src.cli.run_args
from src.service.runner import SumoRunner


def _read_json(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _default_sumocfg() -> Path | None:
	"""Try to guess a reasonable default sumocfg in the repo."""
	candidates = [
		Path("data/phuquoc/sumo/PhuQuoc_v3/phuquoc.sumocfg"),
		Path("data/phuquoc/sumo/PhuQuoc_v3/test.sumocfg"),
	]
	for p in candidates:
		if p.exists():
			return p
	return None


def build_sumo_cfg(cfg: Dict[str, Any], sumocfg: Path | None, gui: bool | None) -> Dict[str, Any]:
	sumo_cfg = dict(cfg)
	if sumocfg is not None:
		sumo_cfg["sumocfg"] = str(sumocfg)
	else:
		# If config points to a valid .sumocfg file, keep it; else try guess
		cfg_val = sumo_cfg.get("sumocfg")
		keep = False
		if cfg_val:
			p = Path(cfg_val)
			if p.is_file() and p.suffix.lower() == ".sumocfg":
				keep = True
		if not keep:
			guessed = _default_sumocfg()
			if guessed and guessed.is_file():
				sumo_cfg["sumocfg"] = str(guessed)
			else:
				raise FileNotFoundError("Missing or invalid sumocfg in config and no default .sumocfg could be inferred.")

	if gui is not None:
		sumo_cfg["gui"] = bool(gui)

	# Ensure required fields exist with sane defaults
	sumo_cfg.setdefault("runner", "libsumo")
	sumo_cfg.setdefault("begin", 0)
	sumo_cfg.setdefault("end", 3600)
	sumo_cfg.setdefault("step_length", 0.1)
	sumo_cfg.setdefault("sample_interval", 10.0)
	return sumo_cfg


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Run traffic light algorithm with SUMO")
	parser.add_argument("--config", type=Path, default=Path("config/config.json"), help="Path to main config.json")
	parser.add_argument("--sumocfg", type=Path, default=None, help="Path to .sumocfg file (overrides config)")
	parser.add_argument("--net-info", type=Path, default=Path("data/phuquoc/sumo/PhuQuoc_v3/net-info.json"), help="Path to net-info.json")
	parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
	parser.add_argument("--no-gui", action="store_true", help="Force headless mode")
	parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

	args = parser.parse_args(argv)

	# Logging setup
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	logger = logging.getLogger("run")

	# Read configs
	if not args.config.exists():
		raise FileNotFoundError(f"Config not found: {args.config}")
	cfg = _read_json(args.config)

	if not args.net_info.exists():
		raise FileNotFoundError(f"net-info.json not found: {args.net_info}")
	net_info = _read_json(args.net_info)

	# Build SUMO settings
	gui_override = True if args.gui else (False if args.no_gui else None)
	sumo_cfg = build_sumo_cfg(cfg.get("sumo", {}), args.sumocfg, gui_override)

	# Controller plan straight from config
	controller_plan: Dict[str, Any] = cfg.get("controllers", {})
	if not controller_plan:
		raise ValueError("No controllers defined in config")

	# Build adaptive mask from net-info: mark all listed TLS as adaptive
	tls_dict = net_info.get("tls", {})
	if not tls_dict:
		raise ValueError("Field 'tls' missing or empty in net-info.json")
	adaptive_mask: Dict[str, str] = {tls_id: tls_info['controller'] for tls_id,tls_info in tls_dict.items() }
	# print to check adaptive_mask
	print("Adaptive mask:", adaptive_mask)

	# Run
	logger.info("Starting SUMO with %s; GUI=%s", sumo_cfg.get("sumocfg"), sumo_cfg.get("gui"))
	runner = SumoRunner(sumo_cfg=sumo_cfg, controller_plan=controller_plan, net_info=net_info)
	runner.run(adaptive_mask)

	logger.info("Completed run")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

