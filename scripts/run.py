"""python scripts/run.py --config configs/smoke.yaml [--models stub] [--stages variants,llm,exec,score] [--estimate]"""
from gsi.experiment.run import main

if __name__ == "__main__":
    main()
