# Decision-model evaluation

This first workflow evaluates Strands Decider 2B on the newest FCC consumer
complaints. The current Mac is Intel, so the deployment script uses CPU
inference.

## Setup

```bash
cd decision-model-eval
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install and download the model (the first run downloads several GB):

```bash
python download_deploy_strands_decider.py --install --download-only
```

Start the local service in a separate terminal:

```bash
python download_deploy_strands_decider.py
```

The service listens on `http://127.0.0.1:8000`.

## Prepare FCC input

```bash
python download_prepare_fcc.py
```

This queries the FCC Socrata API for only the 500 newest records ordered by
`ticket_created DESC`. Outputs are written under `data/` and are ignored by
git. The FCC source is public-domain US government data.

## Run evaluation

With the service running and the data prepared, open
`evaluate_strands_decider.ipynb` in Jupyter and run all cells. Results are
written under `results/`.

The FCC public schema does not provide a free-form complaint narrative. The
prepared state therefore uses non-label structured context (form, method, call
type, and service type). Its `Issue` field is used only to derive the gold
routing label and is not passed to the model, preventing label leakage.
