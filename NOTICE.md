# Source, data and third-party notice

SafeCapital is an academic research handoff containing project code, supplied
trained models, market and factor data, historical reports and verification
artifacts. Inclusion in this package does not transfer ownership of those
materials or establish a new licence for them. This handoff does not grant a
project-wide open-source licence. Ownership and applicable terms remain with
the respective rights holders; preserve accompanying attribution and terms.

The selected original model source archive is recorded as:
`XGBRanker_Model_Prediction_Evaluation_Handoff_20261021.zip` (English display name for the original source archive; its recorded SHA-256 remains unchanged). The original model snapshots, training
references, configuration, source hashes and historical boundaries are recorded
under `external_validation/ensemble_ranker/`, including
`saved_models_manifest.json`. Other project material retains its accompanying
folder documentation and historical provenance. Release packaging changes do
not establish individual authorship of the underlying team material.

Python and the dependencies named in `requirements*.txt` retain their respective
licences and ownership. These requirements list software needed for execution;
they do not relicense that software. The package's data acquisition references
include Yahoo Finance, SEC EDGAR and FRED; provider data and third-party content
remain subject to their applicable source terms. Raw-source completeness and
historical data limitations are documented in `data/README.md` and
`data/raw_inputs/README.md`.

The supplied frozen inputs, predictions, model weights and reports support the
documented academic review and historical reproduction. This handoff makes no
claim that all included material may be publicly redistributed or commercially
reused without its applicable source permissions. Retain provenance and notices
when sharing material within the intended handoff. Credentials, private account
configuration and newly generated local runtime files do not belong in the
release package.

`RELEASE_MANIFEST.json` records the source archive identity and delivered file
hashes. `python -B tools/check_release.py` compares the listed files with that
inventory without running model, inference, evaluation or investment-simulation
code. Its `passed` result confirms file integrity relative to the supplied
manifest; it does not establish legal ownership, authenticate the manifest, or
certify numerical performance. The original source archive need not be present
inside the extracted project, and its recorded hash is provenance metadata.
