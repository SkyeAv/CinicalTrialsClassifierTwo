# CTClassifier2 Snapshot Configuration

## Version 0.1.0

### By Skye Goetz

## Snapshot Configuration

Usage (Example)
```yaml
version: 20250720
zip_directory: /local_raid1/sgoetz/STORE/DATASETS/CT_CLASSIFIER2/ZIPFILES
tables:
  - brief_summaries
  - calculated_values
  - designs
  - detailed_descriptions
  - eligibilities
  - responsible_parties
  - studies

gold_labels: /local_raid1/sgoetz/STORE/DATASETS/CT_CLASSIFIER/LABLED_TRIALS/gold_ncts.txt
pseudo_lables: /local_raid1/sgoetz/STORE/DATASETS/CT_CLASSIFIER/LABLED_TRIALS/pred_ncts.txt
save_to: /local_raid1/sgoetz/STORE/MODELS/WEIGHTS/02_SEP_CTML2.pt
```

Options
|Key|Description|Default|
|-|-|-|
|version|AACT Snapshot Version (Date)|NA|
|zip_directory|Directory Where the {version}.zip Is Stored|NA|
|tables|A List of Tables (Text Files in the Zip) to Use For Classification|NA|
|gold_labels|A Text File of Gold (Hand) Labeled Newline Delimited NCT Identifiers Starting with Positive Labels and Delimited by "[negative]"|NA|
|pseudo_lables|A Text File of Pseudo (Snorkel or Alike) Labeled Newline Delimited NCT Identifiers Starting with Positive Labels and Delimited by "[negative]"|NA|
|save_to|A Path to Save the Final PyTorch Weights To|NA|

## Return To Root

[Click Me](../README.md)