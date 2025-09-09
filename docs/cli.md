# CTClassifier2 CLI

## Version 0.1.0

### By Skye Goetz

## install-snapshot
Fetches Features from a Clinical Traials Snapshot and Outputs a Cached Parquet

Usage
```bash
CTC2 install-snapshot --help
```

Options
|Option|Description|Default|
|-|-|-|
|-c|Path to the snapshot config YAML|NA|

## embed-snapshot
Embeds and Auto-Encodes Freetext Features from a Clinical Traials Snapshot and Outputs a Cached Parquet

Usage
```bash
CTC2 embed-snapshot --help
```
Options
|Option|Description|Default|
|-|-|-|
|-c|Path to the snapshot config YAML|NA|

## train-labels
Trains a Trompt Transformer on a Pre-Intialized and Embedded Clinical Traials Snapshot with Labled Trials

Usage
```bash
CTC2 train-labels --help
```

Options
|Option|Description|Default|
|-|-|-|
|-c|Path to the snapshot config YAML|NA|

## Return To Root

[Click Me](../README.md)