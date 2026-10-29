# Dataloaders

Hx19 owns this directory. Each data form has its own subdirectory and its own
build. A loader reads source files into windows. It does not train an encoder
or score an operation.

| Form | Path | What it loads |
| --- | --- | --- |
| Numeric series | [numeric_series](numeric_series/README.md) | Time series. One memory-mapped kline cache per resolution |
