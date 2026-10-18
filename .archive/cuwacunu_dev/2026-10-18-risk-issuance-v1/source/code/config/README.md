# Configuration module

This standard-library-only module preserves the original project's block syntax:

```text
ENCODER {
  HISTORY_LENGTH = 128;
  EXPORT_WIDTH = 64;
}
```

`cuwacunu::config::Document::read(path)` owns the parsed blocks and remembers the
file location. `parse(text, origin)` also accepts an optional file location.
`block(name)` retrieves a required block; its typed getters read strings, signed
64-bit integers, finite doubles, booleans, and comma-separated integer lists.
Names and keys are case insensitive. The `*_or` getters supply defaults only
when a key is absent; an explicitly empty or malformed value fails.

Consumers must call `reject_unknown_blocks` and `reject_unknown_keys` with their
schema. They own semantic validation such as positive dimensions and compatible
history/patch lengths. `resolve_path` resolves a relative path beside the source
document, never against the launcher's current directory. Relative paths require
an origin.

The imported parser supports `#`, `//`, and `/* ... */` comments. Values are
unquoted; comment tokens are recognized inside every value, including paths.
Do not put literal comment tokens in values or assume quoted strings escape
them. This preserves the original syntax rather than introducing another DSL.

The wrapper rejects unterminated block comments, duplicate blocks and empty
integer-list items and avoids the legacy parser's temporary `single_block()`
storage. It contains no Hero service,
runtime policy, graph, save operation, or backup management. The module and its
tests can be copied unchanged to the `mityane_geometrics` embeddings component
at `C:\Work\Mityane-geometrics\code\embeddings`.

See [IMPORT.md](IMPORT.md) for parser provenance.
