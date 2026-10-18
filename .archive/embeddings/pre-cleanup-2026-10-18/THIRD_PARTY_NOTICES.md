# Third-party notices

The representation model was adapted from Waajacu's `cuwacunu_torch` project.
Its MIT copyright and permission notice is retained in `LICENSE` and applies
to the copied and adapted project code.

The locally staged `.external/libtorch` bundle is PyTorch/LibTorch
`2.6.0+cu124`, including its CUDA and cuDNN runtime libraries. Those components
remain under their respective upstream licenses; the project MIT license does
not relicense them. The external bundle is excluded from source control.

Debian compiler and runtime package copyright notices are available inside the
container under `/usr/share/doc/<package>/copyright`. Exact installed dependency
versions are recorded in `dependencies.lock`.

Redistributions that include third-party binaries must also include the
corresponding upstream license texts and notices.
