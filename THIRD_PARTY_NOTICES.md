# Third-party notices

The environment launcher, setup and SDK verification support are adapted from
Waajacu's MIT-licensed `cuwacunu_dev` and `cuwacunu_embedding` projects. The
original copyright and permission notice is retained in `LICENSE`. See
`doc/PROVENANCE.md` for source identities.

The inherited environment includes LibTorch `2.6.0+cu124`, CUDA 12.4 and cuDNN
components under their upstream licenses. The project MIT license does not
relicense these binaries. The SDK and its notices remain inside the local image
at `/opt/cuwacunu_embedding/libtorch`. Debian package copyright notices remain
under `/usr/share/doc/<package>/copyright`; exact package versions are in
`dependencies.lock`.

Do not put these binaries in Git. Redistribution of an environment image or
other third-party binaries must retain applicable upstream licenses and notices.
