# Source and licence

## Model

- Model: iJB1325 for *Aspergillus niger* ATCC 1015
- File: Additional file 2, SBML Level 3 Version 1
- Original size: 6,676,295 bytes
- SHA-256: `c8f55761d925aa2b532e0b3279d740de29c5cd444ebc1aff0b38d26e007a5ea3`
- [Publisher-hosted SBML](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs40694-018-0060-7/MediaObjects/40694_2018_60_MOESM2_ESM.xml)

The repository stores the exact XML in deterministic gzip form. Decompression
produces the checksum above.

## Citation

Brandl J, Aguilar-Pontes MV, Schäpe P, Nørregaard A, Arvas M, Ram AFJ,
Meyer V, Tsang A, de Vries RP and Andersen MR. [A community-driven
reconstruction of the *Aspergillus niger* metabolic
network](https://doi.org/10.1186/s40694-018-0060-7). *Fungal Biology and
Biotechnology* 5, 16 (2018).

The article is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The publisher states that the associated data use the
[CC0 1.0 waiver](https://creativecommons.org/publicdomain/zero/1.0/) unless
otherwise stated.

## COBRApy compatibility

The original file contains legacy GEMEditor attributes that current libSBML
reports as errors. The runner checks the original hash, removes only four kinds
of `gem:` attributes from an in-memory copy, and adds a model ID. The compressed
source file is never changed.
