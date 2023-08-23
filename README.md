# ANA2B
Repository for the ANA2B publication (([arXiv](https://arxiv.org/abs/2308.08984))). 

## Reproduction / Examples

Weights for the models necessary to reproduce results can be found in source/weights_models/

Necessary datasets (need to be downloaded separately):

For Training:
- DES5M: https://www.nature.com/articles/s41597-021-00833-x
- AlphaML: https://www.nature.com/articles/s41597-019-0157-8
- Intramolecular Gradients/Multipoles (PBE0/def2-TZVP): https://www.research-collection.ethz.ch/handle/20.500.11850/626683
- 
For Validation and Testing:
- BioFragmentDB: http://vergil.chemistry.gatech.edu/active_bfdb/bfdb/cgi-bin/bfdb.py
- S7L: https://www.nature.com/articles/s41467-021-24119-3
- ICE13: https://pubs.aip.org/aip/jcp/article/157/13/134701/2841942/
- X23: https://pubs.rsc.org/en/content/articlelanding/2019/cp/c9cp04488d
- Structures 6th Blindtest: https://www.science.org/doi/10.1126/sciadv.aau3338
- Structures 5th Blindtest: https://journals.iucr.org/b/issues/2011/06/00/bk5106/index.html
- Structures 4th Blindtest: https://journals.iucr.org/b/issues/2009/02/00/bk5081/index.html
- Structures 3rd Blindtest: https://journals.iucr.org/b/issues/2005/05/00/de5014/index.html
- Structures 2nd Blindtest: https://onlinelibrary.wiley.com/iucr/doi/10.1107/S0108768102005669
- Structures 1st Blindtest: https://onlinelibrary.wiley.com/iucr/doi/10.1107/S0108768100004584

Examples and scripts necessary to reproduce results can be found in 
- MD: source/anaff/md 
- CSP: source/anaff/crystal_ranking
- Dimers: source/results
- ANA2B model training: source/anaff/train_model_alpha_gnn.py
- A small example for general usage: source/examples/example.ipynb

## Citation

```
@article{ANA2b
      title={Hybrid Classical/Machine-Learning Force Fields for the Accurate Description of Molecular Condensed-Phase Systems}, 
      author={Moritz Thürlemann and Sereina Riniker},
      year={2023},
      journal={arXiv:2308.08984},
}
```

## License

MIT
