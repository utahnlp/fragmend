# FragMend <img src="image.png" width="50" alt="icon">

 **Repository Under Construction**!   


## Installation
Supported Python Version: 3.13+<br>
To get started with the project, follow the steps mentioned below:
1. Clone the repository to your local working directory.
  ```console
  git clone https://github.com/utahnlp/fragmend.git
  ```
2. Enter the project directory. Create a new virtual environment and activate it.
  ```console
  cd fragmend
  python -m venv <venv_name>
  source activate <venv_name>/bin/activate
  ```
3. Install the project package and requirements
  ```console
  pip install -e .
  ```
4. Check installation
  ```console
  python -c "import xtokenization"
  ```

## Set Up Config File
Open the filepath config file (`src/xtokenization/configs/filepaths.py`) and set the `RESULTS_DATA_HOME` variable to the directory where you want your data, models, and results to be saved.


## Prepare Glot-500c Data
We use fixed-size subsets of the Glot-500c corpus for each language. Prepare the training, tuning, validation and test sets for the unstructured corpus.
```console
  cd src/xtokenization/utils/
  python download_glot500c.sh
```

## Questions?
Please feel free to report an issue if you find any bugs. For quickest reply, email at maitrey.mehta@utah.edu.

---
## Cite Us
```bibtex
@article{mehta2026defragmenting,
  title={Defragmenting Language Models: An Interpretability-based Approach for Vocabulary Expansion},
  author={Mehta, Maitrey and Subramani, Nishant and Xu, Zhichao and Gupta, Ashim and Srikumar, Vivek},
  journal={arXiv preprint arXiv:2604.16656},
  year={2026}
}
```
