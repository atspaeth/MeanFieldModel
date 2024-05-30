# Final Figures for "Model-agnostic Neural Mean-field with the Refractory SoftPlus Transfer Function"

This repository contains the code for our manuscript titled **"Model-agnostic Neural Mean-field with the Refractory SoftPlus transfer function"**.

This code depends on the common libraries `matplotlib`, `numpy`, `joblib`, and `tqdm`, as well as our lab's utility library `braingeneerspy`.
All of these can be easily installed via `pip`.
However, simulations are carried out using the NEST Simulator, which can only be installed via other package managers (see [their docs](https://nest-simulator.readthedocs.io/en/stable/installation/index.html)).


## Usage

You can run the main script as follows:

```bash
python Final.py
```

The figures will then be saved in the working directory with descriptive filenames. Note that some of the longer simulations take an hour or more to run!

The code attempts to cache simulation results to our lab's S3 bucket, so if you are trying to run it from outside of Braingeneers, you will need to change all the `@memoize` decorators in `mfsupport.py` to `@memoize(backend="local")`. This way, all results are cached in `./joblib/`, which should work as long as the dependencies are installed correctly.


## Organization

The file `mfsupport.py` defines the methods used, but it is the script `Final.py` which runs the simulations and generates the figures.
The script is organized into sections with the comment marker `# %%`, each generating a specific figure.
Comments within the script provide detailed explanations of the steps involved in generating each figure.
You can reach out via GitHub issues with any questions.
