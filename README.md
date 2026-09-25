# White hydrogen models

This repository contains three research models:

- [tank/](tank/README.md): a uniform, zero-dimensional hydrogen tank.
- [darts/](darts/README.md): an intermediate three-dimensional gas–water DARTS model.
- [Bourakebougou/](Bourakebougou/README.md): a Bourakébougou-inspired compositional DARTS model, including its configuration, field evidence, scientific notes, tests, notebook and outputs.

The shared [research report](minor%20formation%20of%20hydrogen.docx) and
[3D plotting/VTK helper](visualize_3d.py) remain at the repository root.

## Run

From this repository's root, use the existing native DARTS environment:

```bash
export DARTS_PY="/Users/sanderbertdacosta/.local/share/python-envs/darts-py311/bin/python"
"$DARTS_PY" Bourakebougou/run.py
"$DARTS_PY" Bourakebougou/run.py --suite
"$DARTS_PY" darts/run.py --suite
"$DARTS_PY" tank/run_tank.py
```

Each model writes to its own `output/` directory. Select the `darts-local` kernel
when opening the notebooks. This native environment uses Python 3.11.16,
Open-DARTS 2.0.0, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2 and IAPWS 1.5.5;
the installed Open-DARTS/DARTS-flash build is required for the reservoir models.
The environment and its native sources live outside this repository, under
`~/.local/share/python-envs/darts-py311` and `~/.local/share/darts-workspace`.
No separate project environment is needed on this machine.

## Check

```bash
"$DARTS_PY" -m unittest tank.test_tank_model darts.test_properties darts.test_model_3d Bourakebougou.test_properties -v
"$DARTS_PY" Bourakebougou/validate.py
"$DARTS_PY" darts/validate.py
```

The validators check numerical conservation and refinement using temporary
simulation folders, then save a compact report in the corresponding model's
`output/validation.json`. These checks do not establish field calibration.
