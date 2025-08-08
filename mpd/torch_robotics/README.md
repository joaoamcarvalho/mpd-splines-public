# TorchRobotics

This library implements differentiable robot tree from URDF or MCJF robot format, and the differentiable planning objects such as obstacle avoidance, self-collision avoidance and via point.

## Installation

Simply do

```python
pip install -e .
```

## Examples

For benchmarking on computation time of all available robot kinematics

```azure
python examples/forward_kinematics.py
```

For benchmarking on computation time of distance fields

```azure
python examples/collision_distance.py
```

## Acknowledgements

A part of this implementation is inspired from
the library [differentiable robot model](https://github.com/facebookresearch/differentiable-robot-model).

## Contact

If you have any questions or find any bugs, please
let me know: [An Le](https://www.ias.informatik.tu-darmstadt.de/Team/AnThaiLe), [an@robot-learning.de](an@robot-learning.de)


## Cite TorchRobotics

If you are using TorchRobotics for your scientific publications, please cite it using the [CITATION](CITATION.cff) file,
and the [github help page](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
