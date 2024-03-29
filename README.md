# Visualizing the Impossible
Software for generating animations of the time dependent schrödinger equation using the finite difference method in python, as seen in [Visualizing the Impossible](https://liam-ilan.github.io/time-dependent-schrodinger-equation/).

![](./readme-demo.gif)

## Article
The accompanying article to this repo can be found [here](https://liam-ilan.github.io/time-dependent-schrodinger-equation/).

## About
This repo contains software for generating animations of the TDSE using the finite difference method in python.
This is done by recursively evalutating the following expression:

$$
\begin{aligned}
\psi(x, t + \Delta t) &= \left(\frac{i}{2m} \right) \left(\frac{\Delta t}{{\Delta x}^2} \right) \left( \psi(x + \Delta x, t) - 2 \psi(x, t) + \psi(x - \Delta x, t) \right) \\
&- \left(\frac{i}{\hbar}\right)(\Delta t)V(x)\psi(x,t) \\
&+ \psi(x,t)
\end{aligned}
$$

Derivation of this expression can be found in this repo's accompanying article, [Visualizing the Impossible](https://liam-ilan.github.io/time-dependent-schrodinger-equation/).

This repo and the accompanying article were create for the Summer of Math Exposition 3 (SoME3), read more about the competition at: [https://some.3b1b.co/](https://some.3b1b.co/).

## Render Your Own Animations
First, clone the repo,
```bash
git clone https://github.com/liam-ilan/time-dependent-schrodinger-equation.git
```

The software used to generate these renders was built with Python, Scipy, Numpy, and Matplotlib. To install all necessary packages through pip,
```bash
pip install -r requirements.txt
```

From there, run `main.py` to render all animations and thumbnails for the accompanying article, as well as the demo gif at the top of the readme,
```bash
python main.py
```

### Render Method
`renderer.py` provides a `render()` method that creates a thumbnail and animation of the TDSE.

`render(path, thumb_path, config)`
- `path`: the path to save the animation to
- `thumb_path`: the path to save the thumbnail to
- `config`: a dictionary containing all nescacarry configuration
  - `k`: the wave number of the initial wave packet
  - `std_dev`: the standard deviation of the initial wave packet
  - `x_0`: initial position of wave packet
  - `m`: mass of particle
  - `bounds`: an array of length 2, containing the low (1st item) and high (2nd item) spatial bounds
  - `sample_number`: number of discrete elements to split space into
  - `detla_t`: discrete timestep
  - `real_time`: total simulated time
  - `fps`: fps of animation
  - `anim_duration`: length of animation
  - `pot_func`: a function, that gets a single position, and returns the potential at that position

Example config:
```python
config_free = {
  'k': 6e9,
  'std_dev': 9e-10,
  'x_0': -1.5e-8,
  'm': constants.m_e,
  'bounds': [-2e-8, 2e-8],
  'sample_number': 800,
  'delta_t': 5e-54,
  'real_time': 3e-48,
  'fps': 30,
  'anim_duration': 15,
  'pot_func': lambda x: 0,
}
```

All units are SI units.

## Credit
- Built by [Liam Ilan](https://www.liamilan.com/)
- Check the [accompanying article](https://liam-ilan.github.io/time-dependent-schrodinger-equation/) for a list of references