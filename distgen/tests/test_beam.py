#!/usr/bin/env python
import pathlib
import numpy as np
import pytest

from distgen.beam import Beam
from .. import Generator
from ..physical_constants import PHYSICAL_CONSTANTS
from ..tools import check_abs_and_rel_tols

from .conftest import EXAMPLES_DATA_PATH

coordinates = {
    "x",
    "y",
    "z",
    "px",
    "py",
    "pz",
    "r",
    "theta",
    "pr",
    "ptheta",
    "xp",
    "yp",
    "thetax",
    "thetay",
    "gamma",
    "energy",
    "kinetic_energy",
    "beta_x",
    "beta_y",
    "beta_z",
}

yaml_files = list(EXAMPLES_DATA_PATH.glob("*.yaml"))


@pytest.fixture(params=yaml_files, ids=[fn.name for fn in yaml_files], scope="module")
def yaml_file(request: pytest.FixtureRequest) -> pathlib.Path:
    return request.param


@pytest.fixture(scope="module")
def beam(yaml_file: pathlib.Path) -> Beam:
    if yaml_file.name == "dcm.image.in.yaml":
        from ..tools import dicom

        if dicom is None:
            pytest.skip("pydicom unavailable")

    G = Generator(str(yaml_file), verbose=0)
    G["n_particle"] = 10_000
    return G.beam()


def run_test_on_input_file(input_file, test):
    G = Generator(input_file, verbose=0)
    G["n_particle"] = 10_000
    test(G.beam())


# # Statistical Tests


def test_weight_normalization(beam):
    # $\sum_i w_i = 1$
    check_abs_and_rel_tols(
        "macroparticle weights", np.sum(beam["w"]), 1.0, abs_tol=1e-12, rel_tol=1e-15
    )


def test_avg(beam):
    # $\langle \mathcal{O}\rangle = \sum_i w_i \mathcal{O}_i$

    for var in coordinates:
        avg_beam, avg_numpy = beam.avg(var), np.sum(beam["w"] * beam[var])
        check_abs_and_rel_tols(
            "beam.avg", avg_beam, avg_numpy, abs_tol=1e-12, rel_tol=1e-15
        )


def test_std(beam):
    # $\sigma_{\mathcal{O}}^2 = \sum_i w_i (\mathcal{O}_i-\langle \mathcal{O}\rangle)^2 $

    for var in coordinates:
        sigma2_beam = beam.std(var) ** 2
        sigma2_numpy = np.sum(beam["w"] * (beam[var] - beam.avg(var)) ** 2)
        check_abs_and_rel_tols(
            "beam.std", sigma2_beam, sigma2_numpy, abs_tol=1e-8, rel_tol=1e-15
        )


# # Cylindrical Coordinates
# ---
# # Getting


def test_r(beam):
    # $r=\sqrt{ x^2 + y^2 }$

    check_abs_and_rel_tols(
        "beam.r",
        beam["r"],
        np.sqrt(beam["x"] ** 2 + beam["y"] ** 2),
        abs_tol=1e-15,
        rel_tol=1e-15,
    )


def test_x(beam):
    # $x=r\cos\theta$

    check_abs_and_rel_tols(
        "beam.x = r cos(theta)",
        beam["x"],
        beam["r"] * np.cos(beam["theta"]),
        abs_tol=1e-12,
        rel_tol=1e-11,
    )


def test_y(beam):
    # $y = r\sin\theta$

    check_abs_and_rel_tols(
        "beam.y = r sin(theta)",
        beam["y"],
        beam["r"] * np.sin(beam["theta"]),
        abs_tol=1e-12,
        rel_tol=1e-11,
    )


def test_pr(beam):
    # $p_r = p_x\cos\theta + p_y\sin\theta$

    check_abs_and_rel_tols(
        "beam.pr = px cos(theta) + py sin(theta)",
        beam["pr"],
        beam["px"] * np.cos(beam["theta"]) + beam["py"] * np.sin(beam["theta"]),
        abs_tol=1e-12,
        rel_tol=1e-15,
    )


def test_ptheta(beam):
    # $p_{\theta} = -p_x\sin\theta + p_y\cos\theta$
    check_abs_and_rel_tols(
        "beam.ptheta = -px sin(theta) + py cos(theta)",
        beam["ptheta"],
        -beam["px"] * np.sin(beam["theta"]) + beam["py"] * np.cos(beam["theta"]),
        abs_tol=1e-12,
        rel_tol=1e-15,
    )


# # Transverse Derivatives and Angles


def test_xp(beam):
    # $x^{\prime} = p_x/p_z$

    check_abs_and_rel_tols(
        "beam.xp = px/pz",
        beam["xp"],
        beam["px"].to(beam["pz"].units) / beam["pz"],
        abs_tol=1e-14,
        rel_tol=1e-15,
    )


def test_yp(beam):
    # $y^{\prime} = p_y/p_z$
    check_abs_and_rel_tols(
        "beam.yp = py/pz",
        beam["yp"],
        beam["py"].to(beam["pz"].units) / beam["pz"],
        abs_tol=1e-14,
        rel_tol=1e-15,
    )


def test_thetax(beam):
    # $\theta_x = \arctan(p_x/p_z)$

    check_abs_and_rel_tols(
        "beam.thetax = arctan(px/pz)",
        beam["thetax"],
        np.arctan2(beam["px"].to(beam["pz"].units), beam["pz"]),
        abs_tol=1e-15,
        rel_tol=1e-15,
    )


def test_thetay(beam):
    # $\theta_y = \arctan(p_y/p_z)$

    check_abs_and_rel_tols(
        "beam.thetay = arctan(py/pz)",
        beam["thetay"],
        np.arctan2(beam["py"].to(beam["pz"].units), beam["pz"]),
        abs_tol=1e-15,
        rel_tol=1e-15,
    )


# # Relativistic Quantities
# ---


def test_p(beam):
    # $p=\sqrt{p_x^2 + p_y^2 + p_z^2}$

    deviation = np.abs(
        beam["p"] - np.sqrt(beam["px"] ** 2 + beam["py"] ** 2 + beam["pz"] ** 2)
    )
    check_abs_and_rel_tols(
        "beam.p = sqrt(px^2 + py^2 + pz^2)",
        beam["p"],
        np.sqrt(beam["px"] ** 2 + beam["py"] ** 2 + beam["pz"] ** 2),
        abs_tol=1e-15,
        rel_tol=1e-15,
    )


def test_energy(beam):
    # $E = \sqrt{c^2|\vec{p}|^2 + (mc^2)^2}$

    c = PHYSICAL_CONSTANTS["speed of light in vacuum"]

    check_abs_and_rel_tols(
        "beam.energy = sqrt(c^2p^2 + (mc^2)^2)",
        beam["energy"],
        np.sqrt(c**2 * beam["p"] ** 2 + beam.mc2**2),
        abs_tol=1e-9,
        rel_tol=1e-15,
    )


def test_gamma(beam):
    # $\gamma = \sqrt{1+\left(\frac{p}{mc}\right)^2}$, $E/mc^2$

    mc = beam.species_mass * PHYSICAL_CONSTANTS["speed of light in vacuum"]

    check_abs_and_rel_tols(
        "beam.gamma = sqrt( 1 + (p/mc)^2 )",
        beam["gamma"],
        np.sqrt(1 + (beam["p"] / mc).to_reduced_units() ** 2),
        abs_tol=1e-10,
        rel_tol=1e-10,
    )

    check_abs_and_rel_tols(
        "beam.gamma = E/mc^2",
        beam["gamma"],
        beam["energy"] / beam.mc2,
        abs_tol=1e-15,
        rel_tol=1e-15,
    )


def test_beta(beam):
    # $\beta = \frac{c|\vec{p}|}{E}$

    check_abs_and_rel_tols(
        "beam.beta = c|p|/E",
        beam["beta"],
        PHYSICAL_CONSTANTS["speed of light in vacuum"] * beam.p / beam.energy,
        abs_tol=1e-11,
        rel_tol=1e-14,
    )

    assert max(beam["beta"]) < 1, "max(beta) > 1, faster than light particle!"


def test_beta_xi(beam):
    # $\beta_{x_i} = \frac{cp_{x_i}}{E}$, $\beta_x = x^{\prime}\beta_z$, $\beta_y = y^{\prime}\beta_z$,  $\beta_z = \frac{\beta}{\sqrt{1+(x^{\prime})^2 +(y^{\prime})^2}}$

    for var in ["x", "y", "z"]:
        check_abs_and_rel_tols(
            "beam.beta_xi = c pxi/E )",
            beam[f"beta_{var}"],
            (
                PHYSICAL_CONSTANTS["speed of light in vacuum"]
                * beam[f"p{var}"]
                / beam["energy"]
            ).to_reduced_units(),
            abs_tol=1e-15,
            rel_tol=1e-15,
        )

        check_abs_and_rel_tols(
            "beam.beta_z = sign(pz)*beta/sqrt( 1 + x'^2 + y'^2 )",
            beam["beta_z"],
            np.sign(beam["pz"])
            * beam["beta"]
            / np.sqrt(1 + beam["xp"] ** 2 + beam["yp"] ** 2),
            abs_tol=1e-11,
            rel_tol=5e-9,
        )


def test_kinetic_energy(beam):
    # KE = $mc^2(\gamma-1)$, $E-mc^2$

    if PHYSICAL_CONSTANTS.species(beam["species"])["mass"] > 0:
        check_abs_and_rel_tols(
            "beam.kinetic_energy = mc2*(gamma-1)",
            beam["kinetic_energy"],
            beam.mc2 * (beam["gamma"] - 1),
            abs_tol=1e-9,
            rel_tol=1e-05,
        )

    check_abs_and_rel_tols(
        "beam.kinetic_energy = E - mc2",
        beam["kinetic_energy"],
        beam["energy"] - beam.mc2,
        abs_tol=1e-15,
        rel_tol=1e-15,
    )


# # Twiss Parameters
# ---
# # Getting
#


def test_emitt_normalized(beam):
    # $\epsilon_{n,x_i} = \frac{1}{mc}\sqrt{\sigma_{x_i}^2\sigma_{p_{x_i}}^2 - \langle \left(x_i-\langle x_i\rangle\right)\left(p_{x_i}-\langle p_{x_i}\rangle\right)\rangle^2 }$

    for var in ["x", "y"]:
        mc = beam.species_mass * PHYSICAL_CONSTANTS["speed of light in vacuum"]

        stdx = beam.std(var)
        stdp = (beam.std(f"p{var}") / mc).to_reduced_units()
        dx = beam[var] - beam.avg(var)
        dp = ((beam[f"p{var}"] - beam.avg(f"p{var}")) / mc).to_reduced_units()

        check_abs_and_rel_tols(
            "beam.emitt (normalized)",
            beam.emitt(var),
            np.sqrt(stdx**2 * stdp**2 - (np.sum(beam["w"] * dx * dp)) ** 2),
            abs_tol=1e-11,
            rel_tol=1e-11,
        )


def test_emitt_geometric(beam):
    # $\epsilon_{x} = \sqrt{\sigma_x^2\sigma_{x^{\prime}}^2 - \langle \left(x-\langle x\rangle\right)\left(x^{\prime}-\langle x^{\prime}\rangle\right)\rangle^2 }$

    for var in ["x", "y"]:
        stdx = beam.std(var)
        stdp = beam.std(f"{var}p")
        dx = beam[var] - beam.avg(var)
        dp = beam[f"{var}p"] - beam.avg(f"{var}p")

        check_abs_and_rel_tols(
            "beam.emitt (geometric)",
            beam.emitt(var, "geometric"),
            np.sqrt(stdx**2 * stdp**2 - (np.sum(beam["w"] * dx * dp)) ** 2),
            abs_tol=1e-14,
            rel_tol=1e-15,
        )


def twiss_beta_xi(beam):
    # Twiss $\beta_{x_i} = \frac{\sigma_x^2}{\epsilon_x}$

    for var in ["x", "y"]:
        stdx = beam.std(var)
        epsx = beam.emitt(var, "geometric")

        if epsx > 0:
            check_abs_and_rel_tols(
                "beam.Beta_xi (Twiss)",
                beam.Beta(var),
                stdx**2 / epsx,
                abs_tol=1e-14,
                rel_tol=1e-15,
            )


def test_alpha_xi(beam):
    # Twiss $\alpha_{x_i} = -\frac{\langle(x-\langle x\rangle)(x^{\prime}-\langle x^{\prime}\rangle)\rangle}{\epsilon_x}$

    for var in ["x", "y"]:
        dx = beam[var] - beam.avg(var)
        dp = beam[f"{var}p"] - beam.avg(f"{var}p")
        epsx = beam.emitt(var, "geometric")

        if epsx > 0:
            check_abs_and_rel_tols(
                "beam.Alpha_xi (Twiss)",
                beam.Alpha(var),
                -sum(beam["w"] * dx * dp) / epsx,
                abs_tol=1e-14,
                rel_tol=1e-14,
            )


def test_gamma_xi(beam):
    # Twiss $\gamma_{x_i} = \frac{\sigma_{x^{\prime}}^2}{\epsilon_x}$
    for var in ["x", "y"]:
        stdp = beam.std(f"{var}p")
        epsx = beam.emitt(var, "geometric")

        if epsx > 0:
            check_abs_and_rel_tols(
                "beam.Gamma_xi (Twiss)",
                beam.Gamma(var),
                stdp**2 / epsx,
                abs_tol=1e-14,
                rel_tol=1e-14,
            )
