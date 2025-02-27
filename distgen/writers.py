from .tools import vprint, StopWatch
from .physical_constants import unit_registry

import numpy as np

import os
from collections import OrderedDict as odict
from h5py import File


def writer(output_format, beam, outfile, verbose=0, params=None):
    """Returns a simulaiton code specific writer function"""

    file_writer = {
        "gpt": write_gpt,
        "astra": write_astra,
        "openPMD": write_openPMD,
        "simion": write_simion,
    }

    file_writer[output_format](beam, outfile, verbose, params)


def asci2gdf(gdf_file, txt_file, asci2gdf_bin, remove_txt_file=True):
    """Convert an ASCII GPT file to GDF format"""

    if gdf_file == txt_file:
        os.rename(txt_file, "txt_file.tmp")
        txt_file = "txt_file.tmp"

    result = os.system(f"{asci2gdf_bin} -o {gdf_file} {txt_file}")

    if remove_txt_file:
        os.system(f"rm {txt_file}")

    return result


def write_gpt(beam, outfile, verbose=0, params=None, asci2gdf_bin=None):
    """Writes particles to file in GPT format"""

    watch = StopWatch()

    # Format particles
    gpt_units = {
        "x": "m",
        "y": "m",
        "z": "m",
        "t": "s",
        "gamma_beta_x": "dimensionless",
        "gamma_beta_y": "dimensionless",
        "gamma_beta_z": "dimensionless",
    }

    qspecies = beam.species_charge
    qspecies.ito("coulomb")
    qs = np.full((beam["n_particle"],), 1.0) * qspecies
    qbunch = beam.q.to("coulomb")

    watch.start()

    # assert beam.species == 'electron' # TODO: add more species

    nspecies = np.abs(qbunch.magnitude / qspecies.magnitude)
    nmacro = (
        nspecies * np.abs(beam["w"])
    )  # np.full((beam.n,),1)*np.abs( (beam.q.to("coulomb")).magnitude/beam.n/qspecies.magnitude)

    vprint(
        f'Printing {(beam["n_particle"])} particles to "{outfile}": ',
        verbose > 0,
        0,
        False,
    )

    # Scale parameters to GPT units
    for var in gpt_units:
        beam[var].ito(gpt_units[var])

    headers = odict(
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "gamma_beta_x": "GBx",
            "gamma_beta_y": "GBy",
            "gamma_beta_z": "GBz",
            "t": "t",
            "q": "q",
            "nmacro": "nmacro",
        }
    )

    # Check for spin:
    if hasattr(beam, "sx"):
        headers["sx"], headers["sy"], headers["sz"], headers["g_factor"] = (
            "spinx",
            "spiny",
            "spinz",
            "sping",
        )

    header = "   ".join(headers.values())

    data = np.zeros((len(beam["x"]), len(headers)))
    for index, var in enumerate(headers):
        if var == "q":
            data[:, index] = qs.magnitude
        elif var == "nmacro":
            data[:, index] = nmacro.magnitude
        elif var in ["sx", "sy", "sz"]:
            data[:, index] = (
                beam[var].to("m*eV/c").magnitude
                / np.sqrt(beam["s2"]).to("m*eV/c").magnitude
            )
        elif var == "g_factor":
            data[:, index] = np.abs(beam[var].magnitude)
        else:
            data[:, index] = beam[var].magnitude

    if ".txt" == outfile[-4:]:
        gdffile = outfile[:-4] + ".gdf"
    elif ".gdf" == outfile[-4:]:
        gdffile = outfile
        outfile = outfile + ".txt"
    else:
        gdffile = outfile + ".gdf"

    np.savetxt(outfile, data, header=header, comments="")

    if asci2gdf_bin:
        gdfwatch = StopWatch()
        gdfwatch.start()
        vprint("Converting file to GDF: ", verbose > 0, 1, False)

        try:
            asci2gdf(gdffile, outfile, asci2gdf_bin)
            gdfwatch.stop()

        except Exception as ex:
            print("Error occured while converting ascii to gdf file: ")
            print(str(ex))

        gdfwatch.stop()
        vprint(f"done. Time ellapsed: {gdfwatch.print()}.", verbose > 0, 0, True)

    watch.stop()
    vprint(
        f"...done. Time ellapsed: {watch.print()}.",
        verbose > 0 and asci2gdf_bin,
        0,
        True,
    )
    vprint(
        f"done. Time ellapsed: {watch.print()}.",
        verbose > 0 and not asci2gdf_bin,
        0,
        True,
    )


def write_astra(
    beam, outfile, verbose=False, params=None, species="electron", probe=True
):
    """
    Writes Astra style particles from a beam.

    For now, the species must be electrons.

    If probe, the six standard probe particles will be written.
    """
    watch = StopWatch()
    watch.start()

    vprint(
        f'Printing {(beam["n_particle"])} particles to "{outfile}": ',
        verbose > 0,
        0,
        False,
    )

    assert beam.species == "electron"  # TODO: add more species

    # number of lines in file
    size = beam["n_particle"] + 1  # Allow one for reference particle
    i_start = 1  # Start for data particles
    if probe:
        # Add six probe particles, according to the manual
        size += 6
        i_start += 6

    # macro charge for each particle
    q_macro = beam.q.to("nC").magnitude / beam["n_particle"]

    qs = np.full((beam["n_particle"],), 1.0) * q_macro * unit_registry("nanocoulomb")

    # Astra units and types
    units = ["m", "m", "m", "eV/c", "eV/c", "eV/c", "ns", "nC"]
    names = ["x", "y", "z", "px", "py", "pz", "t", "q", "index", "status"]

    ASTRA_SPECIES_INDEX = {"electron": 1}

    types = 8 * [float] + 2 * [np.int8]
    # Convert to these units in place
    for i in range(8):
        name = names[i]
        if name == "q":
            qs.ito(units[i])
        else:
            beam[name].ito(units[i])

    # Reference particle
    ref_particle = {"q": 0}
    sigma = {}
    for k in ["x", "y", "z", "px", "py", "pz", "t"]:
        ref_particle[k] = np.mean(beam[k]).magnitude
        sigma[k] = np.std(beam[k]).magnitude

    # Make structured array
    dtype = np.dtype(list(zip(names, types)))
    data = np.zeros(size, dtype=dtype)
    for k in ["x", "y", "z", "px", "py", "pz", "t"]:
        data[k][i_start:] = beam[k].magnitude

    # Set these to be the same
    data["q"] = q_macro
    data["index"] = ASTRA_SPECIES_INDEX[beam.species]
    data["status"] = -1  # Particle at cathode

    # Subtract off reference z, pz, t
    for k in ["z", "pz", "t"]:
        data[k] -= ref_particle[k]

    # Put ref particle in first position
    for k in ref_particle:
        data[k][0] = ref_particle[k]

    # Optional: probes, according to the manual
    if probe:
        data[1]["x"] = 0.5 * sigma["x"]
        data[1]["t"] = 0.5 * sigma["t"]
        data[2]["y"] = 0.5 * sigma["y"]
        data[2]["t"] = -0.5 * sigma["t"]
        data[3]["x"] = 1.0 * sigma["x"]
        data[3]["t"] = sigma["t"]
        data[4]["y"] = 1.0 * sigma["y"]
        data[4]["t"] = -sigma["t"]
        data[5]["x"] = 1.5 * sigma["x"]
        data[5]["t"] = 1.5 * sigma["t"]
        data[6]["y"] = 1.5 * sigma["y"]
        data[6]["t"] = -1.5 * sigma["t"]
        data[1:7]["status"] = 3
        data[1:7]["pz"] = 0  # ? This is what the Astra Generator does

    # Save in the 'high_res = T' format
    np.savetxt(outfile, data, fmt=" ".join(8 * ["%20.12e"] + 2 * ["%4i"]))
    watch.stop()
    vprint(f"done. Time ellapsed: {watch.print()}.", verbose > 0, 0, True)


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.bytes_(s)


def write_simion(beam, outfile, verbose=0, params={"color": 0}):
    color = params["color"]

    header = ";0"

    simion_params = [
        "TOB",
        "MASS",
        "CHARGE",
        "X",
        "Y",
        "Z",
        "AZ",
        "EL",
        "KE",
        "CWF",
        "COLOR",
    ]

    simion_units = {
        "TOB": "usec",
        "MASS": "amu",
        "CHARGE": "e",
        "X": "mm",
        "Y": "mm",
        "Z": "mm",
        "AZ": "deg",
        "EL": "deg",
        "CWF": "",
        "COLOR": "",
    }

    N = beam.n_particle

    data = np.zeros((N, len(simion_params)))

    data[:, simion_params.index("TOB")] = beam.t.to(
        "microseconds"
    ).magnitude  # [P.t] = sec, convert to usec

    if beam.species == "electron":
        data[:, simion_params.index("MASS")] = np.full(
            N, beam.species_mass.to("amu").magnitude
        )
        data[:, simion_params.index("CHARGE")] = np.full(N, -1)
    else:
        raise ValueError(f"Species {beam.species} is not supported")

    data[:, simion_params.index("X")] = beam.z.to(simion_units["X"]).magnitude
    data[:, simion_params.index("Y")] = beam.y.to(simion_units["Y"]).magnitude
    data[:, simion_params.index("Z")] = -beam.x.to(simion_units["Z"]).magnitude

    px = beam.pz.to("eV/c").magnitude
    py = beam.py.to("eV/c").magnitude
    pz = -beam.px.to("eV/c").magnitude

    data[:, simion_params.index("KE")] = beam.kinetic_energy.magnitude  # [eV]
    data[:, simion_params.index("AZ")] = np.arctan2(-pz, px) * (180 / np.pi)  # [deg]
    data[:, simion_params.index("EL")] = np.arctan2(py, np.sqrt(px**2 + pz**2)) * (
        180 / np.pi
    )  # [deg]

    # Charge Weighting Factor, derive from particle group weights
    data[:, simion_params.index("CWF")] = (
        beam.q.to("C").magnitude / abs(beam.species_charge.to("C").magnitude) / N
    )
    data[:, simion_params.index("COLOR")] = np.full(N, color)

    np.savetxt(outfile, data, delimiter=",", header=header, comments="", fmt="  %.9e")


def write_openPMD(beam, outfile, verbose=0, params=None):
    with File(outfile, "w") as h5:
        watch = StopWatch()
        watch.start()
        vprint(
            f'Printing {beam["n_particle"]} particles to "{outfile}": ',
            verbose > 0,
            0,
            False,
        )

        opmd_init(h5)
        write_openpmd_h5(beam, h5, name="/data/0/particles/", verbose=0)

        watch.stop()
        vprint(f"done. Time ellapsed: {watch.print()}.", verbose > 0, 0, True)


def opmd_init(h5):
    """
    Root attribute initialization.

    h5 should be the root of the file.
    """
    d = {
        "basePath": "/data/%T/",
        "dataType": "openPMD",
        "openPMD": "2.0.0",
        "openPMDextension": "BeamPhysics;SpeciesType",
        "particlesPath": "particles/",
    }
    for k, v in d.items():
        h5.attrs[k] = np.bytes_(v)
    h5.create_group("/data/")


def write_openpmd_h5(beam, h5, name=None, verbose=0):
    """
    Write particle data at a screen in openPMD BeamPhysics format
    https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md
    """

    if name:
        g = h5.create_group(name)
    else:
        g = h5

    species = beam.species

    n_particle = beam["n_particle"]
    q_total = beam.q.to("C").magnitude

    g.attrs["speciesType"] = fstr(species)
    g.attrs["numParticles"] = n_particle
    g.attrs["chargeLive"] = abs(q_total)
    g.attrs["chargeUnitSI"] = 1
    # g.attrs['chargeUnitDimension']=(0., 0., 1, 1., 0., 0., 0.) # Amp*s = Coulomb
    g.attrs["totalCharge"] = abs(q_total)

    # Position
    g["position/x"] = beam["x"].to("m").magnitude  # in meters
    g["position/y"] = beam["y"].to("m").magnitude
    g["position/z"] = beam["z"].to("m").magnitude
    for component in [
        "position/x",
        "position/y",
        "position/z",
        "position",
    ]:  # Add units to all components
        g[component].attrs["unitSI"] = 1.0
        g[component].attrs["unitDimension"] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # m

    # momenta
    g["momentum/x"] = beam["px"].to("eV/c").magnitude  #  m*c*gamma*beta_x in eV/c
    g["momentum/y"] = beam["py"].to("eV/c").magnitude
    g["momentum/z"] = beam["pz"].to("eV/c").magnitude
    for component in ["momentum/x", "momentum/y", "momentum/z", "momentum"]:
        g[component].attrs["unitSI"] = (
            5.34428594864784788094e-28  # eV/c in J/(m/s) =  kg*m / s
        )
        g[component].attrs["unitDimension"] = (
            1.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )  # kg*m / s

    # Time
    g["time"] = beam["t"].to("s").magnitude
    g["time"].attrs["unitSI"] = 1.0  # s
    g["time"].attrs["unitDimension"] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # s

    # Weights
    # g['weight'] = beam['q'].to('C').magnitude
    g["weight"] = beam["w"].magnitude * abs(q_total)  # should be a charge
    g["weight"].attrs["unitSI"] = 1.0
    g["weight"].attrs["unitDimension"] = (
        0.0,
        0.0,
        1,
        1.0,
        0.0,
        0.0,
        0.0,
    )  # Amp*s = Coulomb

    # Status
    # g['particleStatus'] = astra_data['status']


# def write_astra(beam,outfile,verbose=0,params=None):

#        watch = StopWatch()

# Format particles
#        astra_units={"x":"m", "y":"m", "z":"m","px":"eV/c","py":"eV/c","pz":"eV/c","t":"ns","q":"nC"}

#        qspecies = get_species_charge(beam.species)
#        qspecies.ito("nanocoulomb")

#        watch.start()
#        qs = (np.full( (beam.n,), beam.q.to("nanocoulomb")/beam.n))*np.sign(qspecies)*unit_registry("nanocoulomb")

#        particle_index = 1;
#        particle_status = -1;
#
#        vprint("Printing "+str(beam.n)+" particles to '"+outfile+"': ",verbose>0,0,False)

#        # Scale parameters to ASTRA units
#        ref_particle = {}
#        for var in astra_units:
#            if(var in beam.params.keys()):
#                beam[var].ito(astra_units[var])
#                ref_particle[var]=beam.avg(var)

#        diff_vars = ["t","z","pz"]

#        data = np.zeros( (len(beam["x"]),len(astra_units.keys())+2) )
#        for index, var in enumerate(astra_units.keys()):
#            if(var=="q"):
#                data[:,index] = qs.magnitude
#            else:
#                if(var in diff_vars):
#                    data[:,index] = beam[var].magnitude - ref_particle[var]
#                else:
#                    data[:,index] = beam[var].magnitude

#        ref_particle = [ref_particle["x"].magnitude,
#                        ref_particle["y"].magnitude,
#                        ref_particle["z"].magnitude,
#                        ref_particle["px"].magnitude,
#                        ref_particle["py"].magnitude,
#                        ref_particle["pz"].magnitude,
#                        ref_particle["t"].magnitude,
#                        qs[0].magnitude, 1,-1]


#        data[:,-2]=particle_index
#        data[:,-1]=particle_status
#        data[0,:] = ref_particle

#        np.savetxt(outfile,data)

#        watch.stop()
#        vprint("done. Time ellapsed: "+watch.print()+".",verbose>0,0,True)
