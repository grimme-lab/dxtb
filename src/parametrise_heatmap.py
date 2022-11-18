import torch
from torch import nn
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
import json
import ast
import multiprocessing

import gc

from dxtb.typing import Tensor
from dxtb.data.dataset import SampleDataset
from dxtb.data.samples import Sample
from dxtb.xtb.calculator import Calculator
from dxtb.param.gfn1 import GFN1_XTB
from dxtb.param.base import Param
from dxtb.exlibs.ptable_trends import ptable_plotter

from dxtb.optimizer.param_optim import ParameterOptimizer

# NOTE: due to multiprocess setup might require 'ulimit -Sn unlimited' before running script

# calculate each singlepoint in seperate process
manager = multiprocessing.Manager()
results = manager.dict()  # sharable amongst processes


def calc_sp(
    sample: Sample,
    names: list[str],
):
    # NOTE: to be pickled, needs to be at top level

    print(f"Calculating {sample}", flush=True)

    # calc heatmap via singlepoint
    model = ParameterOptimizer(GFN1_XTB.copy(deep=True), names)
    energy = model(sample)

    # calc gradient on parameters
    energy.backward()
    # NOTE: Runtime error can occure if no parameter in names requires grad

    def t2f(tensor: Tensor) -> float | list[float] | None:
        """Convert tensor to float."""
        if tensor == None:
            return None
        return tensor.item() if len(tensor.shape) == 0 else tensor.tolist()

    # write to multiprocess manager
    results[sample.uid] = {
        name: [t2f(p.data), t2f(p.grad)] for name, p in zip(names, model.params)
    }

    return True


class ParameterHeatmap(BaseModel):

    samples: SampleDataset
    """ Samples to be evaluated on. """
    parametrisation: Param
    """ GFN1-xTB parametrisation used to evaluate parameter importance. """
    data: dict = {}
    """ Data containing parameter value and gradient for given samples and parametrisation. """

    class Config:
        arbitrary_types_allowed = True

    def create_heatmap(
        self, names: list[str] = None
    ) -> dict[dict[tuple[float, float]]]:
        """Creating a heatmap for given parameter names. If no names are given, return gradients for all parameters in parametrisation.

        Parameters
        ----------
        names : list[str], optional
            List of parameter names as defined in parametrisation, by default None.

        Returns
        -------
        dict[dict[tuple[float, float]]]
            Heatmap containing sample- and parameter-wise entries of values for given names.
        """
        # NOTE: The optional argument 'names' is purely cosmetic, values and gradients do not change within heatmap

        # parameters to be optimised
        """names = [
            "hamiltonian.xtb.kpair['H-H']",
            "hamiltonian.xtb.kpair['Sc-Sc']",
            "charge.effective.gexp",
            "element['H'].zeff",
            "element['H'].refocc",
            # "element['H'].kcn",
            # "element['H'].shpoly",
            "element['H'].gam",
            "element['C'].kcn",
        ]"""
        # TODO: put those into tutorial

        if names is None:
            # get all numerical parameters
            names = GFN1_XTB.get_all_param_names()
            assert len(names) == 2348  # 2446 - 6 dups - 6 non-numerical - 86 shells

        # process pool with fixed number of processes
        with multiprocessing.Pool(processes=3) as pool:
            args = [(s, names) for s in self.samples.samples]
            v = pool.starmap(calc_sp, args)
            assert all(v)

        self.data = dict(results)

        # convert tensors to floats
        ParameterHeatmap.heatmap2floats(self.data)

        return self.data

    @staticmethod
    def heatmap2floats(d: dict):
        """Convert heatmap entries from tensor to floats.
        Updates in-place

        Parameters
        ----------
        d : dict
            Input heatmap holding tensors
        """
        for i, k in enumerate(d):
            v = d[k] if isinstance(d, dict) else d[i]
            if isinstance(v, dict) or isinstance(v, list):
                ParameterHeatmap.heatmap2floats(v)
            else:
                if torch.is_tensor(v):
                    if isinstance(d, dict):
                        d.update({k: v.item()})
                    else:
                        d[i] = v.item() if len(v.shape) == 0 else v.tolist()

    def get_df(self) -> pd.DataFrame:
        """For better data handling obtain a pandas dataframe

        Returns
        -------
        pd.DataFrame
            Representation of heatmap object
        """

        heatmap = self.data

        # convert to dataframe
        df = pd.DataFrame.from_dict(
            {(i, j): heatmap[i][j] for i in heatmap.keys() for j in heatmap[i].keys()},
            orient="index",
            columns=["value", "grad"],
        )
        df.index = pd.MultiIndex.from_tuples(df.index, names=["sample", "param"])
        df = df.dropna()  # relevant parameters

        return df


def plot_heatmap_element_wise(
    df, sample_ids: list[str] | None = None, param_name="gam3", title=None
):
    print("Plotting ", param_name)
    output_name = "all"

    # filter relevant entries
    if sample_ids != None:
        df = df[df.index.get_level_values("sample").isin(sample_ids)]
        output_name = "_".join(sample_ids).replace("/", "-")
    df = df.drop(["value"], axis=1)
    df = df.filter(like="element[", axis=0)
    df = df.filter(
        like=param_name, axis=0
    )  # TODO: avoid duplicates (e.g. gam und gam3)

    # sum over list values
    if isinstance(next(df.iterrows())[1]["grad"], list):
        df["grad"] = [sum(map(abs, row["grad"])) for _, row in df.iterrows()]

    # restructure for plotting
    df = df.reset_index(level=1)
    df["param"] = df["param"].str.split("[", 1, expand=True)[1]
    df["param"] = df["param"].str.split("]", 1, expand=True)[0]
    df["param"] = df["param"].str.strip("'")

    print(df)

    # agglomerate grad for multiple samples
    df["grad"] = df["grad"].abs()
    df = df.groupby("param")["grad"].sum()
    print(df)

    # TODO: move to temporary output
    df.to_csv("parametrise_heatmap.csv", header=False)

    ptable_plotter(
        "parametrise_heatmap.csv",
        output_filename=f"parametrise_heatmap_{output_name}_{param_name}.html",
        cmap="turbo",
        title=title,
    )


def parametrise_heatmap():

    # load data as batched sample
    path = Path(__file__).resolve().parents[1] / "data" / "PTB"
    list_of_path = sorted(path.glob("samples_*.json"))
    dataset = SampleDataset.from_json(list_of_path)
    # dataset.samples = [dataset.samples[0], dataset.samples[14], dataset.samples[327]]
    dataset.samples = dataset.samples[10000:]
    print("Number samples", len(dataset))

    ph = ParameterHeatmap(samples=dataset, parametrisation=GFN1_XTB.copy(deep=True))
    ph.create_heatmap()
    df = ph.get_df()

    # save to disk
    with open("parametrise_heatmap_PTB_12000.json", "w") as outfile:
        result = df.to_json(orient="index")
        parsed = json.loads(result)
        json.dump(parsed, outfile)

    print("Finished")

    # plot single sample heatmap
    # plot_heatmap_element_wise(df, None, "gam3")

    # TODO:
    #   2. plot heatmap
    #       a. hamiltonian params in bar chart
    #   3. debug elements
    #       c. are further parameters missing in final heatmap? (should be around ~1400 parameters after all)
    #   4. Allow to normalise gradient by number of occurences of each element species (?)

    # TODO: check units of params (bc absolute gradients scale y = m * x --> dy/dx = m! )
    #    e.g. level, kcn in eV
    #   1. setup normalisation script, aka all things in same units (ideally before heatmapping) -- should be standard in toml param?
    #   2. for comparison beyond same units -- statistically bootstrapping required?
    #       a. agglomerate multiple predictions
    #       b. find statistical measure to compare across units

    return


def plot_from_disk():
    """Plot histograms from data on disk. No singlepoint calculation required."""

    def get_df(path):
        df = pd.read_json(path).T
        df.index = pd.MultiIndex.from_tuples(
            [ast.literal_eval(e) for e in df.index.tolist()], names=["sample", "param"]
        )
        return df

    # load json
    path = Path(__file__).resolve().parents[0]
    # df = get_df(path / "parametrise_heatmap_all.json")
    # print(df[df.index.get_level_values("sample").isin(["PTB:HCNO/01"])])
    df = get_df(path / "parametrise_heatmap.json")
    df1 = get_df(path / "parametrise_heatmap_PTB_4000.json")
    df2 = get_df(path / "parametrise_heatmap_PTB_8000.json")
    df3 = get_df(path / "parametrise_heatmap_PTB_10000.json")
    df4 = get_df(path / "parametrise_heatmap_PTB_12000.json")

    df = pd.concat([df1, df2, df3, df4])

    n_samples = len(set([i[0] for i in df.index]))
    print(n_samples)

    """# only HCNO
    for i in df.index:
        if "HCNO" in i[0]:
            df = df.drop(i)
    n_samples = len(set([i[0] for i in df.index]))
    print(n_samples)"""

    # return

    def get_title(name: str) -> str:
        return f"{name} - Accumulated gradient"

    plot_heatmap_element_wise(df, None, ".en", title=get_title("Electronnegativity"))
    # plot all element wise parameter
    plot_heatmap_element_wise(df, None, ".zeff")
    plot_heatmap_element_wise(df, None, ".arep")
    # plot_heatmap_element_wise(df, None, ".en")
    plot_heatmap_element_wise(df, None, ".levels")
    plot_heatmap_element_wise(df, None, ".slater")
    plot_heatmap_element_wise(df, None, ".refocc")
    plot_heatmap_element_wise(df, None, ".kcn")
    plot_heatmap_element_wise(df, None, ".shpoly")
    plot_heatmap_element_wise(df, None, ".gam")  # duplicate with gam3
    plot_heatmap_element_wise(df, None, ".lgam")
    plot_heatmap_element_wise(df, None, ".gam3")


if __name__ == "__main__":
    # parametrise_heatmap()
    plot_from_disk()
