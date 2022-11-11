from pathlib import Path
from typing import List
import pandas as pd
import pytest
import torch

from dxtb.ml.loss import WTMAD2Loss
from dxtb.data.dataset import ReactionDataset, get_gmtkn55_dataset
from dxtb.ml.util import load_model_from_cfg

from .gmtkn55 import GMTKN55


class TestWTMAD2Loss:
    """Testing the loss calulation based on GMTKN55 weighting."""

    path = Path(Path(__file__).resolve().parents[2], "data", "GMTKN55")
    """Absolute path to fit set data."""

    atol = 0.1
    """Absolute tolerance for `torch.allclose`"""

    rtol = 0.05
    """
    Relative tolerance for `torch.allclose`. Must be somewhat larger as small reference values are rounded to two digits and then multiplied by the WTMAD-2 scaling factor, which is rather large for ACONF.
    """

    def setup_class(self):
        print("Test custom loss function")
        self.dataset = get_gmtkn55_dataset(
            self.path, file_reactions="reactions.json", file_samples="samples.json"
        )

        # subsets in GMTKN55
        self.all_sets = set([r.uid.split("_")[0] for r in self.dataset.reactions])

        # loss function
        self.loss_fn = WTMAD2Loss(self.path)

    def teardown_class(self):
        # teardown_class called once for the class
        pass

    def setup_method(self):
        # setup_method called for every method
        pass

    def teardown_method(self):
        # teardown_method called for every method
        pass

    def test_data(self):

        # datatype
        assert isinstance(self.dataset, ReactionDataset)

        # number of reactions in GMTKN-55
        assert len(self.dataset) == 1505

        # number of subsets in GMTKN-55
        assert len(self.all_sets) == 55

    def test_naming_consistent(self):
        # reactions
        for r in self.dataset.reactions:
            subset = r.uid.split("_")[0]
            partners = [s.split("/")[0] for s in r.partners]
            if subset == "GMTKN55:BH76RC":
                subset = "GMTKN55:BH76"
            assert {subset} == set(partners), "Partner and reaction naming inconsistent"
            assert r.uid.startswith("GMTKN55:")
            assert all([s.startswith("GMTKN55:") for s in r.partners])
        # samples
        for sample in self.dataset.samples:
            assert sample.uid.startswith("GMTKN55:")
            assert sample.buid == "GMTKN55"

    def test_loading(self):
        """Check consistency of GMTKN55 through by comparing with hard-coded averages and counts (number of reactions) of subsets."""
        TOTAL_AVG = 57.82

        for target, ref in zip(self.loss_fn.subsets.values(), GMTKN55.values()):
            # counts (type and value)
            assert ref["count"].is_integer()
            assert target["count"].item().is_integer()
            assert int(ref["count"]) == int(target["count"])

            # averages
            assert torch.allclose(
                torch.tensor(ref["avg"]),
                target["avg"],
                rtol=1.0e-4,
                atol=1.0e-2,
                equal_nan=False,
            )

        # total average
        assert torch.allclose(
            self.loss_fn.total_avg,
            torch.tensor(TOTAL_AVG),
            rtol=1.0e-4,
            atol=1.0e-2,
            equal_nan=False,
        )

        # total len
        assert len(self.loss_fn.subsets) == 55

    @pytest.mark.parametrize("idx", range(0, 15))
    def test_aconf_single(self, idx: int):
        """Test WTMAD-2 for all samples of ACONF subset. The arguments required for the loss function are directly obtained from the dataset without any batching or padding.

        Parameters
        ----------
        idx : int
            Index for reaction in ACONF. Ranges from 0 to 14.
        """
        samples, reaction = self.dataset[idx]

        y = (torch.sum(samples[0].egfn1 - samples[1].egfn1)).unsqueeze(0)
        y_true = reaction.eref.unsqueeze(0)
        subset = [s.split("/")[0] for s in reaction.partners]
        n_partner = torch.count_nonzero(reaction.nu).unsqueeze(0)

        self.loss_fn.reduction = "none"
        losses = self.loss_fn(y, y_true, subset, n_partner)

        ref = torch.tensor(
            [
                8.53081967,
                9.79464481,
                6.95103825,
                33.17540984,
                9.79464481,
                10.11060109,
                7.8989071,
                19.9052459,
                21.48502732,
                7.26699454,
                29.69989071,
                32.22754098,
                42.65409836,
                30.64775956,
                42.65409836,
            ]
        )

        assert losses.shape == torch.Size([1])
        assert torch.allclose(
            losses, ref[idx], rtol=self.rtol, atol=self.atol
        ), f"ref: {ref[idx]:.3f} vs. test: {losses.item():.3f} ({y.item():.3f}, {y_true.item():.3f})"

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 15])
    def test_aconf_batched(self, batch_size: int):
        size = 15
        num_batches = int(size / batch_size)

        dataset = self.dataset[:size]
        dl = dataset.get_dataloader({"batch_size": batch_size})

        losses = []
        for batched_samples, batched_reaction in dl:
            for i, reactant in enumerate(batched_samples):
                e = torch.unsqueeze(reactant.egfn1, 1)
                e = torch.sum(e, dim=-1)
                x = e * batched_reaction.nu[:, i].reshape(-1, 1)

                if i == 0:
                    reactant_contributions = x
                else:
                    reactant_contributions = torch.cat((reactant_contributions, x), 1)

            y = torch.sum(reactant_contributions, 1)
            y_true = batched_reaction.eref

            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            # get WTMAD-2 for every sample in batch, i.e. no reduction
            self.loss_fn.reduction = "none"
            loss = self.loss_fn(y, y_true, subsets, n_partner)

            losses.append(loss)

        losses = torch.stack(losses, dim=0)

        ref = torch.tensor(
            [
                8.53081967,
                9.79464481,
                6.95103825,
                33.17540984,
                9.79464481,
                10.11060109,
                7.8989071,
                19.9052459,
                21.48502732,
                7.26699454,
                29.69989071,
                32.22754098,
                42.65409836,
                30.64775956,
                42.65409836,
            ]
        ).reshape(num_batches, batch_size)

        assert losses.shape == torch.Size([num_batches, batch_size])
        assert torch.allclose(losses, ref, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("batch_size", [1, 5, 7])
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_gmtkn55_all(self, batch_size: int, shuffle: bool):
        dataset = self.dataset
        num_batches = int(len(dataset) / batch_size)
        dl = dataset.get_dataloader({"batch_size": batch_size, "shuffle": shuffle})

        losses = []
        for _, batched_reaction in dl:
            y = batched_reaction.egfn1
            y_true = batched_reaction.eref

            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            # get WTMAD-2 for every sample in batch, i.e. no reduction
            self.loss_fn.reduction = "none"
            losses.append(self.loss_fn(y, y_true, subsets, n_partner))

        losses = torch.stack(losses, dim=0)

        assert losses.shape == torch.Size([num_batches, batch_size])
        assert torch.allclose(
            torch.sum(losses) / 1505,
            torch.tensor(36.14),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_gmtkn55_subsets(self):
        dtype = torch.float32
        dataset = self.dataset

        batch_size = 1
        dl = dataset.get_dataloader({"batch_size": batch_size})

        subset, egfn1, eref, losses = [], [], [], []
        for _, batched_reaction in dl:
            y = batched_reaction.egfn1
            y_true = batched_reaction.eref

            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            # get WTMAD-2 for every sample in batch, i.e. no reduction
            self.loss_fn.reduction = "none"
            loss = self.loss_fn(y, y_true, subsets, n_partner)

            subset.append(subsets[0])
            egfn1.append(y.item())
            eref.append(y_true.item())
            losses.append(loss.item())

        df = pd.DataFrame(
            list(zip(subset, eref, egfn1, losses)),
            columns=["subset", "eref", "egfn1", "losses"],
        )

        refs = torch.tensor(
            [
                60.25,  # basic
                28.89,  # reactions
                36.01,  # barrieres #32.66
                26.48,  # intra
                15.87,  # inter
                36.14,  # total
            ],
            dtype=dtype,
        )

        # sum up losses and divide by 1505
        assert torch.allclose(
            torch.tensor(df["losses"].sum() / len(df.index), dtype=dtype),
            refs[-1],
            rtol=self.rtol,
            atol=self.atol,
        )

        egfn1 = wtmad2(df, "egfn1", "eref", verbose=False, calc_subsets=True)
        assert torch.allclose(
            torch.tensor(egfn1, dtype=dtype), refs, rtol=1.0e-4, atol=1.0e-0
        )

    def stest_evaluate(self):
        root = Path(__file__).resolve().parents[2]

        # bookkeeping
        bset = [r.uid.split("_")[0] for r in self.dataset.reactions]

        # setup dataloader
        dl = self.dataset.get_dataloader({"batch_size": 1})

        # load model
        cfg_ml = {
            "model_architecture": "Basic_CNN",
            "training_optimizer": "Adam",
            "training_loss_fn": "WTMAD2Loss",
            "training_loss_fn_path": Path(root, "data"),
            "training_lr": 0.01,
            "epochs": 3,
            "model_state_dict": torch.load(f"{root}/models/202205171935_model.pt"),
        }
        model, _, loss_fn, _ = load_model_from_cfg(cfg_ml)
        model.eval()

        # evaluate model
        with torch.no_grad():
            eref, egfn1, enn = [], [], []
            for i, (batched_samples, batched_reaction) in enumerate(dl):

                y = model(batched_samples, batched_reaction)
                y_true = batched_reaction.eref

                eref.append(y_true.item())
                egfn1.append(batched_reaction.egfn1.item())
                enn.append(y.item())

                if cfg_ml["training_loss_fn"] == "WTMAD2Loss":
                    # derive subset from partner list
                    subsets = [
                        s.split("/")[0].split(":")[1] for s in batched_reaction.partners
                    ]
                    # different number of partners per reaction
                    n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

                    loss = loss_fn(y, y_true, subsets, n_partner)

                else:
                    loss = loss_fn(y, y_true)

                print(
                    f"Enn: {enn[i]:.2f} | Eref: {eref[i]:.2f} | Egfn1: {egfn1[i]:.2f} | Samples: {batched_samples} | Loss: {loss}"
                )
                print("\n")

        df = pd.DataFrame(
            list(zip(bset, eref, egfn1, enn)),
            columns=["subset", "Eref", "Egfn1", "Enn"],
        )

        egfn1 = wtmad2(df, "Egfn1", "Eref", verbose=False, calc_subsets=True)
        enn = wtmad2(df, "Enn", "Eref", verbose=False, calc_subsets=True)

        assert egfn1[-1] - 36.14 < 0.1

    @pytest.mark.grad
    def stest_grad(self):
        # NOTE: currently no custom backward() functionality implemented

        n_reactions = 2.0
        n_reactions_i = 2
        # NOTE: requires grad=True and double precision
        input = torch.arange(n_reactions, requires_grad=True, dtype=torch.float64)
        target = torch.arange(n_reactions, requires_grad=True, dtype=torch.float64) + 3
        label = ["ACONF"] * n_reactions_i * int(n_reactions)
        n_partner = torch.tensor([n_reactions_i, n_reactions_i])

        assert torch.autograd.gradcheck(
            WTMAD2Loss(self.path),
            (input, target, label, n_partner),
            raise_exception=True,
        )


def wtmad2(
    df: pd.DataFrame,
    colname_target: str,
    colname_ref: str,
    set_column: str = "subset",
    verbose: bool = True,
    calc_subsets=False,
) -> List[float]:
    """Calculate the weighted total mean absolute deviation, as defined in

    - L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich,A. Najibi, Asim, S. Grimme,
      *Phys. Chem. Chem. Phys.*, **2017**, 19, 48, 32184-32215.
      (`DOI <http://dx.doi.org/10.1039/C7CP04913G>`__)

    Args:
        df (pd.DataFrame): Dataframe containing target and reference energy values.
        colname_target (str): Name of target column.
        colname_ref (str): Name of reference column.
        set_column (str, optional): Name of column defining the subset association. Defaults to "subset".
        verbose (bool, optional): Allows for printout of subset-wise MAD. Defaults to "False".

    Returns:
        List[float]: Weighted total mean absolute error of subsets and whole benchmark.
    """

    AVG = 57.82

    basic = [
        "GMTKN55:W4-11",
        "GMTKN55:G21EA",
        "GMTKN55:G21IP",
        "GMTKN55:DIPCS10",
        "GMTKN55:PA26",
        "GMTKN55:SIE4x4",
        "GMTKN55:ALKBDE10",
        "GMTKN55:YBDE18",
        "GMTKN55:AL2X6",
        "GMTKN55:HEAVYSB11",
        "GMTKN55:NBPRC",
        "GMTKN55:ALK8",
        "GMTKN55:RC21",
        "GMTKN55:G2RC",
        "GMTKN55:BH76RC",
        "GMTKN55:FH51",
        "GMTKN55:TAUT15",
        "GMTKN55:DC13",
    ]

    reactions = [
        "GMTKN55:MB16-43",
        "GMTKN55:DARC",
        "GMTKN55:RSE43",
        "GMTKN55:BSR36",
        "GMTKN55:CDIE20",
        "GMTKN55:ISO34",
        "GMTKN55:ISOL24",
        "GMTKN55:C60ISO",
        "GMTKN55:PArel",
    ]

    barriers = [
        "GMTKN55:BH76",
        "GMTKN55:BHPERI",
        "GMTKN55:BHDIV10",
        "GMTKN55:INV24",
        "GMTKN55:BHROT27",
        "GMTKN55:PX13",
        "GMTKN55:WCPT18",
    ]

    intra = [
        "GMTKN55:IDISP",
        "GMTKN55:ICONF",
        "GMTKN55:ACONF",
        "GMTKN55:Amino20x4",
        "GMTKN55:PCONF21",
        "GMTKN55:MCONF",
        "GMTKN55:SCONF",
        "GMTKN55:UPU23",
        "GMTKN55:BUT14DIOL",
    ]

    inter = [
        "GMTKN55:RG18",
        "GMTKN55:ADIM6",
        "GMTKN55:S22",
        "GMTKN55:S66",
        "GMTKN55:HEAVY28",
        "GMTKN55:WATER27",
        "GMTKN55:CARBHB12",
        "GMTKN55:PNICO23",
        "GMTKN55:HAL59",
        "GMTKN55:AHB21",
        "GMTKN55:CHB6",
        "GMTKN55:IL16",
    ]

    basic_wtmad, reactions_wtmad, barriers_wtmad, intra_wtmad, inter_wtmad = (
        0,
        0,
        0,
        0,
        0,
    )
    basic_count, reactions_count, barriers_count, intra_count, inter_count = (
        0,
        0,
        0,
        0,
        0,
    )

    subsets = df.groupby([set_column])
    subset_names = df[set_column].unique()

    wtmad = 0
    for name in subset_names:

        sdf = subsets.get_group(name)
        ref = sdf[colname_ref]
        target = sdf[colname_target]

        # number of reactions in each subset
        count = target.count()
        # print(name, count)
        # print(target)

        # compute average reaction energy for each subset
        avg_subset = ref.abs().mean()

        # pandas' mad is not the MAD we usually use, our MAD is actually MUE/MAE
        # https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/generic.py#L10813
        mae = (ref - target).abs().mean()

        if calc_subsets is True:
            if name in basic:
                basic_wtmad += count * AVG / avg_subset * mae
                basic_count += count
            elif name in reactions:
                reactions_wtmad += count * AVG / avg_subset * mae
                reactions_count += count
            elif name in barriers:
                barriers_wtmad += count * AVG / avg_subset * mae
                barriers_count += count
            elif name in intra:
                intra_wtmad += count * AVG / avg_subset * mae
                intra_count += count
            elif name in inter:
                inter_wtmad += count * AVG / avg_subset * mae
                inter_count += count
            else:
                raise ValueError(f"Subset '{name}' not found in lists.")

        wtmad += count * AVG / avg_subset * mae

        if verbose:
            print(
                f"Subset {name} ({count} entries): MUE {mae:.3f} | count {count} | avg: {avg_subset} | AVG: {AVG}"
            )

    if calc_subsets is True:
        return [
            basic_wtmad / basic_count,
            reactions_wtmad / reactions_count,
            barriers_wtmad / barriers_count,
            intra_wtmad / intra_count,
            inter_wtmad / inter_count,
            wtmad / len(df.index),
        ]
    else:
        return wtmad / 1505
