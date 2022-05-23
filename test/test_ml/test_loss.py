from pathlib import Path
from typing import List
import pandas as pd
import pytest
import torch
from xtbml.ml.evaluation import evaluate

from xtbml.ml.loss import WTMAD2Loss
from xtbml.data.dataset import ReactionDataset, get_gmtkn_dataset
from xtbml.ml.util import load_model_from_cfg

from .gmtkn55 import GMTKN55


class TestWTMAD2Loss:
    """Testing the loss calulation based on GMTKN55 weighting."""

    path = Path(Path(__file__).resolve().parents[2], "data")
    """Absolute path to fit set data."""

    atol = 0.1
    rtol = 0.1

    def setup_class(self):
        print("Test custom loss function")
        self.dataset = get_gmtkn_dataset(self.path)

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
        for r in self.dataset.reactions:
            subset = r.uid.split("_")[0]
            if subset == "BH76RC":
                subset = "BH76"
            partners = [s.split("/")[0] for s in r.partners]
            assert {subset} == set(partners), "Partner and reaction naming inconsistent"

    def test_loading(self):
        """Check consistency of GMTKN55 through by comparing with hard-coded averages and counts (number of reactions) of subsets."""
        atol = 1.0e-2
        rtol = 1.0e-4
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
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

        # total average
        assert torch.allclose(
            self.loss_fn.total_avg,
            torch.tensor(TOTAL_AVG),
            rtol=rtol,
            atol=atol,
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
        output = self.loss_fn(y, y_true, subset, n_partner)

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

        assert output.shape == torch.Size([1])
        assert torch.allclose(output, ref[idx], rtol=self.rtol, atol=self.atol)

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
                    subsets = [s.split("/")[0] for s in batched_reaction.partners]
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

        # print(df)
        # df["dEgfn1"] = (df["Eref"] - df["Egfn1"]).abs()
        # df["dEnn"] = (df["Eref"] - df["Enn"]).abs()
        # print(df[["dEgfn1", "dEnn"]].describe())
        print(
            f"WTMAD-2: Egfn1 = {egfn1[-1]} ; Enn = {enn[-1]}",
        )

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

    @pytest.mark.parametrize("batchsize", [1, 2, 10])
    def stest_gmtkn(self, batchsize):
        # TODO: compare subset and total WTMAD with paper and with implementation in evaluation.py

        print("\nload dl with bs", batchsize)
        dl = self.dataset.get_dataloader({"batch_size": batchsize, "shuffle": True})
        self.loss_fn.reduction = "sum"

        loss = torch.tensor([0.0])
        for _, batched_reaction in dl:

            # derive subset from partner list
            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            loss += self.loss_fn(
                batched_reaction.egfn1, batched_reaction.eref, subsets, n_partner
            )

        assert torch.allclose(
            loss,
            torch.tensor([44728.2891]),
            rtol=1.0e-4,
            atol=1.0e-6,
            equal_nan=False,
        )

    def stest_gmtkn_subsets(self):

        dl = self.dataset.get_dataloader({"batch_size": 2, "shuffle": False})
        self.loss_fn.reduction = "none"

        losses = {k: torch.tensor([0.0]) for k in self.all_sets}

        # calc loss per subset
        for i, (_, batched_reaction) in enumerate(dl):
            # derive subset from partner list
            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            loss = self.loss_fn(
                batched_reaction.egfn1, batched_reaction.eref, subsets, n_partner
            )

            label = [subsets[i] for i in torch.cumsum(n_partner, dim=0) - 1]

            # add each loss to corresponding subset
            for j in range(len(label)):
                losses[label[j]] += loss[j]

        # normalise loss per subset (optional)
        for k, v in losses.items():
            losses[k] = v * len(self.dataset) / self.loss_fn.total_avg

        # print(losses)

        # TODO: compare subset and total WTMAD with paper and with implementation in evaluation.py
        # assert torch.allclose(
        #    loss,
        #    torch.tensor([44055.55078125]),
        #    rtol=1.0e-4,
        #    atol=1.0e-6,
        #    equal_nan=False,
        # )


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
        "W4-11",
        "G21EA",
        "G21IP",
        "DIPCS10",
        "PA26",
        "SIE4x4",
        "ALKBDE10",
        "YBDE18",
        "AL2X6",
        "HEAVYSB11",
        "NBPRC",
        "ALK8",
        "RC21",
        "G2RC",
        "BH76RC",
        "FH51",
        "TAUT15",
        "DC13",
    ]

    reactions = [
        "MB16-43",
        "DARC",
        "RSE43",
        "BSR36",
        "CDIE20",
        "ISO34",
        "ISOL24",
        "C60ISO",
        "PArel",
    ]

    barriers = ["BH76", "BHPERI", "BHDIV10", "INV24", "BHROT27", "PX13", "WCPT18"]

    intra = [
        "IDISP",
        "ICONF",
        "ACONF",
        "Amino20x4",
        "PCONF21",
        "MCONF",
        "SCONF",
        "UPU23",
        "BUT14DIOL",
    ]

    inter = [
        "RG18",
        "ADIM6",
        "S22",
        "S66",
        "HEAVY28",
        "WATER27",
        "CARBHB12",
        "PNICO23",
        "HAL59",
        "AHB21",
        "CHB6",
        "IL16",
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

        # compute average reaction energy for each subset
        avg_subset = ref.abs().mean()

        # pandas' mad is not the MAD we usually use, our MAD is actually MUE/MAE
        # https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/generic.py#L10813
        mue = (ref - target).abs().mean()

        if calc_subsets is True:
            if name in basic:
                basic_wtmad += count * AVG / avg_subset * mue
                basic_count += count
            elif name in reactions:
                reactions_wtmad += count * AVG / avg_subset * mue
                reactions_count += count
            elif name in barriers:
                barriers_wtmad += count * AVG / avg_subset * mue
                barriers_count += count
            elif name in intra:
                intra_wtmad += count * AVG / avg_subset * mue
                intra_count += count
            elif name in inter:
                inter_wtmad += count * AVG / avg_subset * mue
                inter_count += count
            else:
                raise ValueError(f"Subset '{name}' not found in lists.")

        wtmad += count * AVG / avg_subset * mue

        if verbose:
            print(
                f"Subset {name} ({count} entries): MUE {mue:.3f} | count {count} | avg: {avg_subset} | AVG: {AVG}"
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
