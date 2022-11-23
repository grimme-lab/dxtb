"""
Data for testing Coulomb contribution.
"""

import torch

from dxtb.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    q: Tensor
    """Atomic or shell-resolved partial charges for this structure."""

    es2: Tensor
    """Reference values for ES2 (GFN1-xTB)"""

    es3: Tensor
    """Reference values for ES3 (GFN1-xTB)"""

    grad: Tensor
    """Nuclear gradient of ES2 energy."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "MB16_43_01": {
        "es2": torch.tensor(0.10952019883948200, dtype=torch.float64),
        "es3": torch.tensor(0.0212785489857197, dtype=torch.float64),
        "q": torch.tensor(
            [
                7.73347900345264e-1,
                1.07626888948184e-1,
                -3.66999593831010e-1,
                4.92833325937897e-2,
                -1.83332156197733e-1,
                2.33302086605469e-1,
                6.61837152062315e-2,
                -5.43944165050002e-1,
                -2.70264356583716e-1,
                2.66618968841682e-1,
                2.62725033202480e-1,
                -7.15315510172571e-2,
                -3.73300777019193e-1,
                3.84585237785621e-2,
                -5.05851088366940e-1,
                5.17677238544189e-1,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor(
            [
                [
                    +1.9963537059076168e-03,
                    -1.3072889261366563e-03,
                    -1.0232559820753207e-02,
                ],
                [
                    +3.6092678330843625e-04,
                    -6.2351757566852037e-05,
                    -2.2295717795003727e-04,
                ],
                [
                    -3.2658061892964097e-03,
                    +1.3446197903724453e-02,
                    +6.4026973020020811e-03,
                ],
                [
                    +4.2425027255962877e-04,
                    -2.6720719955604816e-04,
                    -4.6684980747799232e-04,
                ],
                [
                    -2.2633967708585580e-03,
                    +2.6307737303351559e-03,
                    +3.6429585951752879e-03,
                ],
                [
                    -3.2546156358640951e-03,
                    -3.7955603487672589e-03,
                    -9.0696291739107682e-04,
                ],
                [
                    +1.9382839571159148e-04,
                    +1.1070066775037369e-04,
                    +4.7073952934522593e-04,
                ],
                [
                    -1.5043706715961339e-02,
                    -6.3493250125166122e-03,
                    +1.0770071133291625e-02,
                ],
                [
                    -1.1097670164288385e-03,
                    -1.2272942113687310e-03,
                    +4.5941370590439424e-03,
                ],
                [
                    +1.7627673632876689e-02,
                    +5.5619439714634528e-03,
                    -6.7326082143218911e-03,
                ],
                [
                    -1.2320250611399133e-03,
                    -3.6628366329912107e-03,
                    +3.8160223228411254e-03,
                ],
                [
                    +9.1112980947616678e-05,
                    +5.9289345875293054e-05,
                    +3.3658798144100492e-04,
                ],
                [
                    -4.8906372095381453e-04,
                    -8.3465679858310830e-05,
                    +1.4763157511530028e-04,
                ],
                [
                    +2.0946428546563560e-03,
                    -1.1732112097738419e-02,
                    -1.5219216432091995e-03,
                ],
                [
                    +3.7663598210658789e-03,
                    +7.6094954110348736e-03,
                    -2.8271791380180184e-03,
                ],
                [
                    +1.0323266346915980e-04,
                    -9.3095916368352537e-04,
                    -7.2698067791341636e-03,
                ],
            ],
            dtype=torch.float64,
        ),
    },
    "MB16_43_02": {
        "es2": torch.tensor(0.16666303125798329, dtype=torch.float64),
        "es3": torch.tensor(0.0155668621882796, dtype=torch.float64),
        "q": torch.tensor(
            [
                7.38394711236234e-2,
                -1.68354976558608e-1,
                -3.47642833746823e-1,
                -7.05489267186003e-1,
                7.73548301641266e-1,
                2.30207581365386e-1,
                1.02748501676354e-1,
                9.47818107467040e-2,
                2.44260351729187e-2,
                2.34984927037408e-1,
                -3.17839896393030e-1,
                6.67112994818879e-1,
                -4.78119977010488e-1,
                6.57536027459275e-2,
                1.08259054549882e-1,
                -3.58215329983396e-1,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor([], dtype=torch.float64),
    },
    "MB16_43_07": {
        "es2": torch.tensor(0.12017418620257683, dtype=torch.float64),
        "es3": torch.tensor(0.0, dtype=torch.float64),
        "q": torch.tensor(  # shell-resolved charges
            [
                +8.8596022129058838e-01,
                -1.0356724262237549e00,
                +2.3449918627738953e-01,
                -2.1833347156643867e-02,
                +1.0902611017227173e00,
                -7.5428396463394165e-01,
                +4.1274033486843109e-02,
                -9.6002155914902687e-03,
                +5.1767293363809586e-02,
                -1.0523837991058826e-02,
                +5.9433255344629288e-02,
                -3.9489799737930298e-01,
                +1.4450673013925552e-02,
                +1.5787012875080109e-01,
                -4.6440556645393372e-01,
                +4.7812232375144958e-01,
                -1.0143736600875854e00,
                +9.1033732891082764e-01,
                -4.6157899498939514e-01,
                +9.0761981904506683e-02,
                -3.0731001868844032e-02,
                +1.1395587772130966e-01,
                -3.9991357922554016e-01,
                +1.0487200692296028e-02,
                +4.1295102238655090e-01,
                -5.2687402814626694e-02,
                +4.0443599224090576e-01,
                -2.7010707184672356e-02,
                +3.1367531418800354e-01,
                -9.4423663616180420e-01,
                +1.7532956600189209e-01,
                -4.2600473761558533e-01,
                +1.2486056089401245e00,
                -6.4642405509948730e-01,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor(
            [
                [
                    -1.1158689420468295e-04,
                    -9.4325224931291287e-05,
                    -2.4541370290146619e-04,
                ],
                [
                    -1.5719738198544373e-03,
                    -2.3977816668625824e-03,
                    +6.1339193923076688e-03,
                ],
                [
                    -1.6042385761491783e-03,
                    -2.1958082700178493e-03,
                    -1.6747812742226708e-03,
                ],
                [
                    -1.6537822527518407e-05,
                    -3.4268680513403798e-05,
                    +1.7429030254374785e-04,
                ],
                [
                    -7.8448349402138331e-05,
                    -1.0770637122240939e-04,
                    -6.3035093160611531e-05,
                ],
                [
                    -1.0110249312850744e-03,
                    -7.5952175228317324e-04,
                    -6.2001222160320635e-04,
                ],
                [
                    -5.1935617552229425e-03,
                    +2.0076885550501626e-03,
                    -5.2221630134030904e-03,
                ],
                [
                    +1.4239939617592678e-04,
                    +4.8767812091113708e-04,
                    -5.0769013976375045e-03,
                ],
                [
                    -2.9508042967427745e-03,
                    -1.0666291927900759e-03,
                    +9.0580572947990962e-03,
                ],
                [
                    +6.7494834706915106e-04,
                    -6.6684447848499543e-04,
                    -3.9681377819825517e-04,
                ],
                [
                    +3.9727845837011881e-03,
                    -1.7093499042606991e-03,
                    +2.0056409654162499e-03,
                ],
                [
                    -6.1100935167253878e-03,
                    -3.1817757178165471e-03,
                    -9.4534227005632597e-03,
                ],
                [
                    +3.2290957603866004e-03,
                    +1.2229365929369219e-02,
                    -9.7694611225466570e-03,
                ],
                [
                    +5.9137014008744070e-03,
                    -8.2934321277995694e-03,
                    +1.5230600607422946e-02,
                ],
                [
                    +3.1104208074774929e-03,
                    +4.4140682408282681e-03,
                    -4.6279173645474153e-03,
                ],
                [
                    +1.6049196664293624e-03,
                    +1.3686425408238178e-03,
                    +4.5474131062944274e-03,
                ],
            ],
            dtype=torch.float64,
        ),
    },
    "MB16_43_08": {
        "es2": torch.tensor(0.11889887832100766, dtype=torch.float64),
        "es3": torch.tensor(0.0, dtype=torch.float64),
        "q": torch.tensor(  # shell-resolved charges
            [
                +9.0625989437103271e-01,
                -1.1173082590103149e00,
                +2.7801734209060669e-01,
                -7.8002899885177612e-01,
                +1.1135281324386597e00,
                -6.9829005002975464e-01,
                +2.0394323766231537e-01,
                -5.2990281581878662e-01,
                +4.3821994215250015e-02,
                -1.8674632534384727e-02,
                +4.6599645167589188e-02,
                +4.9759081006050110e-01,
                -2.5044196844100952e-01,
                +4.8329543322324753e-02,
                -2.2655924782156944e-02,
                +4.5033197849988937e-02,
                -2.1156933158636093e-02,
                +3.1247061491012573e-01,
                -9.1558927297592163e-01,
                +1.0639426708221436e00,
                -6.7195236682891846e-01,
                +1.8232247829437256e00,
                -9.2611002922058105e-01,
                +9.7835713624954224e-01,
                -7.8482419252395630e-01,
                -9.4354927539825439e-02,
                -8.7813399732112885e-03,
                -7.0778317749500275e-02,
                -3.3669296652078629e-02,
                +6.7537510395050049e-01,
                -7.0185703039169312e-01,
                +2.1159812808036804e-01,
                -6.0171592235565186e-01,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor(
            [
                [
                    -6.1303949640299630e-04,
                    +5.6997424253391006e-04,
                    -5.9761215514237147e-04,
                ],
                [
                    +7.9447629147067871e-03,
                    +4.8329528663158955e-03,
                    -2.4571229061198276e-03,
                ],
                [
                    -7.8769177629164355e-03,
                    -1.4236560218130242e-03,
                    +1.4062119424323454e-03,
                ],
                [
                    +4.1863463482451485e-03,
                    +3.1399502005168781e-03,
                    -7.6772246436691090e-04,
                ],
                [
                    +1.9745391960928424e-05,
                    -4.8379805376245314e-05,
                    +1.6629973356697525e-04,
                ],
                [
                    -1.4686986097165072e-03,
                    +1.3211106120352666e-03,
                    -2.4216869372706389e-03,
                ],
                [
                    -9.7786453874435176e-05,
                    -2.4248352683219781e-04,
                    +1.6129397594526418e-04,
                ],
                [
                    -8.1112450511949095e-05,
                    +1.5064705311030092e-04,
                    +1.8616245651508536e-04,
                ],
                [
                    -2.8763665590047678e-03,
                    -1.1285940573681399e-02,
                    -3.1867585750163033e-03,
                ],
                [
                    -3.2669189557977115e-03,
                    +4.2727651319202312e-03,
                    +6.9944207551659310e-03,
                ],
                [
                    -3.3363453740409624e-05,
                    +2.7151841302514470e-03,
                    -2.2520863770750590e-04,
                ],
                [
                    +2.9286376619121868e-04,
                    -1.1782227791104893e-03,
                    +5.0858797541698877e-03,
                ],
                [
                    -2.2155595038014293e-04,
                    -8.0520266415374272e-04,
                    -4.4325245991296100e-04,
                ],
                [
                    +2.7991341328202331e-04,
                    -2.1115535160637008e-04,
                    -3.7315590376107066e-04,
                ],
                [
                    +6.1179573417370354e-06,
                    -7.8912541186972602e-05,
                    +1.9575278939680505e-04,
                ],
                [
                    +3.8060099006175102e-03,
                    -1.7286309729234747e-03,
                    -3.7235013678946815e-03,
                ],
            ],
            dtype=torch.float64,
        ),
    },
    "LiH": {
        "es2": torch.tensor(0.0, dtype=torch.float64),
        "es3": torch.tensor(0.0, dtype=torch.float64),
        "q": torch.tensor(
            [
                +6.1767599778670401e-01,
                -3.4157592431816863e-01,
                -2.7789560584675121e-01,
                +1.7955323782162661e-03,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor(
            [
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    -2.6068304143608591e-03,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +2.6068304143608591e-03,
                ],
            ],
            dtype=torch.float64,
        ),
    },
    "SiH4": {
        "es2": torch.tensor(4.3803610149365790e-003),
        "es3": torch.tensor(-2.9771152185276151e-005),
        "q": torch.tensor(  # shell-resolved charges
            [
                +6.5663937010219886e-01,
                +3.5838834166483924e-02,
                -4.1737062303296402e-01,
                -7.1859260070046282e-02,
                +3.0823647611152977e-03,
                -7.1859260070046060e-02,
                +3.0823647611153662e-03,
                -7.1859260070045838e-02,
                +3.0823647611152409e-03,
                -7.1859260070045838e-02,
                +3.0823647611152799e-03,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor(
            [
                [
                    -3.0730355326386016e-18,
                    -2.3479753297889205e-18,
                    +4.6993387913668583e-18,
                ],
                [
                    +3.8594593382759371e-04,
                    +3.8594593382759365e-04,
                    -3.8594593382759382e-04,
                ],
                [
                    -3.8594593382759203e-04,
                    -3.8594593382759208e-04,
                    -3.8594593382759273e-04,
                ],
                [
                    +3.8594593382759165e-04,
                    -3.8594593382759111e-04,
                    +3.8594593382759083e-04,
                ],
                [
                    -3.8594593382759089e-04,
                    +3.8594593382759143e-04,
                    +3.8594593382759078e-04,
                ],
            ],
            dtype=torch.float64,
        ),
    },
}


extra: dict[str, Record] = {
    "SiH4_atom": {
        "es2": torch.tensor(5.0778974565885598e-004, dtype=torch.float64),
        "es3": torch.tensor(-2.9771152185276151e-005, dtype=torch.float64),
        "numbers": mols["SiH4"]["numbers"],
        "positions": mols["SiH4"]["positions"],
        "q": torch.tensor(
            [
                -8.41282505804719e-2,
                2.10320626451180e-2,
                2.10320626451178e-2,
                2.10320626451179e-2,
                2.10320626451179e-2,
            ],
            dtype=torch.float64,
        ),
        "grad": torch.tensor([], dtype=torch.float64),
    },
}


samples: dict[str, Record] = {**merge_nested_dicts(mols, refs), **extra}
