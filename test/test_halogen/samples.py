"""Samples for test of halogen bond correction."""

from __future__ import annotations
import torch

from xtbml.typing import Molecule, Tensor
from xtbml.utils import symbol2number


class Record(Molecule):
    """Format of reference records."""

    energy: Tensor
    """Reference value for energy from halogen bond correction."""


samples: dict[str, Record] = {
    "br2nh3": {
        "numbers": symbol2number(["Br", "Br", "N", "H", "H", "H"]),
        "positions": torch.tensor(
            [
                [+0.00000000000000, +0.00000000000000, +3.11495251300000],
                [+0.00000000000000, +0.00000000000000, -1.25671880600000],
                [+0.00000000000000, +0.00000000000000, -6.30201130100000],
                [+0.00000000000000, +1.78712709700000, -6.97470840000000],
                [-1.54769692500000, -0.89356260400000, -6.97470840000000],
                [+1.54769692500000, -0.89356260400000, -6.97470840000000],
            ],
        ),
        "energy": torch.tensor(2.4763110097465683e-3),
    },
    "br2nh2o": {
        "numbers": symbol2number(["Br", "Br", "N", "H", "H", "O"]),
        "positions": torch.tensor(
            [
                [+0.00000000000000, +0.00000000000000, +3.11495251300000],
                [+0.00000000000000, +0.00000000000000, -1.25671880600000],
                [+0.00000000000000, +0.00000000000000, -6.30201130100000],
                [+0.00000000000000, +1.78712709700000, -6.97470840000000],
                [-1.54769692500000, -0.89356260400000, -6.97470840000000],
                [+1.54769692500000, -0.89356260400000, -6.97470840000000],
            ]
        ),
        "energy": torch.tensor(1.0010592532310653e-003),
    },
    "br2och2": {
        "numbers": symbol2number(["Br", "Br", "O", "C", "H", "H"]),
        "positions": torch.tensor(
            [
                -1.78533374700000,
                -3.12608299900000,
                0.00000000000000,
                0.00000000000000,
                0.81604226400000,
                0.00000000000000,
                2.65828699900000,
                5.29707580600000,
                0.00000000000000,
                4.88597158600000,
                4.86116137300000,
                0.00000000000000,
                5.61550975300000,
                2.90822215900000,
                0.00000000000000,
                6.28907612600000,
                6.39963643500000,
                0.00000000000000,
            ],
        ).reshape((-1, 3)),
        "energy": torch.tensor(-6.7587305781592112e-4),
    },
    "finch": {
        "numbers": symbol2number(["F", "I", "N", "C", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                0.00000000000000,
                4.37637862700000,
                0.00000000000000,
                0.00000000000000,
                0.69981844700000,
                0.00000000000000,
                0.00000000000000,
                -4.24181123900000,
                0.00000000000000,
                0.00000000000000,
                -6.39520691700000,
                0.00000000000000,
                0.00000000000000,
                -8.41387269200000,
            ],
        ).reshape((-1, 3)),
        "energy": torch.tensor(1.1857937381795408e-2),
    },
    "tmpda": {
        "numbers": symbol2number(
            [
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "F",
                "I",
                "I",
                "I",
                "N",
                "C",
                "C",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
                "C",
                "H",
                "H",
                "C",
                "H",
                "H",
                "N",
                "H",
                "H",
                "C",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "H",
            ]
        ),
        "positions": torch.tensor(
            [
                [0.5290878, 1.2013531, 2.2700441],
                [
                    1.3161209,
                    0.0507214,
                    2.3239534,
                ],
                [
                    0.6223940,
                    -1.1595194,
                    2.3736772,
                ],
                [
                    -0.7685078,
                    -1.2647904,
                    2.3639976,
                ],
                [
                    -1.4744377,
                    -0.0650938,
                    2.3059391,
                ],
                [
                    -0.8667556,
                    1.1909427,
                    2.2412522,
                ],
                [
                    -1.6478959,
                    2.4482878,
                    2.1800916,
                ],
                [
                    -1.6343150,
                    3.3163383,
                    3.2738759,
                ],
                [
                    -2.3370934,
                    4.5191279,
                    3.2717126,
                ],
                [
                    -3.0798735,
                    4.8731320,
                    2.1492348,
                ],
                [
                    -3.1090458,
                    4.0231925,
                    1.0444267,
                ],
                [
                    -2.4006738,
                    2.8250420,
                    1.0521578,
                ],
                [
                    -1.4554005,
                    -2.5773075,
                    2.4728359,
                ],
                [
                    -1.8794846,
                    -3.3087945,
                    1.3501617,
                ],
                [
                    -2.5160427,
                    -4.5322786,
                    1.5338947,
                ],
                [
                    -2.7396783,
                    -5.0486298,
                    2.8091412,
                ],
                [
                    -2.3210977,
                    -4.3306654,
                    3.9255251,
                ],
                [
                    -1.6849255,
                    -3.1054668,
                    3.7437103,
                ],
                [
                    2.7974930,
                    0.1052463,
                    2.3657096,
                ],
                [
                    3.5738583,
                    0.5124953,
                    1.2651516,
                ],
                [
                    4.9590315,
                    0.5446458,
                    1.3823259,
                ],
                [
                    5.5966983,
                    0.1764207,
                    2.5658240,
                ],
                [
                    4.8362413,
                    -0.2339276,
                    3.6567451,
                ],
                [
                    3.4489314,
                    -0.2648897,
                    3.5439542,
                ],
                [
                    1.1486090,
                    2.3979578,
                    2.2468071,
                ],
                [
                    -2.8213958,
                    -0.1191155,
                    2.3332240,
                ],
                [
                    1.3392361,
                    -2.2986085,
                    2.4302239,
                ],
                [
                    -0.9371683,
                    2.9951050,
                    4.3780669,
                ],
                [
                    -2.3073302,
                    5.3287079,
                    4.3380634,
                ],
                [
                    -3.7593254,
                    6.0274999,
                    2.1330375,
                ],
                [
                    -3.8292371,
                    4.4026522,
                    -0.0250375,
                ],
                [
                    -2.9374890,
                    -5.2640344,
                    0.4827576,
                ],
                [
                    -3.3540400,
                    -6.2307782,
                    2.9684280,
                ],
                [
                    -2.5319117,
                    -4.8175816,
                    5.1572286,
                ],
                [
                    -1.2905847,
                    -2.4258816,
                    4.8393372,
                ],
                [
                    5.7366659,
                    0.9258618,
                    0.3483358,
                ],
                [
                    6.9347859,
                    0.2114601,
                    2.6604289,
                ],
                [
                    5.4381467,
                    -0.5873956,
                    4.8021967,
                ],
                [
                    2.7349247,
                    -0.6534853,
                    4.6193825,
                ],
                [
                    -2.4600738,
                    1.6175327,
                    -0.6577418,
                ],
                [
                    -1.5804839,
                    -2.6131434,
                    -0.6415977,
                ],
                [
                    2.6992284,
                    1.0429354,
                    -0.6072158,
                ],
                [
                    1.7595400,
                    1.8721176,
                    -3.1125965,
                ],
                [
                    2.8105299,
                    1.4358390,
                    -4.0406870,
                ],
                [
                    1.7389539,
                    3.3380404,
                    -3.0161015,
                ],
                [
                    0.4310606,
                    1.3606491,
                    -3.4932700,
                ],
                [
                    2.8666704,
                    0.3456466,
                    -4.0583812,
                ],
                [
                    2.6236843,
                    1.8022505,
                    -5.0651462,
                ],
                [
                    3.7727965,
                    1.8214787,
                    -3.6950512,
                ],
                [
                    0.9906252,
                    3.6392908,
                    -2.2786620,
                ],
                [
                    2.7200177,
                    3.6895915,
                    -2.6861906,
                ],
                [
                    1.4983812,
                    3.8101467,
                    -3.9842083,
                ],
                [
                    0.2587515,
                    -0.1356473,
                    -3.2321337,
                ],
                [
                    -0.3064854,
                    1.9132184,
                    -2.9021733,
                ],
                [
                    0.2318430,
                    1.5866051,
                    -4.5581650,
                ],
                [
                    -1.0877227,
                    -0.6121098,
                    -3.7724130,
                ],
                [
                    1.0701060,
                    -0.6941223,
                    -3.7091894,
                ],
                [
                    0.3161864,
                    -0.3289624,
                    -2.1556392,
                ],
                [
                    -1.4190038,
                    -1.9979209,
                    -3.4038086,
                ],
                [
                    -1.8898030,
                    0.0215661,
                    -3.3729522,
                ],
                [
                    -1.1108075,
                    -0.4956079,
                    -4.8731789,
                ],
                [
                    -2.7797088,
                    -2.3259584,
                    -3.8496917,
                ],
                [
                    -0.4558655,
                    -2.9689092,
                    -3.9390583,
                ],
                [
                    -3.0333933,
                    -3.3362405,
                    -3.5190230,
                ],
                [
                    -3.4863100,
                    -1.6233648,
                    -3.3994733,
                ],
                [
                    -2.8747529,
                    -2.2735036,
                    -4.9480640,
                ],
                [
                    -0.7761393,
                    -3.9740022,
                    -3.6542041,
                ],
                [
                    -0.3889685,
                    -2.9144663,
                    -5.0399768,
                ],
                [0.5326283, -2.7908957, -3.5118282],
            ]
        ),
        "energy": torch.tensor(7.6976121430560651e-002),
    },
    "tmpda_mod": {
        "numbers": symbol2number(
            [
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "c",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "f",
                "i",
                "i",
                "br",
                "n",
                "c",
                "c",
                "c",
                "h",
                "h",
                "h",
                "h",
                "h",
                "h",
                "c",
                "h",
                "h",
                "c",
                "o",
                "h",
                "n",
                "h",
                "h",
                "c",
                "c",
                "h",
                "h",
                "h",
                "h",
                "h",
                "h",
                "h",
            ]
        ),
        "positions": torch.tensor(
            [
                [
                    0.99983141582600,
                    2.27022814899754,
                    4.28976145084998,
                ],
                [
                    2.48710823686859,
                    0.09584879876714,
                    4.39163469650243,
                ],
                [
                    1.17615420161033,
                    -2.19117334629995,
                    4.48559943832332,
                ],
                [
                    -1.45226964458390,
                    -2.39010670516543,
                    4.46730877916307,
                ],
                [
                    -2.78628400774097,
                    -0.12300983235639,
                    4.35759317009342,
                ],
                [
                    -1.63793145687613,
                    2.25055610004019,
                    4.23535245626976,
                ],
                [
                    -3.11407212186631,
                    4.62659379420778,
                    4.11977680648764,
                ],
                [
                    -3.08840775136777,
                    6.26697055668918,
                    6.18672900598532,
                ],
                [
                    -4.41646569778001,
                    8.53991424212781,
                    6.18264152837775,
                ],
                [
                    -5.82011835835567,
                    9.20888484914983,
                    4.06146552746007,
                ],
                [
                    -5.87524544886325,
                    7.60273291651153,
                    1.97368098716452,
                ],
                [
                    -4.53661637450985,
                    5.33855567056503,
                    1.98829045983400,
                ],
                [
                    -2.75030929150647,
                    -4.87040436908087,
                    4.67298279111509,
                ],
                [
                    -3.55171190534227,
                    -6.25271446280500,
                    2.55143640387698,
                ],
                [
                    -4.75463218778180,
                    -8.56476603039276,
                    2.89864145393285,
                ],
                [
                    -5.17724108966248,
                    -9.54052800456940,
                    5.30850713545736,
                ],
                [
                    -4.38623952841663,
                    -8.18377078750246,
                    7.41816714537158,
                ],
                [
                    -3.18404868026120,
                    -5.86848211906322,
                    7.07458659002274,
                ],
                [
                    5.28649560555772,
                    0.19888611571236,
                    4.47054399028843,
                ],
                [
                    6.75361282830281,
                    0.96847519024008,
                    2.39079078602254,
                ],
                [
                    9.37121232325518,
                    1.02923177487293,
                    2.61221755494944,
                ],
                [
                    10.57622642224080,
                    0.33338737263260,
                    4.84870464399179,
                ],
                [
                    9.13917096268626,
                    -0.44205985288146,
                    6.91024655759466,
                ],
                [
                    6.51753501273168,
                    -0.50056955315212,
                    6.69710245827200,
                ],
                [
                    2.17055643428028,
                    4.53148387835536,
                    4.24584988489205,
                ],
                [
                    -5.33166572911465,
                    -0.22509472733480,
                    4.40915434740384,
                ],
                [
                    2.53078925623932,
                    -4.34373958787379,
                    4.59245778149254,
                ],
                [
                    -1.77099085276328,
                    5.65992816449727,
                    8.27334758526197,
                ],
                [
                    -4.36022177913278,
                    10.06979871810234,
                    8.19775098137244,
                ],
                [
                    -7.10409466345877,
                    11.39032421618183,
                    4.03085574369338,
                ],
                [
                    -7.23620919628361,
                    8.31980650203590,
                    -0.04731496270838,
                ],
                [
                    -5.55104970410083,
                    -9.94758257071829,
                    0.91228040447209,
                ],
                [
                    -6.33821701103982,
                    -11.77446396334351,
                    5.60951594067063,
                ],
                [
                    -4.78462025165348,
                    -9.10391056292687,
                    9.74575037197764,
                ],
                [
                    -2.43885219055015,
                    -4.58425259065941,
                    9.14502155476810,
                ],
                [
                    10.84072760845242,
                    1.74962560919826,
                    0.65825963934764,
                ],
                [
                    13.10484627288905,
                    0.39960148631337,
                    5.02748218401201,
                ],
                [
                    10.27660845545526,
                    -1.11001756670068,
                    9.07483712649950,
                ],
                [
                    5.16825922139213,
                    -1.23490767655107,
                    8.72936873475216,
                ],
                [
                    -4.64886610631262,
                    3.05669436754430,
                    -1.24295224066360,
                ],
                [
                    -2.98668190435304,
                    -4.93812459448296,
                    -1.21244450210764,
                ],
                [
                    5.10080166792137,
                    1.97086151578658,
                    -1.14747193849076,
                ],
                [
                    3.32504870532403,
                    3.53779029298215,
                    -5.88195397660567,
                ],
                [
                    5.31113196504447,
                    2.71334246905654,
                    -7.63579178533573,
                ],
                [
                    3.28614680332248,
                    6.30798139304581,
                    -5.69960485420991,
                ],
                [
                    0.81458723300731,
                    2.57125396174593,
                    -6.60132357937146,
                ],
                [
                    5.41722118968096,
                    0.65317816579852,
                    -7.66922859938486,
                ],
                [
                    4.95804419756264,
                    3.40575890810679,
                    -9.57173872124372,
                ],
                [
                    7.12955305380973,
                    3.44209645175722,
                    -6.98263440652458,
                ],
                [
                    1.87200994220740,
                    6.87726327781544,
                    -4.30604711059201,
                ],
                [
                    5.14008907405234,
                    6.97231839161025,
                    -5.07616530843462,
                ],
                [
                    2.83152972034289,
                    7.20013432456451,
                    -7.52906194354299,
                ],
                [
                    0.48897041419917,
                    -0.25633567962711,
                    -6.10784805809119,
                ],
                [
                    -0.57917271130593,
                    3.61545803670427,
                    -5.48431213628355,
                ],
                [
                    0.43811977391161,
                    2.99824891796187,
                    -8.61368348085482,
                ],
                [
                    -2.05549856945632,
                    -1.15672025814468,
                    -7.12882739897787,
                ],
                [
                    2.02220726431878,
                    -1.31170047707749,
                    -7.00935135447454,
                ],
                [
                    0.59750494444092,
                    -0.62164808540914,
                    -4.07356733356217,
                ],
                [
                    -2.68152892974847,
                    -3.77552350863844,
                    -6.43226679053632,
                ],
                [
                    -3.57121009949615,
                    0.04075383360368,
                    -6.37395551150874,
                ],
                [
                    -2.09912100731718,
                    -0.93656338517353,
                    -9.20897366627769,
                ],
                [
                    -5.25288871615737,
                    -4.39542359738231,
                    -7.27486354416283,
                ],
                [
                    -0.86145999980253,
                    -5.61042489893657,
                    -7.44374080901614,
                ],
                [
                    -5.73228199835694,
                    -6.30458177574760,
                    -6.64998969625895,
                ],
                [
                    -6.58817108554407,
                    -3.06771525030311,
                    -6.42407293805994,
                ],
                [
                    -5.43249584594631,
                    -4.29629990324119,
                    -9.35048580712029,
                ],
                [
                    -1.46669014464092,
                    -7.50977539871506,
                    -6.90544476351199,
                ],
                [
                    -0.73504299124344,
                    -5.50754253953357,
                    -9.52417620441302,
                ],
                [
                    1.00652104630717,
                    -5.27402908231356,
                    -6.63639311679227,
                ],
                [1.97892875661261, -1.03282225450972, -8.78226972788494],
            ]
        ),
        "energy": torch.tensor(3.1574395196210699e-003),
    },
    "LYS_xao": {
        "numbers": symbol2number(
            [
                "N",
                "C",
                "C",
                "O",
                "C",
                "C",
                "H",
                "H",
                "H",
                "H",
                "H",
                "N",
                "H",
                "C",
                "H",
                "H",
                "H",
                "C",
                "O",
                "C",
                "H",
                "H",
                "H",
                "H",
                "C",
                "H",
                "H",
                "C",
                "H",
                "H",
                "N",
                "H",
                "H",
            ]
        ),
        "positions": torch.tensor(
            [
                -3.08629288118877,
                -2.19561127309795,
                -0.77217893055321,
                -1.06879606686991,
                -0.61608658924481,
                0.23634288443356,
                -1.53102984802615,
                2.13141574530499,
                -0.56939167294673,
                -0.40983742362732,
                3.10958800397462,
                -2.34867178991298,
                1.47971484502131,
                -1.55589129675000,
                -0.67464620457756,
                3.67080881782305,
                -0.22160454696147,
                0.62879195584726,
                -2.83661151752225,
                -2.99270493895025,
                -2.48021099590057,
                -1.20083247698939,
                -0.74777619088370,
                2.28964206874229,
                1.58937150127564,
                -3.58727643425421,
                -0.32481723884977,
                3.54409665680710,
                -0.53132058661176,
                2.66860726830112,
                3.50974930263916,
                1.80833214591191,
                0.30677605183608,
                -3.30361007773525,
                3.31485339209112,
                0.83839008978212,
                -4.34880376214547,
                2.20488174175011,
                1.99552260527961,
                -4.18868179160437,
                5.81154501452241,
                0.15747382156164,
                -5.61569230213706,
                5.74761437268114,
                -1.32808061177308,
                -2.59583748873156,
                6.90533962601390,
                -0.53230790566849,
                -4.98328471638590,
                6.72600829931603,
                1.81417357766629,
                -5.35207705391596,
                -2.42418436069169,
                0.40031295493727,
                -5.84409781413938,
                -1.31026328950066,
                2.38293392048313,
                -7.23017418928331,
                -4.16191765723655,
                -0.85113016451490,
                -7.68462505102300,
                -5.67776148707435,
                0.46000324484997,
                -6.54293162706832,
                -4.97137068473344,
                -2.60950586255879,
                -8.95780193409875,
                -3.11008592668001,
                -1.21083256960746,
                1.60633375878005,
                -1.24066001627663,
                -2.71011415264373,
                6.22896524570903,
                -1.13514352758590,
                -0.31993549624624,
                6.36077259797159,
                -0.80721133290823,
                -2.35609502036389,
                6.39494201117646,
                -3.17620769071747,
                -0.03322800857001,
                8.43206186995541,
                0.18178876655507,
                0.98995969125296,
                8.26394746388151,
                2.21871263547301,
                0.71486715223124,
                8.30590057110490,
                -0.13981983839829,
                3.02371506696762,
                10.95680155885510,
                -0.57943795611462,
                0.16732903266954,
                11.12834218297359,
                -0.27087148385594,
                -1.71428160151084,
                11.15655125465616,
                -2.46919075399082,
                0.40005961272418,
            ]
        ).reshape((-1, 3)),
        "energy": torch.tensor(0.0),
    },
}
