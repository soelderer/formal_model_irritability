from dash import html
import config

parameter_dict = {
    "iteration": {
        "utf8-name": "iteration",
        "range": "'Expected' or 'Iteration i'",
        "interpretation": (
            "Whether to display the average over many model runs ('Expected') "
            "or the results of a single agent"
        ),
    },

    "theta_A_w1": {
        "utf8-name": "θ_A_w1",
        "range": "ℝ",
        "interpretation": (
            "Reactive aggression gain: logit-linear scaling of "
            "aggressive behavior as a function of current "
            "anger/frustration"
        ),
    },

    "C": {
        "utf8-name": "C",
        "range": "[0, 1]",
        "interpretation": "Perceived controllability of the environment",
    },

    "lambda_A": {
        "utf8-name": "λ_A",
        "range": "[0, 1]",
        "interpretation":
        "Affective inertia: higher values → slower emotion updates",
    },

    "eta": {
        "utf8-name": "η",
        "range": "[0, 1]",
        "interpretation":
        "Learning rate: how quickly value expectations update",
    },

    "gamma": {
        "utf8-name": "γ",
        "range": "[0, 1]",
        "interpretation":
        "Discount factor: weight given to future rewards",
    },

    "alpha": {
        "utf8-name": "α",
        "range": "[0, 1]",
        "interpretation": (
            "Relative weighting of prediction errors versus "
            "absolute rewards as affective inputs (1 => only RPE, 0 => "
            "only absolute rewards)"
        ),
    },

    "kappa": {
        "utf8-name": "κ",
        "range": ">1",
        "interpretation": (
            "Negativity bias: negative affective inputs are amplified "
            "relative to positive ones"
        ),
    },

    "lambda_C": {
        "utf8-name": "λ_C",
        "range": "[0, 1]",
        "interpretation": (
            "Speed at which perceived controllability changes across steps"
        ),
    },

    "midpoint_C": {
        "utf8-name": "midpoint_C",
        "range": ">1",
        "interpretation": (
            "Step number at which C reaches 50% between start and stop "
            "value (inflection point of the decay)"
        ),
    },

    "w_v_A": {
        "utf8-name": "w_v_A",
        "range": ">= 0",
        "interpretation":
        "Contribution of frustration/anger to response vigor",
    },

    "I": {
        "utf8-name": "I",
        "range": "[0, 1]",
        "interpretation": "Degree of response inhibition",
    },

    "C_start": {
        "utf8-name": "C_start",
        "range": "[0, 1]",
        "interpretation": "Start value of perceived controllability",
    },

    "C_end": {
        "utf8-name": "C_end",
        "range": "[0, 1]",
        "interpretation": "End value of perceived controllability",
    },

    "I_start": {
        "utf8-name": "I_start",
        "range": "[0, 1]",
        "interpretation": "Start value of response inhibition",
    },

    "I_end": {
        "utf8-name": "I_end",
        "range": "[0, 1]",
        "interpretation": "End value of response inhibition",
    },

    "lambda_I": {
        "utf8-name": "λ_I",
        "range": "[0, 1]",
        "interpretation": (
            "Speed at which response inhibition changes across steps"
        ),
    },

    "midpoint_I": {
        "utf8-name": "midpoint_I",
        "range": ">1",
        "interpretation": (
            "Step number at which I reaches 50% between start and stop "
            "value (inflection point of the decay)"
        ),
    },
}


def parameter_table(param_list: list[str]) -> html.Div:
    trs = [
        html.Tr([
            html.Td(name, style={"padding": "0 12px"}),
            html.Td(parameter_dict[name]["range"],
                    style={"padding": "0 12px"}),
            html.Td(parameter_dict[name]["interpretation"],
                    style={"padding": "0 12px"}),
        ])
        for name in param_list]

    # Append bottom rule to last element
    trs[-1] = html.Tr(
        trs[-1].children,
        style={**config.bottomrule},
    )

    table = html.Div([
        html.Table(
            [
                html.Tr(
                    [
                        html.Th("Parameter", style={
                            "padding": "0 12px"}),
                        html.Th("Range", style={"padding": "0 12px"}),
                        html.Th("Interpretation", style={
                            "padding": "0 12px"}),
                    ],
                    style={**config.toprule, **config.midrule},
                ),
                *trs
            ],
            style=config.table_style,
        )
    ], style={"paddingTop": "10px"})

    return table


def create_tooltip_text(param: str):
    if param in parameter_dict.keys():
        return [
            html.P([parameter_dict[param]["interpretation"]]),
            html.P([
                f"Range: {parameter_dict[param]["range"]}",
            ]),
        ]

    else:
        return []
