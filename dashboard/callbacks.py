from dash import callback, Output, Input, State, ALL
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
import json
import base64


@callback(
    Output("main-url", "hash", allow_duplicate=True),
    Input("main-url", "pathname"),
    Input({"type": "control", "page": ALL, "name": ALL}, "value"),
    State({"type": "control", "page": ALL, "name": ALL}, "id"),
    State("main-url", "hash"),
    prevent_initial_call="initial_duplicate",
)
def _update_hash(pathname, values, ids, current_hash):
    """Update the URL hash with the current app state."""

    page = pathname.strip("/")

    if not page:
        print("update_hash prevents update because no inputs at home page!")
        raise PreventUpdate

    print(f"update_hash: at page '{page}'")

    # only keep controls of current page
    state = {
        comp_id["name"]: value
        for comp_id, value in zip(ids, values)
        if comp_id["page"] == page
    }

    if not state:
        print("update_hash prevents Update because state is empty!")
        raise PreventUpdate

    print(f"update_hash: encoding state: {state}")

    new_hash = "#" + base64.b64encode(json.dumps(state).encode()).decode()

    if new_hash == current_hash:
        print("update_hash prevents Update because new_has == current_hash!")
        raise PreventUpdate

    return new_hash


def _load_hash_to_store_generic(
    hash_value,
    current_store,
    defaults,
):
    """
    Restore sliders/selectors from URL hash if present.
    If no hash or invalid, fallback to persisted values.
    """

    if not hash_value:
        # No hash → use persisted values
        print("load_hash_to_store() prevents Update! no hash")
        raise PreventUpdate

    try:
        decoded = json.loads(base64.b64decode(hash_value[1:]))
    except Exception:
        # Invalid hash → fallback to persisted values
        print("load_hash_to_store() prevents Update! invalid hash")
        raise PreventUpdate

    print(f"load_hash_to_store decoded: {decoded}")

    new_store = {**defaults, **decoded}

    if new_store == current_store:
        raise PreventUpdate
    return new_store


def _sync_sliders_generic(store):
    print(f"sync_sliders() store: {store}")

    if not store:
        print("sync_sliders() prevents update because no store!")
        raise PreventUpdate

    names = [item["id"]["name"] for item in ctx.outputs_list]

    return [store[name] for name in names]
