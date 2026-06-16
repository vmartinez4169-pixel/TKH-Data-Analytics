def make_output():
    """Create and return an Output widget."""
    from ipywidgets import Output
    return Output()


def make_slider(min_val, max_val, step, value, description, width="420px"):
    """Create a styled FloatSlider."""
    import ipywidgets as widgets
    from ipywidgets import FloatSlider
    return FloatSlider(
        value=value,
        min=min_val,
        max=max_val,
        step=step,
        description=description,
        style={"description_width": "120px"},
        layout=widgets.Layout(width=width),
    )


def make_int_slider(min_val, max_val, step, value, description, width="420px"):
    """Create a styled IntSlider."""
    import ipywidgets as widgets
    from ipywidgets import IntSlider
    return IntSlider(
        value=value,
        min=min_val,
        max=max_val,
        step=step,
        description=description,
        style={"description_width": "120px"},
        layout=widgets.Layout(width=width),
    )


def make_dropdown(options, value, description, width="420px"):
    """Create a styled Dropdown."""
    import ipywidgets as widgets
    from ipywidgets import Dropdown
    return Dropdown(
        options=options,
        value=value,
        description=description,
        style={"description_width": "120px"},
        layout=widgets.Layout(width=width),
    )


def make_toggle(options, value, description):
    """Create a styled ToggleButtons widget."""
    from ipywidgets import ToggleButtons
    return ToggleButtons(
        options=options,
        value=value,
        description=description,
        style={"description_width": "120px"},
    )


def display_widget(controls, output):
    """Display controls above output in a VBox."""
    from ipywidgets import VBox
    from IPython.display import display
    if isinstance(controls, list):
        controls = VBox(controls)
    display(VBox([controls, output]))


def register_observer(widgets_list, callback):
    """Register the same callback on multiple widgets."""
    for w in widgets_list:
        w.observe(callback, names="value")
