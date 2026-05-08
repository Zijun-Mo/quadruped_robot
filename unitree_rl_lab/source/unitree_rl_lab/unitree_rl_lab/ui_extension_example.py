"""Example Omniverse UI extension for the Unitree RL Lab package."""

import omni.ext


# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    """Return a small string used by the extension example."""
    print("[unitree_rl_lab] some_public_function was called with x: ", x)
    return x**x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class ExampleExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    """Minimal Omniverse extension that demonstrates UI lifecycle callbacks."""
    def on_startup(self, ext_id):
        """Build the example extension window when the extension starts."""
        print("[unitree_rl_lab] startup")

        self._count = 0

        self._window = omni.ui.Window("My Window", width=300, height=300)
        with self._window.frame:
            with omni.ui.VStack():
                label = omni.ui.Label("")

                def on_click():
                    """Update the label after the example button is clicked."""
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    """Reset the example extension click counter and label."""
                    self._count = 0
                    label.text = "empty"

                on_reset()

                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        """Release references when the example extension shuts down."""
        print("[unitree_rl_lab] shutdown")
