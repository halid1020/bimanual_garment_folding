import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class KeypointGUI:
    def __init__(self, semantics):
        self.semantics = list(semantics)
        self.colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.semantics))))

    def run(self, rgb):
        # reset state each time
        self.rgb = rgb
        self.keypoints = {}
        self.click_order = 0
        self.points = []  # list of (x, y, color, name)
        self.warning_message = None
        self.warning_text = None

        # figure + main image axis
        self.fig = plt.figure(figsize=(9, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.imshow(rgb)
        self.ax.set_title("Click to assign semantic keypoints")

        # Put the legend in its own fixed axes so it never moves
        # Adjust [left, bottom, width, height] to taste.
        self.legend_ax = self.fig.add_axes([0.82, 0.55, 0.16, 0.35])
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=self.colors[i], markersize=8,
                       label=self.semantics[i])
            for i in range(len(self.semantics))
        ]
        self.legend_ax.legend(handles=handles, loc='center')
        self.legend_ax.axis('off')

        # Buttons (positions chosen to the left of legend)
        ax_undo = self.fig.add_axes([0.62, 0.02, 0.12, 0.06])
        ax_reset = self.fig.add_axes([0.75, 0.02, 0.12, 0.06])
        ax_finish = self.fig.add_axes([0.88, 0.02, 0.12, 0.06])

        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_finish = Button(ax_finish, 'Finish')

        # Connect events
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.btn_undo.on_clicked(self.undo)
        self.btn_reset.on_clicked(self.reset)
        self.btn_finish.on_clicked(self.finish)

        plt.show()
        return self.keypoints

    def onclick(self, event):
        # Only accept clicks in the main image axis
        if event.inaxes != self.ax:
            return

        if self.click_order >= len(self.semantics):
            print("All keypoints assigned. Use Undo/Reset or click Finish.")
            return

        name = self.semantics[self.click_order]
        color = self.colors[self.click_order]

        # Plot and store
        self.ax.plot(event.xdata, event.ydata, 'o', color=color, markersize=8)
        self.keypoints[name] = np.array([event.xdata, event.ydata])
        self.points.append((event.xdata, event.ydata, color, name))

        self.click_order += 1

        # If there was a warning, clear it now that user acted
        self.clear_warning()
        self.fig.canvas.draw_idle()

        if self.click_order == len(self.semantics):
            print("All keypoints assigned. Click Finish to close or close the window manually.")

    def undo(self, event):
        if self.click_order == 0:
            print("Nothing to undo.")
            return
        self.click_order -= 1
        removed_name = self.semantics[self.click_order]
        self.keypoints.pop(removed_name, None)
        self.points.pop()
        self.redraw()
        print(f"Undid {removed_name}")

    def reset(self, event):
        self.click_order = 0
        self.keypoints.clear()
        self.points.clear()
        self.warning_message = None
        self.warning_text = None
        self.redraw()
        print("Reset all keypoints.")

    def finish(self, event):
        # If not all semantics assigned, force user to reselect missing ones
        missing = [s for s in self.semantics if s not in self.keypoints]
        if missing:
            msg = "Please assign remaining: " + ", ".join(missing)
            print(msg)
            self.show_warning(msg)
            return
        # otherwise close and allow run() to return
        print("Finishing annotation...")
        plt.close(self.fig)

    def show_warning(self, text):
        # Keep the warning message text and draw it on the main axes
        self.warning_message = text
        # remove previous warning_text if any (we re-create after clears/redraw)
        if getattr(self, 'warning_text', None) is not None:
            try:
                self.warning_text.remove()
            except Exception:
                pass
            self.warning_text = None
        # place the message in axes-relative coords (bottom-center)
        self.warning_text = self.ax.text(
            0.5, 0.03, text, transform=self.ax.transAxes,
            ha='center', va='bottom',
            bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round,pad=0.4')
        )
        self.fig.canvas.draw_idle()

    def clear_warning(self):
        self.warning_message = None
        if getattr(self, 'warning_text', None) is not None:
            try:
                self.warning_text.remove()
            except Exception:
                pass
            self.warning_text = None

    def redraw(self):
        # Redraw only the main image axis (legend_ax is separate and stays put)
        self.ax.clear()
        self.ax.imshow(self.rgb)
        self.ax.set_title("Click to assign semantic keypoints")
        # redrew points
        for x, y, c, _ in self.points:
            self.ax.plot(x, y, 'o', color=c, markersize=8)
        # re-add warning message if present
        if self.warning_message:
            self.warning_text = self.ax.text(
                0.5, 0.03, self.warning_message, transform=self.ax.transAxes,
                ha='center', va='bottom',
                bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round,pad=0.4')
            )
        self.fig.canvas.draw_idle()
