import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class KeypointGUI:
    
    def __init__(self, semantics):
        self.semantics = semantics
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.semantics)))

    def run(self, rgb):
        # reset state each time
        self.rgb = rgb
        self.keypoints = {}
        self.click_order = 0
        self.points = []

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(rgb)
        self.ax.set_title("Click to assign semantic keypoints")

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=self.colors[i], markersize=8,
                              label=self.semantics[i])
                   for i in range(len(self.semantics))]
        self.ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.25, 1))

        # Buttons
        ax_undo = plt.axes([0.7, 0.02, 0.1, 0.05])
        ax_reset = plt.axes([0.81, 0.02, 0.1, 0.05])
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_reset = Button(ax_reset, 'Reset')

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.btn_undo.on_clicked(self.undo)
        self.btn_reset.on_clicked(self.reset)

        plt.show()
        return self.keypoints

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        if self.click_order >= len(self.semantics):
            print("All keypoints already assigned. Use Reset to start again.")
            return

        name = self.semantics[self.click_order]
        color = self.colors[self.click_order]

        self.ax.plot(event.xdata, event.ydata, 'o', color=color, markersize=8)
        self.keypoints[name] = np.array([event.xdata, event.ydata])
        self.points.append((event.xdata, event.ydata, color, name))

        self.click_order += 1
        self.fig.canvas.draw()

        if self.click_order == len(self.semantics):
            print("All keypoints assigned. Close window to continue.")

    def undo(self, event):
        if self.click_order == 0:
            return
        self.click_order -= 1
        self.keypoints.pop(self.semantics[self.click_order], None)
        self.points.pop()
        self.redraw()
        print(f"Undid {self.semantics[self.click_order]}")

    def reset(self, event):
        self.click_order = 0
        self.keypoints.clear()
        self.points.clear()
        self.redraw()
        print("Reset all keypoints.")

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.rgb)
        # Redraw legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=self.colors[i], markersize=8,
                              label=self.semantics[i])
                   for i in range(len(self.semantics))]
        self.ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.25, 1))
        # Redraw points
        for x, y, c, _ in self.points:
            self.ax.plot(x, y, 'o', color=c, markersize=8)
        self.fig.canvas.draw()

    def get_keypoints(self):
        plt.show()
        return self.keypoints
