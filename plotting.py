import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
import time


def count_children(node):
    """
    Counts how many children a node has.

    Args:
        node: root of the subtree

    Returns:
        Number of children, -1 if node is None
    """
    if node is None:
        return -1
    if node['leaf']:
        return 0
    else:
        return 2 + count_children(node['left']) + count_children(node['right'])


size = 0.5
text_size = size/4
colour = 'green'


def plot(node, save=False, axis_equal=False, show=True):
    """
    Plots a decision tree and shows it.

    Args:
        node: Root of tree to be plotted.
        show: Displays the tree if True
        axis_equal: Keeps axis equal if True
        save: Saves the image in png file if True
    """

    def draw(_node, pos):
        """
        Draws the tree recursively. First draws the root and then calls the method on the children.

        Args:
            _node: Root of the subtree.
            pos (tuple): Position (x, y) to draw the node.
        """
        if not _node['leaf']:
            # If it is not a leaf, draws a rectangle with the attribute and value
            rectangle = patches.Rectangle(pos, size, size / 2, fc=None, facecolor=colour)
            ax.add_patch(rectangle)
            tp = TextPath((pos[0], pos[1] + size * 0.25), 'A: ' + str(_node['attribute']), size=text_size)
            plt.gca().add_patch(patches.PathPatch(tp, color="black"))
            tp = TextPath((pos[0], pos[1] + size * 0.02), 'V: ' + str(_node['value']), size=text_size)
            plt.gca().add_patch(patches.PathPatch(tp, color="black"))
        else:
            # If it is leaf, draws a circle with the class it belongs to
            circle = patches.Circle((pos[0] + size / 2, pos[1]), size / 2, fc=None, facecolor=colour)
            ax.add_patch(circle)
            tp = TextPath((pos[0] + size / 10, pos[1]), 'C: ' + str(int(_node['value'])), size=text_size)
            plt.gca().add_patch(patches.PathPatch(tp, color="black"))

        # Recursively call the children of the node with their position, then draw lines to connect them to this node
        if _node['left'] is not None:
            count = count_children(_node['left']['right']) + 1
            x = pos[0] - size * (count + 1)
            y = pos[1] - size
            draw(_node['left'], (x, y))
            x_line = [pos[0] + size / 2, x + size / 2]
            y_line = [pos[1], pos[1] - size / 2]
            plt.plot(x_line, y_line, color='Black')
        if _node['right'] is not None:
            count = count_children(_node['right']['left']) + 1
            x = pos[0] + size * (count + 1)
            y = pos[1] - size
            draw(_node['right'], (x, y))
            x_line = [pos[0] + size / 2, pos[0] + size * (count + 1) + size / 2]
            y_line = [pos[1], pos[1] - size / 2]
            plt.plot(x_line, y_line, color='Black')

    if show is False and save is False:
        return

    # Some systems require TkAgg to be able to plot, if so, import it
    # It can't be imported initially as it won't run on systems that don't need it
    try:
        fig, ax = plt.subplots()
    except AttributeError:
        import matplotlib
        matplotlib.use('TkAgg')
        fig, ax = plt.subplots()

    # In some systems it will fail to plot patches if nothing is drawn before due to Matplotlib bug
    ax.plot(0, 0, '')

    draw(node, (0, 0))
    if axis_equal:
        plt.axis('equal')

    plt.axis('off')

    if save:
        time_now = time.strftime("%Y%m%d-%H%M%S")
        plt.rcParams['savefig.dpi'] = 300
        plt.savefig('tree_{}.png'.format(time_now))
    if show:
        plt.show()
