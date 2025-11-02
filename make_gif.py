import random
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.animation as animation
from IPython.display import HTML  # For displaying in Jupyter

# --- Helper dictionaries from your notebook ---
color_strings = {
    0: "red",
    1: "green",
    2: "blue"
}
color_hexs = {
    0: [0xFF, 0x00, 0x00],
    1: [0x00, 0xFF, 0x00],
    2: [0x00, 0x00, 0xFF]
}
color_one_hots = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1]
}


# --- Cell Class Definition (from your notebook) ---
class Cell():
    def __init__(self, hidden_layers=1, hidden_size=16, activation=nn.ReLU, row=0, col=0):
        self.row_num = row
        self.col_num = col
        self.hp = 50
        self.color = random.randint(0, 2)
        self.hidden_size = hidden_size
        self.next_throw = self.color
        self.neighbors = []
        self.brain = nn.Sequential(
            nn.Linear(15, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )
        self.optimizer = optim.SGD(self.brain.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()
        self.choice_dist = torch.tensor([0.333, 0.333, 0.333], dtype=torch.float, requires_grad=True)

    def make_choice(self, sense_data):
        # use the neural net to choose what color to throw next turn
        vect_sense = torch.reshape(sense_data, (-1,))
        self.choice_dist = self.brain(vect_sense)
        self.next_throw = torch.argmax(self.choice_dist)
        return self.choice_dist

    def learn(self, optimal_throw_tensor):
        # calculate gradients for parameters based on loss
        self.loss_object = self.loss(self.choice_dist, optimal_throw_tensor)
        self.loss_object.backward()

    def update(self):
        # alter weights and biases of neural net
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evaluate(self, my_color_one_hot=None, sense_data=None):
        num_red_neighbors = 0
        num_green_neighbors = 0
        num_blue_neighbors = 0
        for neighbor in self.neighbors:
            if neighbor.color == 0:
                num_red_neighbors += 1
            if neighbor.color == 1:
                num_green_neighbors += 1
            if neighbor.color == 2:
                num_blue_neighbors += 1
        if my_color_one_hot is None:
            self_is_red, self_is_green, self_is_blue = self.color_one_hot()
        else:
            self_is_red, self_is_green, self_is_blue = my_color_one_hot
        wins = 3 * (self_is_red * num_green_neighbors) + 2 * (self_is_green * num_blue_neighbors) + (
                    self_is_blue * num_red_neighbors)
        # consider different values for different kinds of wins, some choices are high risk high reward, others are low risk low reward.
        defeats = 3 * (self_is_red * num_blue_neighbors) + 2 * (self_is_green * num_red_neighbors) + (
                    self_is_blue * num_green_neighbors)
        score = wins - defeats  # UNCOMMENT THIS real code
        # score = 4 if self_is_red == 1 else -4 # test code
        return score

    def find_optimal_throw(self, sense_data):
        # determine what would have been the best color to throw this turn
        throw_options = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        optimal_score = -np.inf
        for throw in throw_options:
            score = self.evaluate(my_color_one_hot=throw, sense_data=sense_data)
            if score > optimal_score:
                optimal_score = score
                best_throw = throw
        optimal_throw_tensor = torch.tensor(best_throw, dtype=torch.float, requires_grad=True)
        return optimal_throw_tensor

    def change_hp(self, hp_bonus):
        self.hp += hp_bonus

    def set_color(self, color_num):
        self.color = int(color_num)

    def get_color(self):
        """
        MODIFIED: This function now clamps HP to the 0-100 range for brightness
        to prevent the Matplotlib clipping error.
        It returns the color as a uint8 array.
        """
        # Clamp HP to a 0-100 range for brightness calculation
        brightness = max(0, min(100, self.hp))
        color_val = int(brightness / 100 * 255)
        color_array = np.array(self.color_one_hot()) * color_val
        # Return as uint8, which imshow expects for RGB data
        return color_array.astype(np.uint8)

    def color_one_hot(self):
        return color_one_hots[self.color]

    def get_lr(self):
        # don't use yet, just a random idea i might use later
        return math.cos((math.pi / 2) * self.hp)

    def __repr__(self):
        return f" {color_strings[self.color]} Hp:{self.hp:.2f} "
        # print the state of the cell

    def sense(self):
        sense_data_list = [self.color_one_hot()]
        for neighbor in self.neighbors:
            sense_data_list.append(neighbor.color_one_hot())
        # sense data list is [self_color, north_neighbor, south_neighbor, west_neighbor, east_neighbor]
        sense_tensor = torch.tensor(sense_data_list, dtype=torch.float, requires_grad=True)
        return sense_tensor


# --- Board Class Definition (from your notebook) ---
class Board():
    def __init__(self, rows, cols):
        self.cells = []
        self.rows = rows
        self.cols = cols
        self.num_turns = 0
        for row in range(rows):
            new_row = []
            for col in range(cols):
                new_row.append(Cell(row=row, col=col, hidden_size=np.random.randint(1, 64)))
            self.cells.append(new_row)
        # Assign neighbors
        for row in range(rows):
            for col in range(cols):
                if self.rows > 1:
                    if row == 0:
                        self.cells[row][col].neighbors.append(self.cells[self.rows - 1][col])
                        self.cells[row][col].neighbors.append(self.cells[row + 1][col])
                    if (row > 0) and (row < self.rows - 1):
                        self.cells[row][col].neighbors.append(self.cells[row - 1][col])
                        self.cells[row][col].neighbors.append(self.cells[row + 1][col])
                    if row == self.rows - 1:
                        self.cells[row][col].neighbors.append(self.cells[row - 1][col])
                        self.cells[row][col].neighbors.append(self.cells[0][col])
                if self.cols > 1:
                    if col == 0:
                        self.cells[row][col].neighbors.append(self.cells[row][self.cols - 1])
                        self.cells[row][col].neighbors.append(self.cells[row][col + 1])
                    if (col > 0) and (col < self.cols - 1):
                        self.cells[row][col].neighbors.append(self.cells[row][col - 1])
                        self.cells[row][col].neighbors.append(self.cells[row][col + 1])
                    if col == self.cols - 1:
                        self.cells[row][col].neighbors.append(self.cells[row][col - 1])
                        self.cells[row][col].neighbors.append(self.cells[row][0])

    def __repr__(self):
        board_string = ''
        for row in self.cells:
            for cell in row:
                board_string += str(cell)
                board_string += ', '
            board_string += '\n'
        return board_string

    def get_img_array(self):
        """
        NEW: Helper function to get the current board state as an RGB numpy array
        for the animation.
        """
        img_list = []
        for row in range(0, self.rows):
            img_list.append([])
            for column in range(0, self.cols):
                img_list[row].append(self.cells[row][column].get_color())
        # Create a numpy array of uint8
        img_array = np.array(img_list, dtype=np.uint8)
        return img_array

    def show_img(self):
        """ MODIFIED: Uses get_img_array() to display the image. """
        img_array = self.get_img_array()
        plt.imshow(img_array)
        plt.show()

    def step(self):
        # all cells throw the color that they picked for this step
        for row in self.cells:
            for cell in row:
                cell.set_color(cell.next_throw)
        # all cells sense states of their neighbors and their own states
        for row in self.cells:
            for cell in row:
                sense_dat = cell.sense()
                hp_bonus = cell.evaluate(sense_data=None)
                cell.change_hp(hp_bonus)
                best_throw_tensor = cell.find_optimal_throw(sense_data=None)
                cell.learn(best_throw_tensor)
                cell.update()
                cell.make_choice(sense_data=sense_dat)
        self.num_turns += 1

    def time(self, num_steps):
        for i in range(num_steps):
            self.step()


# --- Animation Logic ---
def main():
    """Main function to create and display the animation."""

    # --- Configuration ---
    ROWS = 50
    COLS = 50
    NUM_FRAMES = 200  # Number of turns to animate
    INTERVAL = 50  # Milliseconds per frame (20 fps)

    # --- Setup ---
    # This is the 'first_board' object
    board = Board(ROWS, COLS)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Fill the figure
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Turn: 0")

    # Get the initial image array
    initial_array = board.get_img_array()

    # Create the image object that will be updated
    # animated=True and blit=True are for performance
    im = ax.imshow(initial_array, animated=True)

    # --- Update Function ---
    def update_frame(i):
        """This function is called for each frame of the animation."""
        # Run one step of the simulation
        board.step()

        # Get the new image data
        img_array = board.get_img_array()

        # Update the image data
        im.set_data(img_array)

        # Update title to show frame number
        ax.set_title(f"Turn: {i + 1}")

        # Return the artists that were changed
        return [im, ax.title]

    # --- Create and Run Animation ---

    # Create the animation object
    # blit=True tells the animation to only redraw what changed
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=NUM_FRAMES,
        interval=INTERVAL,
        blit=True,
        repeat=False  # Do not loop the animation
    )

    # --- How to display ---

    # Option 1: Show in a popup window (if running as a .py script)
    # print("Displaying animation in a new window...")
    # print("Close the window to exit.")
    # plt.show()

    # Option 2: Save as a GIF (requires 'imagemagick' or 'pillow')
    # print("Saving animation as game_of_life.gif...")
    # try:
    #     ani.save('game_of_life.gif', writer='pillow', fps=1000 / INTERVAL)
    #     print("Saved successfully to game_of_life.gif.")
    # except Exception as e:
    #     print(f"Could not save GIF. Error: {e}")
    #     print("Please make sure you have 'pillow' installed: pip install pillow")

    # Option 3: Display inline in a Jupyter Notebook
    # To use this, you would run this code IN a notebook cell,
    # uncomment the following line, and remove plt.show()
    # return HTML(ani.to_jshtml())

    # (If running in a notebook, just return the jshtml)
    # return HTML(ani.to_jshtml())
    # Option 4: Save as MP4 (requires ffmpeg)
    print("Saving animation as game_of_life.mp4...")
    try:
        ani.save('game_of_life.mp4', writer='ffmpeg', fps=1000/INTERVAL)
        print("Saved successfully to game_of_life.mp4.")
    except Exception as e:
        print(f"Could not save MP4. Error: {e}")
        print("Please make sure you have 'ffmpeg' installed on your system.")

if __name__ == "__main__":
    # This block runs when the script is executed directly
    main()

